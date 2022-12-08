import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from kornia.morphology import erosion
from scipy.optimize import linear_sum_assignment

from detectron2.utils.registry import Registry
from .utils import nested_masks_from_list, is_dist_avail_and_initialized, get_world_size

MATCHER_REGISTRY = Registry("MATCHER")
MATCHER_REGISTRY.__doc__ = "Matcher"
CRITERION_REGISTRY = Registry("CRITERION")
CRITERION_REGISTRY.__doc__ = "Criterion"


@CRITERION_REGISTRY.register()
class Criterion(nn.Module):
    def __init__(self, cfg, matcher):
        super().__init__()
        self.matcher = matcher
        self.losses = cfg.MODEL.CRITERION.ITEMS
        self.weight_dict = self.get_weight_dict(cfg)

    @staticmethod
    def get_weight_dict(cfg):
        return {
            "loss_masks_dice": cfg.MODEL.CRITERION.LOSS_MASKS_DICE_WEIGHT,
            "loss_masks_bce": cfg.MODEL.CRITERION.LOSS_MASKS_BCE_WEIGHT,
            "loss_edges_dice": cfg.MODEL.CRITERION.LOSS_EDGES_DICE_WEIGHT,
            "loss_edges_bce": cfg.MODEL.CRITERION.LOSS_EDGES_BCE_WEIGHT,
            "loss_obj": cfg.MODEL.CRITERION.LOSS_OBJ_WEIGHT,
        }

    @staticmethod
    def get_src_tgt_idx(indices, num_masks):
        """
        indices: (([75, 26], [0, 1]), ([12, 3, 45], [0, 2, 1]), ([55, 36], [0, 1]))
        num_masks: [2, 3, 2]
        src_idx: ([0, 0, 1, 1, 1, 2, 2], [75, 26, 12, 3, 45, 55, 36])
        tgt_idx: ([0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 2, 1, 0, 1])         relative position
        mix_tgt_idx: [0, 1, 2, 4, 3, 5, 6]                              absolute position
        """

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = (batch_idx, torch.cat([src for (src, _) in indices]))
        tgt_idx = (batch_idx, torch.cat([tgt for (_, tgt) in indices]))  # relative

        # if indices tgt do not in order, like [1, 0, 2], using this to change order
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])
        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        return src_idx, tgt_idx, mix_tgt_idx

    @staticmethod
    def get_edges(targets):
        targets = targets.float()
        kernel = torch.ones((5, 5), device=targets.device)
        ero_map = erosion(targets, kernel)
        res = targets - ero_map
        return res

    @staticmethod
    def get_error_prone_region(masks, size):
        masks = masks.float()
        (h, w), (dh, dw) = masks.shape[-2:], size
        masks_down = F.interpolate(masks[:, None], (dh, dw), mode='bilinear', align_corners=False)
        masks_recover = F.interpolate(masks_down, (h, w), mode='bilinear', align_corners=False).squeeze(1)
        masks_res = (masks - masks_recover).abs()
        masks_epr = F.interpolate(masks_res[:, None], (dh, dw), mode='bilinear', align_corners=False).squeeze(1)
        masks_epr[masks_epr >= 0.01] = 1.

        return masks_epr

    @staticmethod
    def dice_loss(inputs, targets, num_instance, reduction='mean'):
        inputs = inputs.sigmoid()
        assert inputs.shape == targets.shape
        numerator = 2 * (inputs * targets).sum(1)
        denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
        loss = 1 - numerator / (denominator + 1e-4)
        if reduction == 'none':
            return loss
        return loss.sum() if reduction == 'sum' else loss.sum() / num_instance

    @staticmethod
    def loss_obj(outputs, targets, idxs, num_instances, **kwargs):
        assert "pred_obj" in outputs
        pred_obj = outputs['pred_obj'].flatten(1)
        tgt_obj = torch.zeros_like(pred_obj)
        tgt_obj[idxs[0]] = 1  # src_idx

        losses = {'loss_obj': F.binary_cross_entropy_with_logits(pred_obj, tgt_obj, reduction='mean')}
        return losses

    def loss_edges(self, outputs, targets, idxs, num_instances, input_shape):

        src_idx, _, mix_tgt_idx = idxs
        assert "pred_edges" in outputs
        src_edges = outputs["pred_edges"]
        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
        target_masks = target_masks.to(src_edges)
        target_edges = self.get_edges(target_masks.unsqueeze(1))
        if len(target_masks) == 0:
            losses = {
                "loss_edges_dice": src_edges.sum() * 0.0,
                "loss_edges_bce": src_edges.sum() * 0.0,
            }
            return losses

        src_edges = src_edges[src_idx]
        target_edges = F.interpolate(
            target_edges, size=src_edges.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        src_edges = src_edges.flatten(1)
        target_edges = target_edges[mix_tgt_idx].flatten(1)  # change order to abs position

        losses = {
            "loss_edges_dice": self.dice_loss(src_edges, target_edges, num_instances, reduction='mean'),
            "loss_edges_bce": F.binary_cross_entropy_with_logits(src_edges, target_edges, reduction='mean')
        }
        return losses

    def loss_masks(self, outputs, targets, idxs, num_instances, input_shape):

        src_idx, _, mix_tgt_idx = idxs
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]
        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
        target_masks = target_masks.to(src_masks)
        if len(target_masks) == 0:
            losses = {
                "loss_masks_dice": src_masks.sum() * 0.0,
                "loss_masks_bce": src_masks.sum() * 0.0,
            }
            return losses

        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(
            target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        src_masks = src_masks.flatten(1)
        target_masks = target_masks[mix_tgt_idx].flatten(1)  # change order to abs position

        losses = {
            "loss_masks_dice": self.dice_loss(src_masks, target_masks, num_instances, reduction='mean'),
            "loss_masks_bce": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):

        loss_map = {
            "masks": self.loss_masks,
            "edges": self.loss_edges,
            "obj": self.loss_obj,
        }

        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    def forward(self, outputs, targets, input_shape):

        indices = self.matcher(outputs, targets, input_shape)
        num_masks = [len(t['masks']) for t in targets]
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()

        losses = {}
        idxs = self.get_src_tgt_idx(indices, num_masks)
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, idxs, num_instances, input_shape=input_shape))

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses


@MATCHER_REGISTRY.register()
class Matcher(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        """
        alpha beta gama [0 - 1]
        lower value higher score (power)
        """
        self.alpha = cfg.MODEL.MATCHER.ALPHA
        self.beta = cfg.MODEL.MATCHER.BETA
        self.gama = cfg.MODEL.MATCHER.GAMA

    @staticmethod
    def dice_score(inputs, targets):
        numerator = 2 * torch.matmul(inputs, targets.t())
        denominator = (inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
        score = numerator / (denominator + 1e-4)
        return score

    @staticmethod
    def iou_score(inputs, targets):
        threshold = 0.5
        inputs, targets = (inputs >= threshold).float(), (targets >= threshold).float()
        intersection = (inputs * targets).sum(-1)
        union = targets.sum(-1) + inputs.sum(-1) - intersection
        score = intersection / (union + 1e-6)
        return score

    @staticmethod
    def get_edges(targets):
        targets = targets.float()
        kernel = torch.ones((5, 5), device=targets.device)
        ero_map = erosion(targets, kernel)
        res = targets - ero_map
        return res

    def forward(self, outputs, targets, input_shape):
        with torch.no_grad():
            pred_masks = outputs['pred_masks'].sigmoid()  # B N H W
            pred_edges = outputs['pred_edges'].sigmoid()  # B N H W
            pred_obj = outputs['pred_obj'].sigmoid()  # B N 1
            B, N, H, W = pred_masks.shape

            tgt_ids = torch.cat([v["labels"] for v in targets])
            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(pred_masks), torch.as_tensor([]).to(pred_masks))] * B
            tgt_masks, _ = nested_masks_from_list([t["masks"].tensor for t in targets], input_shape).decompose()
            tgt_masks = tgt_masks.to(pred_masks)  # BitMask to float
            tgt_masks = F.interpolate(tgt_masks[:, None], size=(H, W), mode="bilinear", align_corners=False)
            tgt_edges = self.get_edges(tgt_masks)

            pred_masks, pred_edges, pred_obj = \
                pred_masks.view(B * N, -1), pred_edges.view(B * N, -1), pred_obj.view(B * N, -1)
            tgt_masks, tgt_edges = tgt_masks.flatten(1), tgt_edges.flatten(1)
            with autocast(enabled=False):  # fp16
                pred_masks, pred_edges, pred_obj = pred_masks.float(), pred_edges.float(), pred_obj.float()
                tgt_masks, tgt_edges = tgt_masks.float(), tgt_edges.float()
                mask_score, edge_score, obj_score = \
                    self.dice_score(pred_masks, tgt_masks), self.dice_score(pred_edges, tgt_edges), pred_obj
                score = (mask_score ** self.alpha) * (edge_score ** self.beta) * (obj_score ** self.gama)

            score = score.view(B, N, -1).cpu()  # B NUM_MASK NUM_GTs
            # hungarian matching
            # TODO : do it dynamic-k times to make more positive sample?
            sizes = [len(v["masks"]) for v in targets]
            indices = [linear_sum_assignment(s[i], maximize=True)
                       for i, s in enumerate(score.split(sizes, -1))]  # split to make each sample have owner GT
            indices = [(torch.as_tensor(i, dtype=torch.int64),
                        torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            return indices


def build_criterion(cfg):
    matcher = MATCHER_REGISTRY.get('Matcher')(cfg)
    return CRITERION_REGISTRY.get('Criterion')(cfg, matcher)
