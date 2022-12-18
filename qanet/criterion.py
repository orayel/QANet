import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math
from kornia.morphology import erosion
from scipy.optimize import linear_sum_assignment

from detectron2.utils.registry import Registry
from detectron2.utils.events import get_event_storage
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
            "loss_masks_aux_dice": cfg.MODEL.CRITERION.LOSS_MASKS_AUX_DICE_WEIGHT,
        }

    @staticmethod
    def get_src_tgt_idx(indices, num_masks, k):
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
        cum_sum, cum_idx = 0, 0
        for num_mask in num_masks:  # cause each Gt match k-Samples !!
            mix_tgt_idx[cum_sum: cum_sum + num_mask * k] = cum_idx
            cum_sum += num_mask * k
            cum_idx += num_mask
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
        masks_epr[masks_epr != 1] = 0.

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

    def loss_masks_aux(self, outputs, targets, idxs, num_instances, input_shape):

        src_idx, _, mix_tgt_idx = idxs
        assert "pred_masks_aux" in outputs
        src_masks_aux = outputs["pred_masks_aux"]
        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
        target_masks = target_masks.to(src_masks_aux[0])
        if len(target_masks) == 0:
            losses = {
                "loss_masks_aux_dice": src_masks_aux[0].sum() * 0.0
            }
            return losses

        total_loss = src_masks_aux[0].sum() * 0.0
        for src_mask_aux in src_masks_aux:
            src_mask_aux_tmp = src_mask_aux[src_idx]
            target_mask_aux_tmp = F.interpolate(
                target_masks[:, None], size=src_mask_aux_tmp.shape[-2:],
                mode='bilinear', align_corners=False).squeeze(1)
            src_mask_aux_tmp = src_mask_aux_tmp.flatten(1)
            target_mask_aux_tmp = target_mask_aux_tmp[mix_tgt_idx].flatten(1)  # change order to abs position
            total_loss += self.dice_loss(src_mask_aux_tmp, target_mask_aux_tmp, num_instances, reduction='mean')

        losses = {
            "loss_masks_aux_dice": total_loss,
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):

        loss_map = {
            "masks": self.loss_masks,
            "edges": self.loss_edges,
            "obj": self.loss_obj,
            "masks_aux": self.loss_masks_aux
        }

        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    def forward(self, outputs, targets, input_shape):

        indices, k = self.matcher(outputs, targets, input_shape)
        num_masks = [len(t['masks']) for t in targets]
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()

        losses = {}
        idxs = self.get_src_tgt_idx(indices, num_masks, k)
        for loss in self.losses:  # num_instance should * k
            losses.update(self.get_loss(loss, outputs, targets, idxs, num_instances * k, input_shape=input_shape))

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
        self.dynamic_k = cfg.MODEL.MATCHER.DYNAMIC_K
        self.start_k = cfg.MODEL.MATCHER.START_K
        self.total_iter = cfg.SOLVER.MAX_ITER
        self.cur_iter = 0

    @staticmethod
    def dice_score(inputs, targets):
        numerator = 2 * torch.matmul(inputs, targets.t())
        denominator = (inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
        score = numerator / (denominator + 1e-6)
        return score

    @staticmethod
    def iou_score(inputs, targets):
        threshold = 0.5
        inputs, targets = (inputs >= threshold).float(), (targets >= threshold).float()
        intersection = torch.matmul(inputs, targets.t())
        union = targets.sum(-1)[None, :] + inputs.sum(-1)[:, None] - intersection
        score = intersection / (union + 1e-6)
        return score

    def combined_score(self, inputs, targets):
        return (self.dice_score(inputs, targets) + self.iou_score(inputs, targets)) / 2

    @staticmethod
    def get_edges(targets):
        targets = targets.float()
        kernel = torch.ones((5, 5), device=targets.device)
        ero_map = erosion(targets, kernel)
        res = targets - ero_map
        return res

    def forward(self, outputs, targets, input_shape):
        self.cur_iter += 1
        with torch.no_grad():

            pred_masks = outputs['pred_masks'].sigmoid()  # B N H W
            pred_edges = outputs['pred_edges'].sigmoid()  # B N H W
            pred_obj = outputs['pred_obj'].sigmoid()  # B N 1
            pred_masks_aux = []
            for pred_mask_aux in outputs['pred_masks_aux']:
                pred_masks_aux.append(pred_mask_aux.sigmoid())  # B N H W
            B, N, H, W = pred_masks.shape

            tgt_ids = torch.cat([v["labels"] for v in targets])
            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(pred_masks), torch.as_tensor([]).to(pred_masks))] * B

            tgt_masks, _ = nested_masks_from_list([t["masks"].tensor for t in targets], input_shape).decompose()
            tgt_masks = tgt_masks.to(pred_masks)  # BitMask to float
            tgt_masks_aux = []
            for pred_mask_aux in pred_masks_aux:
                _, _, h, w = pred_mask_aux.shape
                tgt_masks_aux.append(
                    F.interpolate(tgt_masks[:, None], size=(h, w),
                                  mode="bilinear", align_corners=False).flatten(1).float())
            tgt_masks = F.interpolate(tgt_masks[:, None], size=(H, W),
                                      mode="bilinear", align_corners=False)
            tgt_edges = self.get_edges(tgt_masks).flatten(1).float()
            tgt_masks = tgt_masks.flatten(1).float()

            pred_masks, pred_edges, pred_obj = \
                pred_masks.view(B * N, -1).float(), pred_edges.view(B * N, -1).float(), pred_obj.view(B * N, -1).float()
            for i in range(len(pred_masks_aux)):
                pred_masks_aux[i] = pred_masks_aux[i].view(B * N, -1).float()

            with autocast(enabled=False):  # fp16

                mask_score = self.combined_score(pred_masks, tgt_masks)
                # mask_score = self.dice_score(pred_masks, tgt_masks)
                edge_score = self.combined_score(pred_edges, tgt_edges)
                mask_score_aux = sum([self.combined_score(pred_mask_aux, tgt_mask_aux) for
                                      pred_mask_aux, tgt_mask_aux in zip(pred_masks_aux, tgt_masks_aux)])
                obj_score = pred_obj

                # finally score for matching
                score = self.alpha * (mask_score + obj_score) + (1 - self.alpha) * (edge_score + mask_score_aux)
                # score = (mask_score ** 0.8) * (obj_score ** 0.2)

            score = score.view(B, N, -1).cpu()  # B NUM_MASK NUM_GTs
            # hungarian matching
            sizes = [len(v["masks"]) for v in targets]

            if self.dynamic_k:  # do it dynamic-k times to make more positive sample in early time
                cnt = k = max(1, math.ceil(self.start_k * (1 - self.cur_iter / self.total_iter)))
                indices = [([], [])] * B
                while cnt:
                    indices_cur = [linear_sum_assignment(s[i], maximize=True)
                                   for i, s in enumerate(score.split(sizes, -1))]
                    # Update the score, make all the sample-a score to zero
                    for i, s in enumerate(score.split(sizes, -1)):
                        for a, _ in zip(*indices_cur[i]):
                            s[i][a][:] = 0
                    for i in range(len(indices)):
                        indices[i] = (indices[i][0] + indices_cur[i][0].tolist(),
                                      indices[i][1] + indices_cur[i][1].tolist())
                    cnt -= 1
                indices = [(torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

            else:
                k = 1  # just do matcher once
                indices = [linear_sum_assignment(s[i], maximize=True)
                           for i, s in enumerate(score.split(sizes, -1))]  # split to make each sample have owner GT
                indices = [(torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

            storage = get_event_storage()
            storage.put_scalar("k", k)
            return indices, k


def build_criterion(cfg):
    matcher = MATCHER_REGISTRY.get('Matcher')(cfg)
    return CRITERION_REGISTRY.get('Criterion')(cfg, matcher)
