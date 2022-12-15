import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math
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
            "loss_mask_dice": cfg.MODEL.CRITERION.LOSS_MASK_DICE_WEIGHT,
            "loss_mask_bce": cfg.MODEL.CRITERION.LOSS_MASK_BCE_WEIGHT,
            "loss_obj": cfg.MODEL.CRITERION.LOSS_OBJ_WEIGHT,
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

    def loss_mask(self, outputs, targets, idxs, num_instances, input_shape):

        src_idx, _, mix_tgt_idx = idxs
        assert "pred_mask" in outputs
        src_mask = outputs["pred_mask"]
        with torch.no_grad():
            target_mask, _ = nested_masks_from_list([t["masks"].tensor for t in targets], input_shape).decompose()
        target_mask = target_mask.to(src_mask)
        if len(target_mask) == 0:
            losses = {
                "loss_mask_dice": src_mask.sum() * 0.0,
                "loss_mask_bce": src_mask.sum() * 0.0
            }
            return losses

        src_mask = src_mask[src_idx]
        target_mask = F.interpolate(target_mask[:, None], size=src_mask.shape[-2:],
                                    mode='bilinear', align_corners=False).squeeze(1)
        src_mask = src_mask.flatten(1)
        target_mask = target_mask[mix_tgt_idx].flatten(1)  # change order to abs position

        losses = {
            "loss_mask_dice": self.dice_loss(src_mask, target_mask, num_instances, reduction='mean'),
            "loss_mask_bce": F.binary_cross_entropy_with_logits(src_mask, target_mask, reduction='mean'),
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):

        loss_map = {
            "mask": self.loss_mask,
            "obj": self.loss_obj,
        }

        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    def forward(self, outputs, targets, input_shape):

        indices, k = self.matcher(outputs, targets, input_shape)
        num_masks = [len(t['masks']) for t in targets]
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values()))[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()

        losses = {}
        idxs = self.get_src_tgt_idx(indices, num_masks, k)
        for loss in self.losses:  # num_instance should * k
            losses.update(self.get_loss(loss, outputs, targets, idxs, num_instances * k, input_shape=input_shape))

        for key in losses.keys():
            if key in self.weight_dict:
                losses[key] *= self.weight_dict[key]

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

    def forward(self, outputs, targets, input_shape):
        self.cur_iter += 1
        with torch.no_grad():

            pred_mask = outputs['pred_mask'].sigmoid()
            pred_obj = outputs['pred_obj'].sigmoid()

            B, N, H, W = pred_mask.shape
            pred_mask = pred_mask.view(B * N, -1).float()
            pred_obj = pred_obj.view(B * N, -1).float()

            tgt_ids = torch.cat([v["labels"] for v in targets])
            if not tgt_ids.shape[0]:
                return [(torch.as_tensor([]).to(pred_mask), torch.as_tensor([]).to(pred_mask))] * B
            tgt_mask, _ = nested_masks_from_list([t["masks"].tensor for t in targets], input_shape).decompose()
            tgt_mask = tgt_mask.to(pred_mask)  # BitMask to float
            tgt_mask = F.interpolate(tgt_mask[:, None], size=(H, W),
                                     mode="bilinear", align_corners=False).flatten(1).float()

            with autocast(enabled=False):  # fp16
                # TODO : change score formulation
                mask_score = self.dice_score(pred_mask, tgt_mask)
                obj_score = pred_obj
                score = (mask_score ** self.alpha) * (obj_score ** (1 - self.alpha))

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
