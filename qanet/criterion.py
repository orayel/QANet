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
            "loss_masks": cfg.MODEL.CRITERION.LOSS_MASKS_WEIGHT,
            "loss_objs": cfg.MODEL.CRITERION.LOSS_OBJS_WEIGHT,
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
    def loss_objs(outputs, targets, idxs, num_instances, **kwargs):
        assert "pred_objs" in outputs
        pred_objs = [pred_obj.flatten(1) for pred_obj in outputs['pred_objs']]
        tgt_obj = torch.zeros_like(pred_objs[0])
        tgt_obj[idxs[0]] = 1  # src_idx

        total_loss = None
        for pred_obj in pred_objs:
            if total_loss is None:
                total_loss = F.binary_cross_entropy(pred_obj.sigmoid(), tgt_obj, reduction='mean')
            else:
                total_loss += F.binary_cross_entropy(pred_obj.sigmoid(), tgt_obj, reduction='mean')

        losses = {'loss_objs': total_loss}
        return losses

    def loss_masks(self, outputs, targets, idxs, num_instances, input_shape):

        src_idx, _, mix_tgt_idx = idxs
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]
        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
        target_masks = target_masks.to(src_masks[0])
        if len(target_masks) == 0:
            losses = {
                "loss_masks": src_masks[0].sum() * 0.0
            }
            return losses

        total_loss = None
        for src_mask in src_masks:
            src_mask_ = src_mask[src_idx]
            target_mask_ = F.interpolate(target_masks[:, None], size=src_mask_.shape[-2:],
                                         mode='bilinear', align_corners=False).squeeze(1)
            src_mask_ = src_mask_.flatten(1)
            target_mask_ = target_mask_[mix_tgt_idx].flatten(1)  # change order to abs position

            if total_loss is None:
                total_loss = self.dice_loss(src_mask_, target_mask_, num_instances, reduction='mean')
            else:
                total_loss += self.dice_loss(src_mask_, target_mask_, num_instances, reduction='mean')

        losses = {
            "loss_masks": total_loss,
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):

        loss_map = {
            "masks": self.loss_masks,
            "objs": self.loss_objs,
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

    def forward(self, outputs, targets, input_shape):
        self.cur_iter += 1
        with torch.no_grad():

            pred_masks = [pred_mask.sigmoid() for pred_mask in outputs['pred_masks']]
            pred_objs = [pred_obj.sigmoid() for pred_obj in outputs['pred_objs']]

            tgt_ids = torch.cat([v["labels"] for v in targets])
            if not tgt_ids.shape[0]:
                return [(torch.as_tensor([]).to(pred_masks[0]),
                         torch.as_tensor([]).to(pred_masks[0]))] * pred_masks[0].shape[0]
            tgt_mask, _ = nested_masks_from_list([t["masks"].tensor for t in targets], input_shape).decompose()
            tgt_mask = tgt_mask.to(pred_masks[0])  # BitMask to float

            score = None
            for pred_mask, pred_obj in zip(pred_masks, pred_objs):
                B, N, H, W = pred_mask.shape

                tgt_mask_ = F.interpolate(tgt_mask[:, None], size=(H, W),
                                          mode="bilinear", align_corners=False).flatten(1).float()
                pred_mask_ = pred_mask.view(B * N, -1).float()
                pred_obj_ = pred_obj.view(B * N, -1).float()

                with autocast(enabled=False):  # fp16

                    mask_score = self.combined_score(pred_mask_, tgt_mask_)
                    obj_score = pred_obj_

                    # TODO: change score formulation
                    if score is None:
                        score = (mask_score ** self.alpha) * (obj_score ** (1 - self.alpha))
                    else:
                        score += (mask_score ** self.alpha) * (obj_score ** (1 - self.alpha))

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
