import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.utils.registry import Registry

QUESTION_TO_ANSWER_REGISTRY = Registry("QUESTION_TO_ANSWER")
QUESTION_TO_ANSWER_REGISTRY.__doc__ = "registry for question to answer"


@QUESTION_TO_ANSWER_REGISTRY.register()
class Question2Answer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale_factor = cfg.MODEL.QANET.QA_BRANCH.SCALE_FACTOR
        self.N = cfg.MODEL.QANET.QA_BRANCH.NUM_MASKS

    def forward(self, lsf, mfs, of):
        """
        location sensitive features:  B N D
        mask features:                [(B D H1 W1), (B D H2 W2) ...]
        object feature:               B D 1
        """
        pred_masks = []
        for mf in mfs:
            _, _, h, w = mf.shape
            pred_mask = torch.bmm(lsf, mf.flatten(2)).view(-1, self.N, h, w)
            pred_mask = F.interpolate(pred_mask, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
            pred_masks.append(pred_mask)
        pred_obj = torch.bmm(lsf, of)

        output = {
            "pred_masks": pred_masks,
            "pred_obj": pred_obj.flatten(1),
        }

        return output


def build_question2answer(cfg):
    return QUESTION_TO_ANSWER_REGISTRY.get('Question2Answer')(cfg)
