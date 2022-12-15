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

    def forward(self, lsf, mfs, ofs):
        """
        location sensitive features:  B N D
        mask features:                B D H W
        object features:              B D 1
        """

        _, _, H, W = mfs.shape
        pred_mask = torch.bmm(lsf, mfs.flatten(2)).view(-1, self.N, H, W)
        pred_obj = torch.bmm(lsf, ofs)

        # large scale_factor to compute loss
        pred_mask = F.interpolate(pred_mask, scale_factor=self.scale_factor,
                                  mode='bilinear', align_corners=False)

        output = {
            "pred_mask": pred_mask,
            "pred_obj": pred_obj,
        }
        return output


def build_question2answer(cfg):
    return QUESTION_TO_ANSWER_REGISTRY.get('Question2Answer')(cfg)
