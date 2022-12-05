import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from detectron2.utils.registry import Registry

QUESTION_TO_ANSWER_REGISTRY = Registry("QUESTION_TO_ANSWER")
QUESTION_TO_ANSWER_REGISTRY.__doc__ = "registry for question to answer"


@QUESTION_TO_ANSWER_REGISTRY.register()
class Question2Answer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale_factor = cfg.MODEL.QANET.QA_BRANCH.SCALE_FACTOR
        self.N = cfg.MODEL.QANET.QA_BRANCH.NUM_MASKS
        self.D = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM * 2  # make embeding dimension bigger
        self.epr_expand = nn.Conv2d(self.N, self.N*self.D, 1)  # expand the dimension of epr
        init.kaiming_normal_(self.epr_expand.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.epr_expand.bias, val=0.0)

    def forward(self, lsf, mf, ef, of, eprf):
        """
        location sensitive features: B N D
        mask features:               B D H W
        edge features:               B D H W
        object features:             B D 1
        error-prone region features: B D H W
        """
        _, _, H, W = mf.shape
        pred_masks = torch.bmm(lsf, mf.flatten(2)).view(-1, self.N, H, W)
        pred_edges = torch.bmm(lsf, ef.flatten(2)).view(-1, self.N, H, W)
        pred_eprs = torch.bmm(lsf, eprf.flatten(2)).view(-1, self.N, H, W)
        pred_eprs = self.epr_expand(pred_eprs).view(-1, self.N, self.D, H, W)
        pred_obj = torch.bmm(lsf, of)

        # large scale_factor to compute loss
        pred_masks = F.interpolate(pred_masks, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        pred_edges = F.interpolate(pred_edges, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        output = {
            "pred_masks": pred_masks,
            "pred_edges": pred_edges,
            "pred_eprs": pred_eprs,
            "pred_obj": pred_obj,
        }

        return output


def build_question2answer(cfg):
    return QUESTION_TO_ANSWER_REGISTRY.get('Question2Answer')(cfg)
