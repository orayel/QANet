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

    def forward(self, lsf, mf, ef, of, mf_auxs):
        """
        location sensitive features:  B N D
        mask features:                B D H W
        edge features:                B D H W
        object features:              B D 1
        mask features auxiliarys:     [(B D H1 W1), (B D H2 W2)]
        """
        _, _, H, W = mf.shape
        pred_masks = torch.bmm(lsf, mf.flatten(2)).view(-1, self.N, H, W)
        pred_edges = torch.bmm(lsf, ef.flatten(2)).view(-1, self.N, H, W)
        pred_mask_auxs = []
        for mf_aux in mf_auxs:
            _, _, h, w = mf_aux.shape
            pred_mask_aux = torch.bmm(lsf, mf_aux.flatten(2)).view(-1, self.N, h, w)
            pred_mask_auxs.append(pred_mask_aux)
        pred_obj = torch.bmm(lsf, of)

        # large scale_factor to compute loss
        pred_masks = F.interpolate(pred_masks, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        pred_edges = F.interpolate(pred_edges, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        for i in range(len(pred_mask_auxs)):
            pred_mask_auxs[i] = \
                F.interpolate(pred_mask_auxs[i], scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        output = {
            "pred_masks": pred_masks,
            "pred_edges": pred_edges,
            "pred_obj": pred_obj,
            "pred_masks_aux": pred_mask_auxs,
        }

        return output


def build_question2answer(cfg):
    return QUESTION_TO_ANSWER_REGISTRY.get('Question2Answer')(cfg)
