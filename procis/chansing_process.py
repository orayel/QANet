import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill
from detectron2.utils.registry import Registry

CHASING_PROCESS_REGISTRY = Registry("CHASING_PROCESS")
CHASING_PROCESS_REGISTRY.__doc__ = "registry for chasing process module"


@CHASING_PROCESS_REGISTRY.register()
class ChasingProcessModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale_factor = cfg.MODEL.DECODER.SCALE_FACTOR
        self.N = cfg.MODEL.DECODER.NUM_MASKS
        self.D = cfg.MODEL.EPRDETECTOR.DIM
        self.epr_proj = nn.Conv2d(self.N, self.N*self.D, 1, groups=self.N)  # using group conv to expand dim
        c2_msra_fill(self.epr_proj)

    def forward(self, haunter, epr_preys, mask_prey, edge_prey, cls_prey):
        """
        haunter: B N D
        preys: B D H W
        cls_prey: B D 1
        """
        _, _, H, W = mask_prey.shape
        pred_masks = torch.bmm(haunter, mask_prey.flatten(2)).view(-1, self.N, H, W)
        pred_edges = torch.bmm(haunter, edge_prey.flatten(2)).view(-1, self.N, H, W)
        pred_eprs = []
        for epr_prey in epr_preys:
            _, _, h, w = epr_prey.shape
            pred_epr = torch.bmm(haunter, epr_prey.flatten(2)).view(-1, self.N, h, w)
            pred_epr = self.epr_proj(pred_epr).view(-1, self.N, self.D, h, w)
            pred_eprs.append(pred_epr)
        pred_obj = torch.bmm(haunter, cls_prey)

        pred_masks = F.interpolate(
            pred_masks, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)
        pred_edges = F.interpolate(
            pred_edges, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)

        output = {
            "pred_masks": pred_masks,
            "pred_edges": pred_edges,
            "pred_eprs": pred_eprs,
            "pred_obj": pred_obj,
        }

        return output


def build_chasing_process(cfg):
    return CHASING_PROCESS_REGISTRY.get('ChasingProcessModule')(cfg)
