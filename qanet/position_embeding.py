import torch
import torch.nn as nn

from detectron2.utils.registry import Registry

POSITION_EMBEDING_REGISTRY = Registry("POSITION_EMBEDING")
POSITION_EMBEDING_REGISTRY.__doc__ = "registry for position embeding"


@POSITION_EMBEDING_REGISTRY.register()
class PositionEmbeding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.is_pos_ebd = cfg.MODEL.QANET.POSITION_EMBEDING.IS_USING

    @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features, features_aux):

        if not self.is_pos_ebd:
            return features, features_aux

        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)  # B C+2 H W

        features_aux_out = []
        for f in features_aux:
            coord_features = self.compute_coordinates(f)
            f = torch.cat([coord_features, f], dim=1)  # B C+2 H W
            features_aux_out.append(f)

        return features, features_aux_out


def build_position_embeding(cfg):
    return POSITION_EMBEDING_REGISTRY.get('PositionEmbeding')(cfg)
