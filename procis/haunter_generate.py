import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill
from detectron2.utils.registry import Registry

HAUNTER_GENERATE_REGISTRY = Registry("HAUNTER_GENERATE")
HAUNTER_GENERATE_REGISTRY.__doc__ = "registry for haunter generate module"


@HAUNTER_GENERATE_REGISTRY.register()
class HaunterGenerateModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.FEATURES_ENHANCE.NUM_CHANNELS
        hidden_dim = cfg.MODEL.DECODER.HIDDEN_DIM
        num_masks = cfg.MODEL.DECODER.NUM_MASKS
        self.num_groups = cfg.MODEL.DECODER.GROUPS

        self.init_conv = nn.Conv2d(channels+2, channels, 3)  # +2 position information
        expand_dim = channels * self.num_groups
        self.iam_conv = nn.Conv2d(
            channels, num_masks*self.num_groups, 3, padding=1, groups=self.num_groups)
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.haunter = nn.Linear(expand_dim, hidden_dim)

        self.prior_prob = 0.01
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

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

    def forward(self, features):
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)  # B C+2 H W

        features = self.init_conv(features)  # B C H W
        iam = self.iam_conv(features)  # B G*N H W
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        iam_prob = iam_prob.view(B, N, -1)  # B G*N H*W
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]

        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))  # B G*N C
        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        inst_features = F.relu_(self.fc(inst_features))
        haunter = self.haunter(inst_features)
        return haunter


def build_haunter_generate(cfg):
    return HAUNTER_GENERATE_REGISTRY.get('HaunterGenerateModule')(cfg)
