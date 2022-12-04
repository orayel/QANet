import torch
import torch.nn as nn

from fvcore.nn.weight_init import c2_msra_fill
from detectron2.utils.registry import Registry

PREYS_GENERATE_REGISTRY = Registry("PREYS_GENERATE")
PREYS_GENERATE_REGISTRY.__doc__ = "registry for preys(mask and edge) generate module"
EPR_PREYS_GENERATE_REGISTRY = Registry("EPR_PREYS_GENERATE")
EPR_PREYS_GENERATE_REGISTRY.__doc__ = "registry for error-prone region preys generate module"


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u


'''SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation'''
class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_attention = AttentionModule(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_attention(x)
        x = self.proj_2(x)

        return x + shortcut


class PreyBranch(nn.Module):
    def __init__(self, cfg, channels):
        super().__init__()
        hidden_dim = cfg.MODEL.DECODER.HIDDEN_DIM

        self.attention = SpatialAttention(channels)
        self.projection = nn.Conv2d(channels, hidden_dim, 1)

    def forward(self, features):
        return self.projection(self.attention(features))


class ObjPreyBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.MODEL.DECODER.HIDDEN_DIM

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Linear(dim, dim)

        self.alpha, self.beta, self.gama = \
            [nn.Parameter(data=torch.ones(2), requires_grad=True) for _ in range(3)]

    def forward(self, mask_prey, edge_prey):

        mp_avg_f, mp_max_f = self.avg_pool(mask_prey), self.max_pool(mask_prey)
        ep_avg_f, ep_max_f = self.avg_pool(edge_prey), self.max_pool(edge_prey)

        alpha_exp, beta_exp, gama_exp = torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gama)
        avg_f = alpha_exp[0] / torch.sum(alpha_exp) * mp_avg_f + alpha_exp[1] / torch.sum(alpha_exp) * ep_avg_f
        max_f = beta_exp[0] / torch.sum(beta_exp) * mp_max_f + beta_exp[1] / torch.sum(beta_exp) * ep_max_f
        fuse_f = gama_exp[0] / torch.sum(gama_exp) * avg_f + gama_exp[1] / torch.sum(gama_exp) * max_f
        fuse_f = self.proj(fuse_f.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1)

        return fuse_f


@PREYS_GENERATE_REGISTRY.register()
class PreysGenerateModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS + 2  # position information cost 2 channels

        self.mask_prey_branch = PreyBranch(cfg, channels)
        self.edge_prey_branch = PreyBranch(cfg, channels)
        self.obj_prey_branch = ObjPreyBranch(cfg)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                c2_msra_fill(m)

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
        features = torch.cat([coord_features, features], dim=1)

        mask_prey = self.mask_prey_branch(features)
        edge_prey = self.edge_prey_branch(features)
        obj_prey = self.obj_prey_branch(mask_prey, edge_prey)

        return mask_prey, edge_prey, obj_prey


@EPR_PREYS_GENERATE_REGISTRY.register()
class EprPreysGenerateModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS + 2  # position information cost 2 channels

        nums_features = len(cfg.MODEL.QANET.FEATURES_ENHANCE.IN_FEATURES)
        epr_prey_branchs = [PreyBranch(cfg, channels) for _ in range(nums_features)]
        self.epr_prey_branchs = nn.ModuleList(epr_prey_branchs)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                c2_msra_fill(m)

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
        features = [torch.cat([self.compute_coordinates(feature), feature], dim=1) for feature in features]

        epr_preys = []  # large, mid, small
        for epr_prey_branch, feature in zip(self.epr_prey_branchs, features):
            epr_preys.append(epr_prey_branch(feature))
        return epr_preys


def build_preys_generate(cfg):
    return PREYS_GENERATE_REGISTRY.get('PreysGenerateModule')(cfg)


def build_epr_preys_generate(cfg):
    return EPR_PREYS_GENERATE_REGISTRY.get('EprPreysGenerateModule')(cfg)
