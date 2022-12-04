import torch
import torch.nn as nn

from fvcore.nn.weight_init import c2_msra_fill
from detectron2.utils.registry import Registry

ANSWER_BRANCH_REGISTRY = Registry("ANSWER_BRANCH")
ANSWER_BRANCH_REGISTRY.__doc__ = "registry for answer branch"


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
        hidden_dim = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM

        self.attention = SpatialAttention(channels)
        self.projection = nn.Conv2d(channels, hidden_dim, 1)

    def forward(self, features):
        return self.projection(self.attention(features))


class ObjPreyBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM

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


@ANSWER_BRANCH_REGISTRY.register()
class AnswerBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS
        channels = channels+2 if cfg.MODEL.QANET.POSITION_EMBEDING.IS_USING else channels  # +2 position information

        self.mask_branch = PreyBranch(cfg, channels)
        self.edge_branch = PreyBranch(cfg, channels)
        self.obj_branch = ObjPreyBranch(cfg)
        self.epr_branch = PreyBranch(cfg, channels)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):

        mask_features = self.mask_branch(features)
        edge_features = self.edge_branch(features)
        obj_features = self.obj_branch(mask_prey, edge_prey)
        epr_features = self.epr_branch(features)

        return mask_features, edge_features, obj_features, epr_features

def build_answer_branch(cfg):
    return ANSWER_BRANCH_REGISTRY.get('AnswerBranch')(cfg)

