import torch
import torch.nn as nn
from torch.nn import init

from detectron2.utils.registry import Registry

ANSWER_BRANCH_REGISTRY = Registry("ANSWER_BRANCH")
ANSWER_BRANCH_REGISTRY.__doc__ = "registry for answer branch"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


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


'''SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation'''


class MSCABlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        dim = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS
        ng = cfg.MODEL.QANET.QA_BRANCH.GNGROUPS
        mlp_ratio = cfg.MODEL.QANET.QA_BRANCH.MLPRATIO

        self.norm1 = nn.GroupNorm(num_groups=ng, num_channels=dim)
        self.attn = SpatialAttention(dim)
        self.norm2 = nn.GroupNorm(num_groups=ng, num_channels=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))

        return x


class ObjBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS
        dim = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.obj_proj = nn.Linear(in_dim, dim)

        self.alpha = nn.Parameter(data=torch.ones(2), requires_grad=True)

    def forward(self, features):
        avg_f, max_f = self.avg_pool(features), self.max_pool(features)

        alpha_exp = torch.exp(self.alpha)
        f = alpha_exp[0] / torch.sum(alpha_exp) * avg_f + alpha_exp[1] / torch.sum(alpha_exp) * max_f
        obj_features = self.obj_proj(f.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1)

        return obj_features


@ANSWER_BRANCH_REGISTRY.register()
class AnswerBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS
        in_channels = channels + 2 if cfg.MODEL.QANET.FEATURES_MERGING.IS_USING_POS else channels  # +2 position
        init_convs_num = cfg.MODEL.QANET.QA_BRANCH.INIT_CONVS
        mask_convs_num = cfg.MODEL.QANET.QA_BRANCH.MASK_CONVS
        num_blocks = cfg.MODEL.QANET.QA_BRANCH.MSCA_NUMS
        dim = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM

        self.init_conv = self.make_convs(init_convs_num, in_channels, channels)
        self.FeaturesExt = nn.Sequential(*[MSCABlock(cfg) for _ in range(num_blocks)])
        self.MaskBranch = self.make_convs(mask_convs_num, channels, dim)
        self.ObjBranch = ObjBranch(cfg)

        self.init_weights()

    @staticmethod
    def make_convs(num_convs, in_channels, out_channels):
        convs = []
        for _ in range(num_convs):
            convs.append(
                nn.Conv2d(in_channels, out_channels, 3, padding=1))
            convs.append(nn.ReLU(True))
            in_channels = out_channels
        return nn.Sequential(*convs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, val=0.0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, val=0.0)

    def forward(self, features):

        features = self.init_conv(features)
        features = self.FeaturesExt(features)
        mask_features, obj_features = self.MaskBranch(features), self.ObjBranch(features)

        return mask_features, obj_features


def build_answer_branch(cfg):
    return ANSWER_BRANCH_REGISTRY.get('AnswerBranch')(cfg)
