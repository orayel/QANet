import torch
import torch.nn as nn
from torch.nn import init

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


class MaskBranch(nn.Module):
    def __init__(self, cfg, channels):
        super().__init__()
        hidden_dim = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM

        self.attention = SpatialAttention(channels)
        self.projection = nn.Conv2d(channels, hidden_dim, 1)

    def forward(self, features):
        return self.projection(self.attention(features))


class ObjBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.obj_proj = nn.Linear(dim, dim)

        self.alpha = nn.Parameter(data=torch.ones(2), requires_grad=True)

    def forward(self, mask_features):
        mask_avg_f, mask_max_f = self.avg_pool(mask_features), self.max_pool(mask_features)

        alpha_exp = torch.exp(self.alpha)
        f = alpha_exp[0] / torch.sum(alpha_exp) * mask_avg_f + alpha_exp[1] / torch.sum(alpha_exp) * mask_max_f
        f = self.obj_proj(f.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1)
        return f

@ANSWER_BRANCH_REGISTRY.register()
class AnswerBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS
        in_channels = channels + 2 if cfg.MODEL.QANET.POSITION_EMBEDING.IS_USING else channels
        num_convs = cfg.MODEL.QANET.QA_BRANCH.INIT_CONVS

        self.init_op = self.init_convs(num_convs, in_channels, channels)
        self.mask_branch = MaskBranch(cfg, channels)
        self.obj_branch = ObjBranch(cfg)

        self.init_weights()

    @staticmethod
    def init_convs(num_convs, in_channels, out_channels):
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
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, val=1.0)
                init.constant_(m.bias, val=0.0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, val=0.0)

    def forward(self, features):

        features = self.init_op(features)
        mask_features = self.mask_branch(features)
        obj_features = self.obj_branch(mask_features)

        return mask_features, obj_features


def build_answer_branch(cfg):
    return ANSWER_BRANCH_REGISTRY.get('AnswerBranch')(cfg)
