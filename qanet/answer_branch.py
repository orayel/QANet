import torch
import torch.nn as nn
from torch.nn import init

from detectron2.utils.registry import Registry

ANSWER_BRANCH_REGISTRY = Registry("ANSWER_BRANCH")
ANSWER_BRANCH_REGISTRY.__doc__ = "registry for answer branch"


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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


class MSCA(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0., ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class MaskEdgeBranch(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()

        dim = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM
        drop_path_rate = cfg.MODEL.QANET.QA_BRANCH.DROP_RATE
        num_blocks = cfg.MODEL.QANET.QA_BRANCH.NUM_MSCA_BLOCKS
        mlp_ratio = cfg.MODEL.QANET.QA_BRANCH.MLP_RATIO

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.blocks = nn.ModuleList(MSCA(in_channels, mlp_ratio, dpr[i]) for i in range(num_blocks))
        self.proj = nn.Conv2d(in_channels, dim, 1)

    def forward(self, features):
        for block in self.blocks:
            features = block(features)
        features = self.proj(features)
        return features


class ObjEprBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.obj_proj = nn.Linear(dim, dim)
        self.epr_proj = nn.Conv2d(dim, dim, 1)

        self.alpha, self.beta, self.gama, self.omega_mask, self.omega_edge = \
            [nn.Parameter(data=torch.ones(2), requires_grad=True) for _ in range(5)]

    def forward(self, mask_features, edge_features):
        mask_avg_f, mask_max_f = self.avg_pool(mask_features), self.max_pool(mask_features)
        edge_avg_f, edge_max_f = self.avg_pool(edge_features), self.max_pool(edge_features)

        # object features
        alpha_exp, beta_exp, gama_exp = torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gama)
        avg_f = alpha_exp[0] / torch.sum(alpha_exp) * mask_avg_f + alpha_exp[1] / torch.sum(alpha_exp) * edge_avg_f
        max_f = beta_exp[0] / torch.sum(beta_exp) * mask_max_f + beta_exp[1] / torch.sum(beta_exp) * edge_max_f
        obj_f = gama_exp[0] / torch.sum(gama_exp) * avg_f + gama_exp[1] / torch.sum(gama_exp) * max_f
        obj_features = self.obj_proj(obj_f.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1)

        # error-prone region features
        omega_mask_exp, omega_edge_exp = torch.exp(self.omega_mask), torch.exp(self.omega_edge)
        mask_fuse = omega_mask_exp[0] / torch.sum(omega_mask_exp) * mask_avg_f + \
                    omega_mask_exp[1] / torch.sum(omega_mask_exp) * mask_max_f
        edge_fuse = omega_edge_exp[0] / torch.sum(omega_edge_exp) * edge_avg_f + \
                    omega_edge_exp[1] / torch.sum(omega_edge_exp) * edge_max_f
        epr_f = mask_fuse.sigmoid() * mask_features + edge_fuse.sigmoid() * edge_features
        epr_features = self.epr_proj(epr_f)

        return obj_features, epr_features


@ANSWER_BRANCH_REGISTRY.register()
class AnswerBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS
        in_channels = channels + 2 if cfg.MODEL.QANET.POSITION_EMBEDING.IS_USING else channels  # +2 position information

        self.init_conv = nn.Conv2d(in_channels, channels, 3)
        self.mask_branch = MaskEdgeBranch(cfg, channels)
        self.edge_branch = MaskEdgeBranch(cfg, channels)
        self.obj_epr_branch = ObjEprBranch(cfg)
        self.init_weights()

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
        features = self.init_conv(features)
        mask_features = self.mask_branch(features)
        edge_features = self.edge_branch(features)
        obj_features, epr_features = self.obj_epr_branch(mask_features, edge_features)

        return mask_features, edge_features, obj_features, epr_features


def build_answer_branch(cfg):
    return ANSWER_BRANCH_REGISTRY.get('AnswerBranch')(cfg)
