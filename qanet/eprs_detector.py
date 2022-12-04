import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill
from detectron2.utils.registry import Registry

EPRS_DETECTOR_REGISTRY = Registry("EPRS_DETECTOR")
EPRS_DETECTOR_REGISTRY.__doc__ = "registry for error-prone region detector module"


def pos_embed(x, temperature=10000, scale=2 * math.pi, normalize=True):

    batch_size, channel, height, width = x.size()
    mask = x.new_ones((batch_size, height, width))
    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    num_pos_feats = channel // 2
    assert num_pos_feats * 2 == channel, (
        'The input channel number must be an even number.')
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                         pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                         pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.MODEL.EPRDETECTOR.DIM
        nhead = cfg.MODEL.EPRDETECTOR.NUM_HEAD
        dim_feedforward = cfg.MODEL.EPRDETECTOR.DIM_FORWARD
        dropout = cfg.MODEL.EPRDETECTOR.DROPOUT
        activation = cfg.MODEL.EPRDETECTOR.ACT

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, mask):
        q = k = v = src
        src2 = self.self_attn(q, k, v, mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


@EPRS_DETECTOR_REGISTRY.register()
class EprsDetector(nn.Module):
    def __init__(self, encoder_layer, cfg):
        super().__init__()
        d_model = cfg.MODEL.EPRDETECTOR.DIM
        self.num_layers = cfg.MODEL.EPRDETECTOR.NUM_LAYER
        self.layers = _get_clones(encoder_layer, self.num_layers)
        self.detector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, src, mask):

        output = src
        for layer in self.layers:
            output = layer(output, mask)

        return self.detector(output)


def build_eprs_detector(cfg):
    encoder_layer = TransformerEncoderLayer(cfg)
    return EPRS_DETECTOR_REGISTRY.get('EprsDetector')(encoder_layer, cfg)
