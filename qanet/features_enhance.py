import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill
from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

FEATURES_ENHANCE_REGISTRY = Registry("FEATURES_ENHANCE")
FEATURES_ENHANCE_REGISTRY.__doc__ = "registry for features enhance module"
RF_ENHANCE_REGISTRY = Registry("RF_ENHANCE")
RF_ENHANCE_REGISTRY.__doc__ = "registry for receptive field enhance"


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


'''Rethinking Atrous Convolution for Semantic Image Segmentation'''
@RF_ENHANCE_REGISTRY.register()
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=None):
        super(ASPP, self).__init__()
        if atrous_rates is None:
            atrous_rates = [6, 12, 18]

        modules = [nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))]

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, in_channels, rate1))
        modules.append(ASPPConv(in_channels, in_channels, rate2))
        modules.append(ASPPConv(in_channels, in_channels, rate3))
        modules.append(ASPPPooling(in_channels, in_channels))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


@RF_ENHANCE_REGISTRY.register()
class PPM(nn.Module):
    def __init__(self, in_channels, channels=None, sizes=(1, 2, 3, 6)):
        super().__init__()
        if not channels:
            channels = in_channels // 4

        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


@RF_ENHANCE_REGISTRY.register()
class NORFE(nn.Module):
    def __init__(self, nothing):
        super(NORFE, self).__init__()
        self.fn = nn.Identity()

    def forward(self, x):
        return self.fn(x)


@FEATURES_ENHANCE_REGISTRY.register()
class FeaturesEnhanceModule(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.in_features = cfg.MODEL.QANET.FEATURES_ENHANCE.IN_FEATURES
        self.num_channels = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS
        self.rfe_name = cfg.MODEL.QANET.FEATURES_ENHANCE.RFENAME
        self.in_channels = [input_shape[f].channels for f in self.in_features]

        fpn_laterals = []
        fpn_outputs = []
        for in_channel in reversed(self.in_channels):
            lateral_conv = Conv2d(in_channel, self.num_channels, 1)
            output_conv = Conv2d(self.num_channels, self.num_channels, 3, padding=1)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)

        self.rf = RF_ENHANCE_REGISTRY.get(self.rfe_name)(self.num_channels)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

    def forward(self, features):

        features = [features[f] for f in self.in_features]
        features = features[::-1]

        prev_features = self.rf(self.fpn_laterals[0](features[0]))
        outputs = [self.fpn_outputs[0](prev_features)]
        for feature, lat_conv, output_conv in zip(features[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]):
            lat_features = lat_conv(feature)
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode='nearest')
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))

        return outputs


def build_features_enhance(cfg, input_shape):
    return FEATURES_ENHANCE_REGISTRY.get('FeaturesEnhanceModule')(cfg, input_shape)
