import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from detectron2.utils.registry import Registry

QUESTION_BRANCH_REGISTRY = Registry("QUESTION_BRANCH")
QUESTION_BRANCH_REGISTRY.__doc__ = "registry for question branch"


@QUESTION_BRANCH_REGISTRY.register()
class QuestionBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS
        in_channels = channels + 2 if cfg.MODEL.QANET.FEATURES_MERGING.IS_USING_POS else channels  # +2 position
        hidden_dim = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM
        init_convs_num = cfg.MODEL.QANET.QA_BRANCH.INIT_CONVS
        num_masks = cfg.MODEL.QANET.QA_BRANCH.NUM_MASKS
        self.num_groups = cfg.MODEL.QANET.QA_BRANCH.GROUPS

        self.init_conv = self.init_convs(init_convs_num, in_channels, channels)
        expand_dim = channels * self.num_groups
        self.iam_conv = nn.Conv2d(
            channels, num_masks*self.num_groups, 3, padding=1, groups=self.num_groups)
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.proj = nn.Linear(expand_dim, hidden_dim)

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
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, val=0.0)

    def forward(self, features):

        features = self.init_conv(features)  # B C H W
        iam = self.iam_conv(features)  # B G*N H W
        iam_prob = iam.sigmoid()  # activation, so do not use relu

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        iam_prob = iam_prob.view(B, N, -1)  # B G*N H*W
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]

        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))  # B G*N C
        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        inst_features = F.relu_(self.fc(inst_features))
        lsf = self.proj(inst_features)
        return lsf


def build_question_branch(cfg):
    return QUESTION_BRANCH_REGISTRY.get('QuestionBranch')(cfg)
