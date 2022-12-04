import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill
from detectron2.utils.registry import Registry

QUESTION_BRANCH_REGISTRY = Registry("QUESTION_BRANCH")
QUESTION_BRANCH_REGISTRY.__doc__ = "registry for question branch"


@QUESTION_BRANCH_REGISTRY.register()
class QuestionBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.QANET.FEATURES_ENHANCE.NUM_CHANNELS
        in_channels = channels+2 if cfg.MODEL.QANET.POSITION_EMBEDING.IS_USING else channels  # +2 position information
        hidden_dim = cfg.MODEL.QANET.QA_BRANCH.HIDDEN_DIM
        num_masks = cfg.MODEL.QANET.QA_BRANCH.NUM_MASKS
        self.num_groups = cfg.MODEL.QANET.QA_BRANCH.GROUPS

        self.init_conv = nn.Conv2d(in_channels, channels, 3)
        expand_dim = channels * self.num_groups
        self.iam_conv = nn.Conv2d(
            channels, num_masks*self.num_groups, 3, padding=1, groups=self.num_groups)
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.proj = nn.Linear(expand_dim, hidden_dim)

        self.prior_prob = 0.01
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):

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
        lsf = self.proj(inst_features)
        return lsf


def build_question_branch(cfg):
    return QUESTION_BRANCH_REGISTRY.get('QuestionBranch')(cfg)
