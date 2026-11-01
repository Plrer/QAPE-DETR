"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from ..core import register
from .hybrid_encoder import ConvNormLayer_fuse, RepNCSPELAN4

__all__ = ['LiteEncoder']


class GAP_Fusion(nn.Module):
    """BiFusion Block in PAN"""
    def __init__(self, in_channels, out_channels, act=None):
        super().__init__()
        self.cv = ConvNormLayer_fuse(out_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1)
        x = x + gap
        return self.cv(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class QualityEstimator(nn.Module):
    def __init__(self, in_ch, hidden=32, q_dim=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, q_dim, 1, bias=True)
        )

    def forward(self, x):
        return self.mlp(self.pool(x)).flatten(1)


class SpatialGate(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, in_ch, k=3, s=1, g=in_ch),
            nn.Conv2d(in_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class ChannelGateFromQ(nn.Module):
    def __init__(self, q_dim, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(q_dim, channels),
            nn.Sigmoid()
        )

    def forward(self, q):
        return self.fc(q).unsqueeze(-1).unsqueeze(-1)


class LGPBlock(nn.Module):
    def __init__(self, channels, q_dim=16, pool_ks=3):
        super().__init__()
        self.pool_ks = pool_ks
        self.spatial_gate = SpatialGate(channels)
        self.channel_gate = ChannelGateFromQ(q_dim, channels)
        self.refine = ConvBNAct(channels, channels, k=3, s=1)

    def low_pass(self, x):
        return F.avg_pool2d(
            x,
            kernel_size=self.pool_ks,
            stride=1,
            padding=self.pool_ks // 2
        )

    def forward(self, x, q):
        low = self.low_pass(x)
        high = x - low
        ms = self.spatial_gate(low)
        mc = self.channel_gate(q)
        high = high * ms * mc
        out = low + high
        out = self.refine(out)
        return out


class ECRBlock(nn.Module):
    def __init__(self, channels, q_dim=16):
        super().__init__()
        self.dw3 = ConvBNAct(channels, channels, k=3, s=1, g=channels)
        self.dw5 = ConvBNAct(channels, channels, k=5, s=1, g=channels)
        self.pw = ConvBNAct(channels * 2, channels, k=1, s=1)
        self.gamma_fc = nn.Linear(q_dim, channels)

    def forward(self, x, q):
        e1 = self.dw3(x)
        e2 = self.dw5(x)
        edge = self.pw(torch.cat([e1, e2], dim=1))
        gamma = torch.sigmoid(self.gamma_fc(q)).unsqueeze(-1).unsqueeze(-1)
        return x + gamma * edge


class QAFLiteBlock(nn.Module):
    """Use deep p5 to refine shallow p4."""
    def __init__(self, channels, q_dim=16):
        super().__init__()
        self.alpha_fc = nn.Linear(q_dim, channels)
        self.beta_fc = nn.Linear(q_dim, channels)
        self.out = ConvBNAct(channels, channels, k=3, s=1)

    def forward(self, p4, p5, q):
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        alpha = torch.sigmoid(self.alpha_fc(q)).unsqueeze(-1).unsqueeze(-1)
        beta = torch.sigmoid(self.beta_fc(q)).unsqueeze(-1).unsqueeze(-1)
        fused = alpha * p4 + beta * p5_up
        return self.out(fused)


@register()
class LiteEncoder(nn.Module):
    __share__ = ['eval_spatial_size']

    def __init__(self,
                 in_channels=[256],
                 feat_strides=[16],
                 hidden_dim=64,
                 expansion=0.34,
                 depth_mult=0.5,
                 act='silu',
                 eval_spatial_size=None,
                 csp_type='csp2',
                 use_lgp=False,
                 use_ecr=False,
                 use_qaf_lite=False,
                 q_dim=16,
                 lgp_pool_ks=3,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.eval_spatial_size = eval_spatial_size

        # LiteEncoder 输出两级特征: P4, P5
        base_stride = feat_strides[0]
        self.out_channels = [hidden_dim, hidden_dim]
        self.out_strides = [base_stride, base_stride * 2]

        self.use_lgp = use_lgp
        self.use_ecr = use_ecr
        self.use_qaf_lite = use_qaf_lite
        self.q_dim = q_dim
        self.lgp_pool_ks = lgp_pool_ks

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim))
            ]))
            self.input_proj.append(proj)

        # get the small-scale feature
        down_sample = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act)
        )
        self.down_sample1 = copy.deepcopy(down_sample)
        self.down_sample2 = copy.deepcopy(down_sample)

        # Bi-Fusion
        self.bi_fusion = GAP_Fusion(hidden_dim, hidden_dim, act=act)

        # fuse blocks
        c1 = hidden_dim
        c2 = hidden_dim
        c3 = hidden_dim * 2
        c4 = round(expansion * hidden_dim // 2)
        num_blocks = round(3 * depth_mult)

        fuse_block = RepNCSPELAN4(
            c1=c1, c2=c2, c3=c3, c4=c4,
            n=num_blocks, act=act, csp_type=csp_type
        )
        self.fpn_block = copy.deepcopy(fuse_block)
        self.pan_block = copy.deepcopy(fuse_block)

        # QG-LPE modules
        if self.use_lgp or self.use_ecr or self.use_qaf_lite:
            self.quality_estimator = QualityEstimator(hidden_dim, hidden=32, q_dim=q_dim)
            self.lgp_block = LGPBlock(hidden_dim, q_dim=q_dim, pool_ks=lgp_pool_ks)
            self.ecr_block = ECRBlock(hidden_dim, q_dim=q_dim)
            self.qaf_lite_block = QAFLiteBlock(hidden_dim, q_dim=q_dim)

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)

        # 1) input projection
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Atto + LiteEncoder: single backbone feature input
        x = proj_feats[0]

        # 2) quality vector
        if self.use_lgp or self.use_ecr or self.use_qaf_lite:
            q = self.quality_estimator(x)

        # 3) LGP
        if self.use_lgp:
            x = self.lgp_block(x, q)

        # 4) ECR
        if self.use_ecr:
            x = self.ecr_block(x, q)

        proj_feats[0] = x

        # 5) generate small-scale feature
        proj_feats.append(self.down_sample1(proj_feats[-1]))

        # 6) Bi-Fusion on deep feature
        proj_feats[-1] = self.bi_fusion(proj_feats[-1])

        outs = []

        # 7) FPN branch -> p4
        fuse_feat = proj_feats[0] + F.interpolate(
            proj_feats[1],
            scale_factor=2.0,
            mode='nearest'
        )
        p4 = self.fpn_block(fuse_feat)
        outs.append(p4)

        # 8) PAN branch -> p5
        fuse_feat = proj_feats[1] + self.down_sample2(outs[-1])
        p5 = self.pan_block(fuse_feat)
        outs.append(p5)

        # 9) QAF-lite: use p5 to refine p4
        if self.use_qaf_lite:
            outs[0] = self.qaf_lite_block(outs[0], outs[1], q)

        return outs