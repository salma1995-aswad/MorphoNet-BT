"""Standard lightweight CNN building blocks; no individual novelty claim."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dilation=1, activation=True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation=dilation, groups=groups, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.SiLU(inplace=False) if activation else nn.Identity())


class MultiScaleStem(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, branch_channels=16):
        super().__init__()
        self.branch_a = ConvNormAct(in_channels, branch_channels, 3, stride=2)
        self.branch_b = nn.Sequential(ConvNormAct(in_channels, in_channels, 5, stride=2, groups=in_channels),
                                      ConvNormAct(in_channels, branch_channels))
        self.branch_c = nn.Sequential(ConvNormAct(in_channels, in_channels, 3, stride=2, groups=in_channels, dilation=2),
                                      ConvNormAct(in_channels, branch_channels))
        self.project = ConvNormAct(3 * branch_channels, out_channels)

    def forward(self, image):
        return self.project(torch.cat([self.branch_a(image), self.branch_b(image), self.branch_c(image)], dim=1))


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                  nn.SiLU(inplace=False), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, features):
        return features * self.gate(features)


class MFEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_ratio=2, gate_reduction=16):
        super().__init__()
        if stride not in (1, 2) or expansion_ratio < 1:
            raise ValueError('MFE supports stride 1/2 and expansion >= 1')
        hidden = int(in_channels * expansion_ratio)
        self.expand = ConvNormAct(in_channels, hidden)
        self.depthwise = ConvNormAct(hidden, hidden, 3, stride=stride, groups=hidden)
        self.project = ConvNormAct(hidden, out_channels, activation=False)
        self.channel_gate = ChannelGate(out_channels, gate_reduction)
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == out_channels else
                         ConvNormAct(in_channels, out_channels, stride=stride, activation=False))
        self.activation = nn.SiLU(inplace=False)

    def forward(self, features):
        branch = self.channel_gate(self.project(self.depthwise(self.expand(features))))
        shortcut = self.shortcut(features)
        if branch.shape != shortcut.shape:
            raise RuntimeError(f'MFE residual shape mismatch: {branch.shape} versus {shortcut.shape}')
        return self.activation(branch + shortcut)


class MorphologyHead(nn.Module):
    """Stage-2 features -> shared bottleneck -> maps; boundary also guides classification."""
    def __init__(self, in_channels=80, hidden_channels=32, auxiliary=True):
        super().__init__()
        self.features = nn.Sequential(ConvNormAct(in_channels, hidden_channels),
                                      ConvNormAct(hidden_channels, hidden_channels, 3, groups=hidden_channels))
        self.boundary_predictor = nn.Conv2d(hidden_channels, 1, 1)
        self.auxiliary = auxiliary
        if auxiliary:
            self.region_predictor = nn.Conv2d(hidden_channels, 1, 1)
            self.distance_predictor = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, features, *, output_size=(256, 256), return_aux=False):
        shared = self.features(features)
        boundary_low = self.boundary_predictor(shared)
        result = {'boundary_logits_low': boundary_low}
        if return_aux and self.auxiliary:
            def resize(value):
                return F.interpolate(value, size=output_size, mode='bilinear', align_corners=False)
            result.update(region_logits=resize(self.region_predictor(shared)),
                          boundary_logits=resize(boundary_low),
                          distance=torch.sigmoid(resize(self.distance_predictor(shared))))
        return result


class GlobalContextPath(nn.Sequential):
    def __init__(self, channels=128):
        # Effective 9x9 local receptive field from a dilated 5x5 depthwise operation.
        # "Global" refers to contextual branch intent, not all-to-all attention.
        super().__init__(ConvNormAct(channels, channels, 5, groups=channels, dilation=2),
                         ConvNormAct(channels, channels))


class LocalMorphologyPath(nn.Module):
    def __init__(self, channels=128, use_baa=True):
        super().__init__()
        from .attention import BoundaryAwareAttention
        self.depthwise = ConvNormAct(channels, channels, 3, groups=channels)
        self.attention = BoundaryAwareAttention(channels) if use_baa else None
        self.project = ConvNormAct(channels, channels)

    def forward(self, features, learned_boundary_logits=None):
        local = self.depthwise(features)
        attention = None
        if self.attention is not None:
            local, attention = self.attention(local, learned_boundary_logits, return_attention=True)
        return self.project(local), attention
