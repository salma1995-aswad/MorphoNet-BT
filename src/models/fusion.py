"""Channel-wise adaptive convex fusion of compatible local/global features."""
import torch
from torch import nn


class AdaptiveMorphologicalGatedFusion(nn.Module):
    def __init__(self, channels=128, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                  nn.SiLU(inplace=False), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, global_features, local_features, *, return_gate=False):
        if global_features.shape != local_features.shape or global_features.ndim != 4:
            raise ValueError('AMGF requires identical [B,C,H,W] input shapes')
        alpha = self.gate(global_features + local_features)
        fused = alpha * local_features + (1 - alpha) * global_features
        return (fused, alpha) if return_gate else fused
