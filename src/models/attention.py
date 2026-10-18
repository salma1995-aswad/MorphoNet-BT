"""Boundary guidance uses predictions from MRI, never ground-truth targets."""
import torch
from torch import nn
from torch.nn import functional as F


class BoundaryAwareAttention(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.local_score = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                                          nn.SiLU(inplace=False), nn.Conv2d(channels, 1, 1))

    def forward(self, features, learned_boundary_logits=None, *, return_attention=False):
        score = self.local_score(features)
        if learned_boundary_logits is not None:
            if (learned_boundary_logits.ndim != 4 or learned_boundary_logits.shape[0] != features.shape[0]
                    or learned_boundary_logits.shape[1] != 1):
                raise ValueError('Learned boundary guidance must be [B,1,H,W]')
            score = score + F.interpolate(learned_boundary_logits, size=features.shape[-2:],
                                           mode='bilinear', align_corners=False)
        attention = torch.sigmoid(score)
        result = features * (1 + attention)
        return (result, attention) if return_attention else result
