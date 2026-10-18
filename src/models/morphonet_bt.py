"""Proposed MorphoNet-BT; both public forward modes accept MRI as the only data input."""
import torch
from torch import nn
from .blocks import MultiScaleStem, ConvNormAct, MFEBlock, MorphologyHead, GlobalContextPath, LocalMorphologyPath
from .attention import BoundaryAwareAttention
from .fusion import AdaptiveMorphologicalGatedFusion

MODEL_CONFIG = dict(name='MorphoNet-BT', input_channels=1, num_classes=3, dropout=0.30,
                    stem_channels=32, stem_branch_channels=16,
                    stage_channels=[48, 80, 128, 192], stage_blocks=[2, 2, 3, 2],
                    expansion_ratio=2, channel_gate_reduction=16, morphology_hidden_channels=32,
                    fusion_reduction=8, use_mss=True, use_morphology_supervision=True,
                    use_baa=True, use_dual_path=True, use_amgf=True)


class MorphoNetBT(nn.Module):
    def __init__(self, *, name='MorphoNet-BT', input_channels=1, num_classes=3, dropout=.30,
                 stem_channels=32, stem_branch_channels=16, stage_channels=(48, 80, 128, 192),
                 stage_blocks=(2, 2, 3, 2), expansion_ratio=2, channel_gate_reduction=16,
                 morphology_hidden_channels=32, fusion_reduction=8, use_mss=True,
                 use_morphology_supervision=True, use_baa=True, use_dual_path=True, use_amgf=True):
        super().__init__()
        if input_channels != 1 or num_classes != 3 or name != 'MorphoNet-BT':
            raise ValueError('Expected grayscale MorphoNet-BT with three classes')
        if len(stage_channels) != 4 or len(stage_blocks) != 4 or min(stage_blocks) < 1:
            raise ValueError('Expected four nonempty stages')
        if use_amgf and not use_dual_path:
            raise ValueError('AMGF requires the dual path')
        self.use_morphology_supervision = use_morphology_supervision
        self.use_dual_path = use_dual_path
        self.use_baa = use_baa
        self.stem = (MultiScaleStem(input_channels, stem_channels, stem_branch_channels) if use_mss else
                     ConvNormAct(input_channels, stem_channels, 3, stride=2))
        current = stem_channels
        for index, (channels, count) in enumerate(zip(stage_channels, stage_blocks), 1):
            blocks = [MFEBlock(current, channels, stride=2, expansion_ratio=expansion_ratio,
                               gate_reduction=channel_gate_reduction)]
            blocks += [MFEBlock(channels, channels, expansion_ratio=expansion_ratio,
                                gate_reduction=channel_gate_reduction) for _ in range(count - 1)]
            setattr(self, f'stage{index}', nn.Sequential(*blocks))
            current = channels
        self.morphology_head = (MorphologyHead(stage_channels[1], morphology_hidden_channels,
                                               auxiliary=use_morphology_supervision)
                                if use_morphology_supervision or use_baa else None)
        channels = stage_channels[2]
        if use_dual_path:
            self.global_path = GlobalContextPath(channels)
            self.local_path = LocalMorphologyPath(channels, use_baa=use_baa)
            self.fusion = AdaptiveMorphologicalGatedFusion(channels, fusion_reduction) if use_amgf else None
        else:
            self.standalone_attention = BoundaryAwareAttention(channels) if use_baa else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(stage_channels[3], num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image, *, return_aux=False):
        if image.ndim != 4 or image.shape[1:] != (1, 256, 256):
            raise ValueError(f'Expected MRI [B,1,256,256], got {tuple(image.shape)}')
        if not torch.is_floating_point(image):
            raise TypeError('MRI must be floating point')
        features = self.stage1(self.stem(image))
        stage2 = self.stage2(features)
        morphology = (self.morphology_head(stage2, output_size=image.shape[-2:], return_aux=return_aux)
                      if self.morphology_head is not None else {})
        boundary_low = morphology.get('boundary_logits_low')
        stage3 = self.stage3(stage2)
        attention, gate = None, None
        if self.use_dual_path:
            global_features = self.global_path(stage3)
            local_features, attention = self.local_path(stage3, boundary_low)
            if self.fusion is not None:
                fused, gate = self.fusion(global_features, local_features, return_gate=True)
            else:
                fused = .5 * (global_features + local_features)
        elif self.standalone_attention is not None:
            fused, attention = self.standalone_attention(stage3, boundary_low, return_attention=True)
        else:
            fused = stage3
        final = self.stage4(fused)
        logits = self.classifier(self.dropout(self.flatten(self.pool(final))))
        if not return_aux:
            return logits
        return {'logits': logits, **{k: v for k, v in morphology.items() if k != 'boundary_logits_low'},
                'boundary_attention': attention, 'fusion_gate': gate}
