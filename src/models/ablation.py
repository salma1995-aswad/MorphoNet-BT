"""Controlled sequential ablations; frozen full-model implementation is untouched."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from .morphonet_bt import MorphoNetBT, MODEL_CONFIG
from .blocks import LocalMorphologyPath
from src.training.losses import MorphoNetLoss

FLAGS = ('use_mss', 'use_morphology_supervision', 'use_baa', 'use_dual_path', 'use_amgf')
VARIANTS = {f'A{i}': dict(zip(FLAGS, [j < i for j in range(5)])) for i in range(6)}


def validate_flags(flags):
    if set(flags) != set(FLAGS) or any(type(value) is not bool for value in flags.values()):
        raise ValueError('Provide exactly five explicit Boolean component flags')
    if flags['use_baa'] and not flags['use_morphology_supervision']:
        raise ValueError('Primary ablation BAA requires supervised MRI-derived boundary predictions')
    if flags['use_amgf'] and not flags['use_dual_path']:
        raise ValueError('AMGF requires the Global/Local dual path')


class ControlledAblation(MorphoNetBT):
    """Reuse frozen encoder construction; retain a real local path when dual is off.

    Classification-only inference skips morphology heads when BAA is absent.
    All enabled auxiliary heads participate in the supervised training loss.
    """
    def __init__(self, config):
        validate_flags({key: config[key] for key in FLAGS})
        super().__init__(**config)
        if not self.use_dual_path:
            del self.standalone_attention
            self.local_path = LocalMorphologyPath(config['stage_channels'][2], use_baa=self.use_baa)
            self.local_path.apply(self._initialize)

    def forward(self, image, *, return_aux=False):
        if image.ndim != 4 or image.shape[1:] != (1, 256, 256):
            raise ValueError('Expected MRI [B,1,256,256]')
        if not torch.is_floating_point(image):
            raise TypeError('MRI must be floating point')
        stage2 = self.stage2(self.stage1(self.stem(image)))
        needed = self.use_baa or (return_aux and self.use_morphology_supervision)
        morphology = self.morphology_head(stage2, output_size=image.shape[-2:], return_aux=return_aux) if needed else {}
        stage3 = self.stage3(stage2)
        local, attention = self.local_path(stage3, morphology.get('boundary_logits_low'))
        gate = None
        if self.use_dual_path:
            global_features = self.global_path(stage3)
            if self.fusion is not None:
                fused, gate = self.fusion(global_features, local, return_gate=True)
            else:
                fused = .5 * (global_features + local)
        else:
            fused = local
        logits = self.classifier(self.dropout(self.flatten(self.pool(self.stage4(fused)))))
        if not return_aux:
            return logits
        return {'logits': logits, **{k: v for k, v in morphology.items() if k != 'boundary_logits_low'},
                'boundary_attention': attention, 'fusion_gate': gate}


def build_ablation(variant, base_config=None):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown ablation variant: {variant}; expected A0 through A5')
    config = deepcopy(MODEL_CONFIG if base_config is None else base_config)
    # Only component flags may differ from the frozen structural configuration.
    for key, value in MODEL_CONFIG.items():
        if key not in FLAGS and config.get(key) != value:
            raise ValueError(f'Frozen structural setting changed: {key}')
    config.update(VARIANTS[variant])
    validate_flags({key: config[key] for key in FLAGS})
    return MorphoNetBT(**config) if variant == 'A5' else ControlledAblation(config)


class AblationLoss(torch.nn.Module):
    """CE-only variants never access morphology targets; others reuse Phase 6 loss."""
    def __init__(self, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown variant')
        self.morphology_enabled = VARIANTS[variant]['use_morphology_supervision']
        self.morphology_loss = MorphoNetLoss() if self.morphology_enabled else None

    def forward(self, outputs, batch):
        if self.morphology_enabled:
            return self.morphology_loss(outputs, batch)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        loss = F.cross_entropy(logits.float(), batch['label'])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite classification loss')
        return dict(classification_loss=loss, total_loss=loss)
