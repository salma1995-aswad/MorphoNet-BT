"""Small architecture diagnostics only. No optimizer, trainer, or model selection."""
import json
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
from .blocks import MFEBlock

SMOKE_LOSS_WEIGHTS = dict(lambda_region=1., lambda_boundary=1., lambda_distance=.5, lambda_morphology=.3)


def validate_outputs(outputs, batch_size):
    shapes = {'logits': (batch_size, 3), 'region_logits': (batch_size, 1, 256, 256),
              'boundary_logits': (batch_size, 1, 256, 256), 'distance': (batch_size, 1, 256, 256),
              'boundary_attention': (batch_size, 1, 16, 16), 'fusion_gate': (batch_size, 128, 1, 1)}
    for key, shape in shapes.items():
        if outputs[key].shape != shape or not torch.isfinite(outputs[key]).all():
            raise ValueError(f'Invalid output: {key}, shape={outputs[key].shape}')
    for key in ('distance', 'boundary_attention', 'fusion_gate'):
        value = outputs[key]
        if value.min() < 0 or value.max() > 1:
            raise ValueError(f'{key} outside [0,1]')
    return {key: list(value.shape) for key, value in outputs.items()}


def smoke_losses(outputs, batch, weights=None):
    weights = SMOKE_LOSS_WEIGHTS if weights is None else weights
    def segmentation(logits, target):
        if logits.shape != target.shape:
            raise ValueError('Auxiliary target shape mismatch')
        probability = logits.sigmoid()
        axes = (1, 2, 3)
        dice = 1 - ((2 * (probability * target).sum(axes) + 1e-6) /
                    (probability.sum(axes) + target.sum(axes) + 1e-6)).mean()
        return F.binary_cross_entropy_with_logits(logits, target) + dice
    cls = F.cross_entropy(outputs['logits'], batch['label'])
    region = segmentation(outputs['region_logits'], batch['region'])
    boundary = segmentation(outputs['boundary_logits'], batch['boundary'])
    distance = F.smooth_l1_loss(outputs['distance'], batch['distance'])
    morphology = (weights['lambda_region'] * region + weights['lambda_boundary'] * boundary
                  + weights['lambda_distance'] * distance)
    total = cls + weights['lambda_morphology'] * morphology
    values = dict(classification=cls, region=region, boundary=boundary, distance=distance,
                  morphology=morphology, total=total)
    if not all(torch.isfinite(value) for value in values.values()):
        raise ValueError('Nonfinite smoke loss')
    return values


def gradient_table(model):
    rows = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        rows.append(dict(parameter=name, numel=parameter.numel(), present=gradient is not None,
                         finite=bool(torch.isfinite(gradient).all()) if gradient is not None else None,
                         norm=float(gradient.detach().norm().cpu()) if gradient is not None else 0.))
    return pd.DataFrame(rows)


def gradient_groups(table):
    prefixes = ['stem', 'stage1', 'stage2', 'stage3', 'stage4', 'morphology_head',
                'global_path', 'local_path', 'local_path.attention', 'fusion', 'classifier']
    # Include every MFE separately, not only stage totals.
    prefixes += [f'stage{stage}.{block}' for stage, count in [(1, 2), (2, 2), (3, 3), (4, 2)] for block in range(count)]
    rows = []
    for prefix in prefixes:
        subset = table[table.parameter.str.startswith(prefix + '.')]
        rows.append(dict(module=prefix, parameter_tensors=len(subset), with_gradient=int(subset.present.sum()),
                         all_finite=bool(subset.loc[subset.present, 'finite'].all()),
                         gradient_norm=float((subset.norm ** 2).sum() ** .5)))
    return pd.DataFrame(rows)


def tensor_statistics(value):
    value = value.detach().float()
    return dict(min=float(value.min().cpu()), max=float(value.max().cpu()),
                mean=float(value.mean().cpu()), std=float(value.std(unbiased=False).cpu()))


def shape_table(model, image):
    rows = [dict(module='Input', input_shape=str(list(image.shape)), output_shape=str(list(image.shape)), purpose='Grayscale MRI only')]
    modules = [('MSS', model.stem, 'Parallel receptive fields'), ('Stage 1', model.stage1, '2 MFE blocks'),
               ('Stage 2', model.stage2, '2 MFE blocks; shared supervision features'),
               ('MorphologyHead', model.morphology_head, 'Region/boundary/distance; learned boundary guidance'),
               ('Stage 3', model.stage3, '3 MFE blocks'),
               ('Global Context Path', model.global_path, 'Dilated depthwise context'),
               ('BAA', model.local_path.attention, 'MRI-derived spatial guidance'),
               ('Local Morphology Path', model.local_path, 'Local texture with BAA'),
               ('AMGF', model.fusion, 'Adaptive channel-wise convex fusion'),
               ('Stage 4', model.stage4, '2 MFE blocks'), ('GAP', model.pool, 'Spatial average'),
               ('Flatten', model.flatten, 'Channel vector'), ('Classifier', model.classifier, '3 raw logits')]
    def shapes(value):
        if torch.is_tensor(value): return list(value.shape)
        if isinstance(value, (tuple, list)): return [shapes(v) for v in value]
        if isinstance(value, dict): return {k: shapes(v) for k, v in value.items()}
        return None
    handles = []
    for name, module, purpose in modules:
        def hook(_module, inputs, output, name=name, purpose=purpose):
            rows.append(dict(module=name, input_shape=json.dumps(shapes(inputs)),
                             output_shape=json.dumps(shapes(output)), purpose=purpose))
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            outputs = model(image, return_aux=True)
        validate_outputs(outputs, len(image))
    finally:
        for handle in handles: handle.remove()
    return pd.DataFrame(rows)


def residual_checks(device):
    rows = []
    for in_channels, out_channels, stride in [(48, 48, 1), (32, 48, 1), (32, 48, 2)]:
        block = MFEBlock(in_channels, out_channels, stride=stride).to(device).eval()
        image = torch.randn(2, in_channels, 16, 16, device=device)
        with torch.no_grad(): output = block(image)
        expected = (2, out_channels, 16 // stride, 16 // stride)
        if output.shape != expected or not torch.isfinite(output).all():
            raise ValueError('Residual validation failed')
        rows.append(dict(input_channels=in_channels, output_channels=out_channels, stride=stride,
                         shortcut=type(block.shortcut).__name__, output_shape=str(list(output.shape)), status='PASS'))
    return pd.DataFrame(rows)
