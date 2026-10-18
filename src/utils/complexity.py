"""Read-only forward complexity and hardware-specific efficiency measurement."""
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from thop import profile


def component(name):
    if name.startswith('local_path.attention.'): return 'BAA'
    if name.startswith(('morphology_head.region_predictor', 'morphology_head.distance_predictor')):
        return 'Morphology auxiliary-only'
    if name.startswith('morphology_head.'): return 'Morphology guidance/shared'
    return {'stem': 'MSS', 'stage1': 'Stage 1', 'stage2': 'Stage 2', 'stage3': 'Stage 3', 'stage4': 'Stage 4',
            'global_path': 'Global Path', 'local_path': 'Local Path (excluding BAA)', 'fusion': 'AMGF',
            'classifier': 'Classifier'}.get(name.split('.')[0], name.split('.')[0])


def parameter_breakdown(model):
    counts = Counter()
    for name, parameter in model.named_parameters(): counts[component(name)] += parameter.numel()
    total = sum(counts.values())
    return pd.DataFrame([dict(module=k, parameters=v, percentage_of_total=100*v/total) for k, v in counts.items()])


def fingerprint(model, config, root):
    sources = {name: hashlib.sha256((root / 'src/models' / name).read_bytes()).hexdigest()
               for name in ('blocks.py', 'attention.py', 'fusion.py', 'morphonet_bt.py', '__init__.py')}
    payload = dict(model_class=type(model).__name__, parameters=sum(p.numel() for p in model.parameters()),
                   model_config=config, source_sha256=sources,
                   parameter_shapes={k: list(v.shape) for k, v in model.named_parameters()})
    payload['fingerprint_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


class ForwardPath(nn.Module):
    def __init__(self, model, auxiliary=False):
        super().__init__()
        self.model, self.auxiliary = model, auxiliary

    def forward(self, image):
        return self.model(image, return_aux=True) if self.auxiliary else self.model(image)


def thop_measure(model, image, auxiliary=False):
    def zero(module, inputs, output):
        module.total_ops += 0
    # Profile disposable copies: THOP attaches counters/hooks and does not clean all custom containers.
    raw, executed_params = profile(ForwardPath(deepcopy(model), auxiliary), (image,), verbose=False)
    conv_linear, _ = profile(ForwardPath(deepcopy(model), auxiliary), (image,), verbose=False,
                             custom_ops={nn.BatchNorm2d: zero, nn.AdaptiveAvgPool2d: zero})
    return dict(thop_default_mixed_operations=int(raw), thop_executed_parameter_count=int(executed_params),
                thop_conv_linear_macs=int(conv_linear))


def trace_layers(model, image, auxiliary=False):
    rows, handles = [], []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.AdaptiveAvgPool2d)): continue
        def hook(layer, args, out, name=name):
            macs = 0
            if isinstance(layer, nn.Conv2d):
                macs = out.numel() * (layer.in_channels // layer.groups) * np.prod(layer.kernel_size)
            elif isinstance(layer, nn.Linear): macs = out.numel() * layer.in_features
            rows.append(dict(layer=name, module=component(name), type=type(layer).__name__,
                             input_shape=json.dumps(list(args[0].shape)), output_shape=json.dumps(list(out.shape)),
                             conv_linear_macs=int(macs), output_elements=out.numel(),
                             bias_additions=out.numel() if isinstance(layer, (nn.Conv2d, nn.Linear)) and layer.bias is not None else 0,
                             batchnorm_scalar_ops=4*out.numel() if isinstance(layer, nn.BatchNorm2d) else 0,
                             pooling_scalar_ops=args[0].numel() if isinstance(layer, nn.AdaptiveAvgPool2d) else 0,
                             thop_pooling_ops=args[0].numel()+out.numel() if isinstance(layer, nn.AdaptiveAvgPool2d) else 0))
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            output = model(image, return_aux=True) if auxiliary else model(image)
    finally:
        for handle in handles: handle.remove()
    return pd.DataFrame(rows)


class OperationAudit(TorchDispatchMode):
    """Observe actual ATen calls; no replacement of architecture operations."""
    def __init__(self):
        super().__init__()
        self.rows = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        tensor = output if torch.is_tensor(output) else None
        self.rows.append(dict(operation=str(func), output_elements=tensor.numel() if tensor is not None else 0,
                              output_shape=str(list(tensor.shape)) if tensor is not None else 'tuple/other'))
        return output


def additional_operations(model, image, layers, auxiliary=False):
    mode = OperationAudit()
    with torch.inference_mode(), mode:
        output = model(image, return_aux=True) if auxiliary else model(image)
    rows = pd.DataFrame(mode.rows)
    sums = rows.groupby('operation').output_elements.sum()
    elementwise = sum(int(count) for op, count in sums.items()
                      if op.startswith(('aten.add.', 'aten.mul.', 'aten.sub.', 'aten.rsub.', 'aten.div.')))
    sigmoid = sum(int(count) for op, count in sums.items() if op.startswith('aten.sigmoid.'))
    silu = sum(int(count) for op, count in sums.items() if op.startswith('aten.silu.'))
    interpolated = sum(int(count) for op, count in sums.items() if op.startswith('aten.upsample_bilinear2d.'))
    # Per bilinear output: 4 weighted samples + 3 additions; weights/coordinates excluded.
    extra = dict(batchnorm_ops_estimate=int(layers.batchnorm_scalar_ops.sum()),
                 pooling_ops_estimate=int(layers.pooling_scalar_ops.sum()), bias_additions=int(layers.bias_additions.sum()),
                 functional_elementwise_ops=elementwise, bilinear_output_elements=interpolated,
                 bilinear_ops_estimate=7*interpolated, silu_output_elements=silu,
                 sigmoid_output_elements=sigmoid)
    # sigmoid: negation + exponential + add + reciprocal (4 operations, exp unit-cost assumption).
    extra['nonlinear_ops_unit_cost_estimate'] = 5*silu + 4*sigmoid
    extra['additional_ops_estimate'] = (extra['batchnorm_ops_estimate'] + extra['pooling_ops_estimate'] +
        extra['bias_additions'] + elementwise + 7*interpolated + extra['nonlinear_ops_unit_cost_estimate'])
    return extra, rows


def depthwise_audit(model):
    expected = ['stem.branch_b.0.0', 'stem.branch_c.0.0', 'morphology_head.features.1.0',
                'global_path.0.0', 'local_path.depthwise.0', 'local_path.attention.local_score.0']
    expected += [f'stage{s}.{b}.depthwise.0' for s, count in [(1,2), (2,2), (3,3), (4,2)] for b in range(count)]
    modules = dict(model.named_modules())
    rows = []
    for name in expected:
        layer = modules[name]
        valid = isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels and layer.out_channels == layer.in_channels
        if not valid: raise ValueError(f'Intended depthwise convolution is not depthwise: {name}')
        rows.append(dict(layer=name, in_channels=layer.in_channels, out_channels=layer.out_channels,
                         groups=layer.groups, kernel=str(layer.kernel_size), dilation=str(layer.dilation), status='PASS'))
    return pd.DataFrame(rows)


def cuda_memory(model, batch_size, auxiliary=False):
    gc.collect(); torch.cuda.empty_cache()
    image = torch.randn(batch_size, 1, 256, 256, device='cuda', dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(3): output = model(image, return_aux=True) if auxiliary else model(image)
        del output
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = model(image, return_aux=True) if auxiliary else model(image)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        result = dict(path='auxiliary' if auxiliary else 'inference', batch_size=batch_size,
                      baseline_allocated_bytes=baseline, peak_allocated_bytes=peak,
                      incremental_forward_peak_bytes=peak-baseline,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      input_bytes=image.numel()*image.element_size(),
                      model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                      model_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()))
        result['other_baseline_bytes'] = (baseline-result['model_parameter_bytes']-
                                         result['model_buffer_bytes']-result['input_bytes'])
    del image, output
    return result


def cuda_latency(model, auxiliary=False, warmup=30, repeats=100):
    image = torch.randn(1, 1, 256, 256, device='cuda', dtype=torch.float32)
    samples = []
    with torch.inference_mode():
        for _ in range(warmup): output = model(image, return_aux=True) if auxiliary else model(image)
        del output
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(image, return_aux=True) if auxiliary else model(image)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start)*1000)
            del output
    mean = float(np.mean(samples))
    return dict(path='auxiliary' if auxiliary else 'inference', batch_size=1, warmup=warmup, timed_iterations=repeats,
                mean_ms=mean, std_ms=float(np.std(samples, ddof=1)), median_ms=float(np.median(samples)),
                p95_ms=float(np.percentile(samples,95)), images_per_second=1000/mean,
                hardware=torch.cuda.get_device_name(0), timer='synchronized wall-clock perf_counter'), samples
