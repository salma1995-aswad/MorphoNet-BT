"""Disposable architecture, complexity, and memory checks; no model training."""
import gc
import hashlib
import importlib
import json
import platform
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from torch import nn
from torchvision import models

from src.data.dataset import BrainTumorDataset, verify_frozen_splits
from src.baselines.data import MRIOnlyBaselineDataset
from src.data.kfold import audit_splits, load_manifest, reload_splits
from src.models.baselines import BASELINES, DISPLAY_NAMES, INPUT_LAYERS, OUTPUT_LAYERS, build_baseline


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def verify_protected(root):
    """Check the pre-Phase-8 snapshot without deserializing any checkpoint."""
    snapshot = json.loads((root / 'results/baselines/setup/protected_files_before.json').read_text())
    for name, expected in snapshot.items():
        require(digest(root / name) == expected, f'Protected file changed: {name}')
    return len(snapshot)


def verify_data_protocol(root):
    """Audit split metadata and one training sample; never access test images."""
    count = verify_protected(root)
    hashes = verify_frozen_splits(root)
    approved = yaml.safe_load((root / 'results/architecture/preprocessing_v1_approved.yaml').read_text())
    current = yaml.safe_load((root / 'configs/default.yaml').read_text())
    require({key: current[key] for key in approved} == approved, 'Preprocessing config mismatch')
    manifest = load_manifest(root / 'results/dataset_statistics.csv')
    audit, coverage = audit_splits(reload_splits(root / 'splits'), manifest)
    path = root / 'splits/fold_1_train.csv'
    baseline = MRIOnlyBaselineDataset(path, 'train', config=approved)
    original = BrainTumorDataset(path, 'train', config=approved, include_targets=True)
    baseline.set_epoch(3)
    original.set_epoch(3)
    sample, reference = baseline[0], original[0]
    require(set(sample) == {'image', 'label', 'pid', 'sample_id'}, 'Auxiliary target escaped dataset')
    require(torch.equal(sample['image'], reference['image']), 'Baseline augmentation differs')
    require(torch.equal(sample['label'], reference['label']), 'Baseline class label differs')
    return dict(protected_files=count, split_sha256=hashes, coverage=coverage,
                training_image_bitwise_equal=True, dataset_keys=sorted(sample), test_images_accessed=False), audit


def parameter_digest(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        result.update(name.encode())
        result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def check_native_adaptation(name, model):
    """Verify every non-input parameter equals the seeded native constructor."""
    seed_everything()
    native = getattr(models, name)(weights=None, num_classes=3)
    first = INPUT_LAYERS[name]
    native_modules = dict(native.named_modules())
    require(set(native_modules) == set(dict(model.named_modules())), 'Module topology changed')
    for path, module in model.named_modules():
        require(type(module) is type(native_modules[path]), f'Module type changed: {path}')
        if path != first:
            require(module.extra_repr() == native_modules[path].extra_repr(), f'Module settings changed: {path}')
    old, new = native.get_submodule(first), model.get_submodule(first)
    require(new.in_channels == 1 and old.in_channels == 3, 'Input adaptation mismatch')
    for attribute in ['out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'padding_mode']:
        require(getattr(old, attribute) == getattr(new, attribute), f'Input spatial setting changed: {attribute}')
    native_state = native.state_dict()
    for key, value in model.state_dict().items():
        if key != first + '.weight':
            require(torch.equal(value, native_state[key]), f'Unnecessary parameter adaptation: {key}')
    classifier = model.get_submodule(OUTPUT_LAYERS[name])
    require(classifier.out_features == 3, 'Classifier adaptation mismatch')
    expected = sum(p.numel() for p in native.parameters()) - 2*old.out_channels*np.prod(old.kernel_size)
    require(sum(p.numel() for p in model.parameters()) == expected, 'Analytical parameter delta mismatch')
    return dict(native_topology_preserved=True, deeper_initial_parameters_unchanged=True,
                input_layer=first, output_layer=OUTPUT_LAYERS[name],
                input_kernel=list(new.kernel_size), input_stride=list(new.stride), weights=None)


def measure_complexity(model):
    """Count only executed Conv2d/Linear MACs; independently cross-check THOP.

    Bias, normalization, activation, pooling, residual adds, and gating multiply
    costs are excluded. Depthwise/grouped convolutions use in_channels/groups.
    """
    image = torch.randn(1, 1, 256, 256)
    rows, handles = [], []
    model.eval()
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        def hook(layer, inputs, output, name=name):
            per_output = (layer.in_channels // layer.groups)*np.prod(layer.kernel_size) if isinstance(layer, nn.Conv2d) else layer.in_features
            rows.append(dict(Layer=name, Type=type(layer).__name__, OutputShape=str(list(output.shape)),
                             MACs=int(output.numel()*per_output)))
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    thop_module = importlib.import_module('thop.profile')
    def zero(module, inputs, output):
        module.total_ops += 0
    overrides = {kind: zero for kind in thop_module.register_hooks if kind not in (nn.Conv2d, nn.Linear)}
    thop_macs, _ = thop_module.profile(deepcopy(model), inputs=(image,), custom_ops=overrides, verbose=False)
    macs = sum(row['MACs'] for row in rows)
    require(macs == int(thop_macs), 'Independent Conv/Linear MAC counts disagree')
    return macs, pd.DataFrame(rows)


def validate_backward(model):
    """FP32 forward and CE backward on two synthetic images, without an optimizer."""
    model.train()
    image = torch.randn(2, 1, 256, 256, device='cuda')
    labels = torch.tensor([0, 2], device='cuda')
    before = parameter_digest(model)
    logits = model(image)
    require(tuple(logits.shape) == (2, 3) and bool(torch.isfinite(logits).all()), 'Invalid FP32 logits')
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    parameters = list(model.named_parameters())
    reached = [name for name, parameter in parameters if parameter.requires_grad and parameter.grad is not None]
    require(len(reached) == len(parameters), 'Classification gradients do not reach all trainable parameters')
    require(all(bool(torch.isfinite(p.grad).all()) for _, p in parameters), 'Nonfinite FP32 gradient')
    require(float(parameters[0][1].grad.norm()) > 0 and float(parameters[-1][1].grad.norm()) > 0,
            'Missing nonzero input/classifier gradient')
    require(parameter_digest(model) == before, 'Parameters changed during backward-only test')
    model.zero_grad(set_to_none=True)
    return dict(status='PASS', input_shape=[2, 1, 256, 256], logits_shape=[2, 3], finite_logits=True,
                finite_gradients=True, parameter_tensors=len(parameters), gradient_tensors=len(reached),
                parameters_unchanged=True, optimizer_steps=0, synthetic_data=True)


def memory_batch32(model):
    """AMP forward/CE/backward peak; does not allocate optimizer states or update weights."""
    gc.collect()
    torch.cuda.empty_cache()
    model.train().zero_grad(set_to_none=True)
    baseline = torch.cuda.memory_allocated()
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    image = logits = loss = labels = None
    row = dict(BatchSize=32, Precision='CUDA autocast float16; FP32 parameters',
               BaselineAllocatedMiB=baseline/2**20, FreeBeforeMiB=free/2**20, DeviceTotalMiB=total/2**20,
               OptimizerStatesIncluded=False, OptimizerSteps=0)
    before = parameter_digest(model)
    try:
        image = torch.randn(32, 1, 256, 256, device='cuda')
        labels = torch.arange(32, device='cuda') % 3
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(image)
            loss = nn.CrossEntropyLoss()(logits, labels)
        require(tuple(logits.shape) == (32, 3) and bool(torch.isfinite(logits).all()), 'Invalid AMP logits')
        require(bool(torch.isfinite(loss)), 'Nonfinite AMP loss')
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Invalid AMP gradients')
        torch.cuda.synchronize()
        row.update(Status='PASS', Detail='Forward, CE loss and backward completed')
    except torch.cuda.OutOfMemoryError:
        row.update(Status='CUDA OOM', Detail='Batch size unchanged; fairness decision required before training')
    finally:
        row['PeakAllocatedMiB'] = torch.cuda.max_memory_allocated()/2**20
        row['PeakReservedMiB'] = torch.cuda.max_memory_reserved()/2**20
        row['IncrementalPeakMiB'] = row['PeakAllocatedMiB']-row['BaselineAllocatedMiB']
        model.zero_grad(set_to_none=True)
        del image, logits, loss, labels
        gc.collect()
        torch.cuda.empty_cache()
    require(parameter_digest(model) == before, 'Memory test modified parameters')
    row['ParametersUnchanged'] = True
    row['AdamWTwoMomentStatesMiBEstimate'] = sum(p.numel() for p in model.parameters())*8/2**20
    return row


def measure_latency(model):
    """FP32 batch-one synchronized wall time, matching Phase 5's 30/100 protocol."""
    model.eval()
    before = parameter_digest(model)
    image = torch.randn(1, 1, 256, 256, device='cuda')
    values = []
    with torch.inference_mode():
        for _ in range(30):
            model(image)
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(image)
            torch.cuda.synchronize()
            values.append((time.perf_counter()-start)*1000)
    require(parameter_digest(model) == before, 'Latency measurement modified parameters')
    return dict(BatchSize=1, Precision='FP32', Warmup=30, Iterations=100,
                MeanMs=float(np.mean(values)), SDMs=float(np.std(values, ddof=1)),
                MedianMs=float(np.median(values)), ImagesPerSecond=1000/float(np.mean(values))), values


def run_model(root, name):
    """Validate one disposable baseline; retain only measurements, never weights."""
    seed_everything()
    require(torch.cuda.is_available(), 'CUDA required for the requested batch-size safety check')
    model = build_baseline(name)
    adaptation = check_native_adaptation(name, model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, layers = measure_complexity(model)
    output = root / 'results/baselines/setup'
    layers.to_csv(output / f'{name}_layer_macs.csv', index=False)
    model.to('cuda')
    latency, timings = measure_latency(model)
    validation = validate_backward(model)
    memory = memory_batch32(model)
    summary = dict(Model=DISPLAY_NAMES[name], Role='Baseline', InputChannels=1, Classes=3, Pretrained=False,
                   Parameters=total, TrainableParameters=trainable, ParametersM=total/1e6,
                   TrainableParametersM=trainable/1e6, ConvLinearMACs=macs, ConvLinearFLOPs=2*macs,
                   FP32Bytes=total*4, FP32MB=total*4/1e6, FP32MiB=total*4/2**20)
    pd.DataFrame({'Iteration': range(1, 101), 'LatencyMs': timings}).to_csv(output / f'{name}_latency_raw.csv', index=False)
    result = dict(summary=summary, adaptation=adaptation, validation=validation,
                  memory={'Model': DISPLAY_NAMES[name], **memory}, latency={'Model': DISPLAY_NAMES[name], **latency})
    write_json(output / f'{name}_setup.json', result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f'{DISPLAY_NAMES[name]}: architecture/FP32 gradients PASS; batch 32 {memory["Status"]}; no optimizer step.')
    return result


def save_setup(root, results, data_audit):
    output = root / 'results/baselines/setup'
    reference = json.loads((root / 'results/complexity/complexity_summary.json').read_text())
    profile = reference['profiles']['inference']
    require(reference['total_parameters'] == 579599 and profile['conv_linear_flops'] == 551431936,
            'Frozen reference values mismatch')
    proposed = dict(Model='MorphoNet-BT', Role='Proposed reference (Phase 5)', InputChannels=1, Classes=3,
                    Pretrained=False, Parameters=reference['total_parameters'], TrainableParameters=reference['trainable_parameters'],
                    ParametersM=reference['total_parameters']/1e6, TrainableParametersM=reference['trainable_parameters']/1e6,
                    ConvLinearMACs=profile['independent_conv_linear_macs'], ConvLinearFLOPs=profile['conv_linear_flops'],
                    FP32Bytes=reference['storage']['fp32_parameter_bytes'], FP32MB=reference['storage']['fp32_MB'],
                    FP32MiB=reference['storage']['fp32_MiB'])
    summary = pd.DataFrame([proposed]+[results[name]['summary'] for name in BASELINES])
    memory = pd.DataFrame([results[name]['memory'] for name in BASELINES])
    latency = pd.DataFrame([results[name]['latency'] for name in BASELINES])
    summary.to_csv(output / 'baseline_architecture_summary.csv', index=False)
    summary[['Model', 'Role', 'ConvLinearMACs', 'ConvLinearFLOPs', 'FP32Bytes', 'FP32MB', 'FP32MiB']].to_csv(output / 'baseline_complexity.csv', index=False)
    memory.to_csv(output / 'baseline_memory_test.csv', index=False)
    latency.to_csv(output / 'baseline_latency.csv', index=False)
    write_json(output / 'baseline_validation.json', {name: results[name]['validation'] | results[name]['adaptation'] for name in BASELINES})
    metadata = dict(seed=42, python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), input_shape=[1, 1, 256, 256],
                    pretraining_policy='weights=None; from scratch only', optimizer_steps=0, full_training_started=False,
                    real_test_images_accessed=False, data_audit=data_audit,
                    augmentation_mask_policy='User approved internal mask use for frozen augmentation only; MRI and class labels exposed',
                    complexity_convention='Executed Conv2d/Linear MACs; FLOPs=2*MACs; all other operations excluded',
                    complexity_cross_check='Independent forward hooks equal THOP with all non-Conv/Linear rules zeroed',
                    storage_convention='FP32 parameter count times four bytes; MB=1e6, MiB=2**20; buffers/serialization excluded',
                    memory_protocol='Synthetic batch 32, CUDA FP16 autocast, FP32 parameters, CE backward, no optimizer/scaler',
                    memory_limitation='No optimizer state or optimizer-step temporary allocations; not a full-training memory guarantee',
                    latency_protocol='FP32, batch 1, 30 warmup, 100 synchronized wall-clock samples; hardware-specific',
                    deterministic_algorithms=True, cudnn_benchmark=False, tf32=False,
                    protected_files_unchanged=verify_protected(root),
                    sources={str(p.relative_to(root)): digest(p) for p in [root/'src/models/baselines.py', root/'src/baselines/data.py', root/'src/utils/baseline_setup.py', root/'configs/baselines.yaml']})
    write_json(output / 'baseline_setup_metadata.json', metadata)
    return summary, memory, latency
