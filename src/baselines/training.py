"""Shared baseline training protocol, with explicit manual start and locked test data."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
import gc
import hashlib
import json
import os
import platform
import shutil
import time
import uuid
import warnings

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.baselines.data import MRIOnlyBaselineDataset
from src.data.dataset import verify_frozen_splits
from src.data.kfold import audit_splits, load_manifest, reload_splits
from src.models.baselines import BASELINES, INPUT_LAYERS, OUTPUT_LAYERS, build_baseline
from src.training.scheduler import EarlyStopping
from src.utils.checkpoint import seed_everything, capture_rng, restore_rng, cpu_tree, atomic_torch_save
from src.utils.training_metrics import classification_metrics

CLASS_MAPPING = {0: 'Meningioma', 1: 'Glioma', 2: 'Pituitary'}
PREPROCESSING_VERSION = 'MorphoNet-BT Preprocessing v1.0'
LEGACY_TRAINER_SHA256 = '5b98546edb45c63abfdee01d9d6e46df831250bab24e43a5c62eb2ee06d1f1d7'
IMPLEMENTATION_FIX = 'AMP GradScaler overflow handling, 2026-10-06'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compatible_identity(saved, current):
    """Allow only the audited pre-fix trainer hash; all scientific fields stay strict.

    Existing checkpoints and completion records remain immutable. Arbitrary source,
    architecture, split, runtime, or configuration differences still fail closed.
    """
    if saved == current:
        return True
    adapted = deepcopy(saved)
    sources = adapted.get('trainer_sources', {})
    trainer = 'src/baselines/training.py'
    if (sources.get(trainer) != LEGACY_TRAINER_SHA256 or
            current.get('trainer_sources', {}).get(trainer) != file_hash(__file__)):
        return False
    sources[trainer] = current['trainer_sources'][trainer]
    return adapted == current


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def runtime():
    return dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                numpy=np.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')


def preflight(root):
    """Read only frozen configs/sources and split metadata, never test images/predictions."""
    root = Path(root).resolve()
    setup = json.loads((root / 'results/baselines/setup/baseline_setup_metadata.json').read_text())
    for name, expected in setup['sources'].items():
        require(file_hash(root / name) == expected, f'Phase 8 source/config changed: {name}')
    config = yaml.safe_load((root / 'configs/baselines.yaml').read_text())
    require(tuple(config['models']) == BASELINES, 'Baseline architecture list changed')
    require(config['pretrained'] is False and config['weights'] is None, 'Pretrained weights forbidden')
    require(config['input_channels'] == 1 and config['num_classes'] == 3, 'Input/output protocol mismatch')
    require(config['class_mapping'] == CLASS_MAPPING and config['seed'] == 42, 'Seed/class mapping mismatch')
    cfg = config['training']
    require(cfg['max_epochs'] == 100 and cfg['batch_size'] == 32 and not cfg['persistent_workers'], 'Epoch/batch/worker protocol mismatch')
    require(cfg['loss'] == {'name': 'CrossEntropyLoss', 'class_weights': None}, 'Classification-only loss required')
    require(cfg['max_grad_norm'] is None and cfg['amp'] is True, 'Clipping/AMP protocol mismatch')
    frozen = verify_frozen_splits(root)
    prior = json.loads((root / 'results/architecture/frozen_input_hashes.json').read_text())
    require(prior['splits'] == frozen, 'Frozen split mismatch')
    for name, expected in prior['preprocessing_sources'].items():
        require(file_hash(root / name) == expected, f'Frozen preprocessing source changed: {name}')
    preprocessing = yaml.safe_load((root / 'results/architecture/preprocessing_v1_approved.yaml').read_text())
    main = yaml.safe_load((root / 'configs/default.yaml').read_text())
    require({key: main[key] for key in preprocessing} == preprocessing, 'Approved preprocessing mismatch')
    for key in ['max_epochs', 'batch_size', 'optimizer', 'early_stopping', 'amp', 'grad_scaler_init_scale', 'max_grad_norm', 'num_workers', 'persistent_workers']:
        require(cfg[key] == main['training'][key], f'Shared hyperparameter mismatch: {key}')
    require(cfg['scheduler'] == {'name': 'CosineAnnealingLR', 'T_max': 100, 'eta_min': 0.0}, 'Scheduler mismatch')
    splits = reload_splits(root / 'splits')
    audits, coverage = audit_splits(splits, load_manifest(root / 'results/dataset_statistics.csv'))
    relevant_training = {key: value for key, value in cfg.items() if key not in ('enabled', 'requires_phase9_approval')}
    contract = dict(seed=42, training=relevant_training, gradient_accumulation=False, preprocessing=preprocessing,
                    class_mapping=CLASS_MAPPING, input_channels=1, num_classes=3, pretrained=False,
                    factory_sha256=file_hash(root / 'src/models/baselines.py'),
                    dataset_sha256=file_hash(root / 'src/baselines/data.py'))
    source_paths = ['src/baselines/training.py', 'src/utils/checkpoint.py', 'src/training/scheduler.py',
                    'src/utils/training_metrics.py', 'src/data/inspection.py']
    sources = {name: file_hash(root / name) for name in source_paths}
    return dict(root=root, config=config, preprocessing=preprocessing, split_hashes=frozen,
                split_audit=audits, coverage=coverage, contract=contract,
                configuration_fingerprint=object_hash(contract), trainer_sources=sources, runtime=runtime())


def run_identity(info, name, fold):
    require(name in BASELINES and type(fold) is int and fold in range(1, 6), 'Invalid architecture/fold')
    return dict(architecture=name, fold=fold, configuration_fingerprint=info['configuration_fingerprint'],
                split_hashes=info['split_hashes'], trainer_sources=info['trainer_sources'], runtime=info['runtime'])


def fresh_model(name, info, device):
    model = build_baseline(name, num_classes=3, input_channels=1, pretrained=False)
    require(model.get_submodule(INPUT_LAYERS[name]).in_channels == 1, 'Factory input mismatch')
    require(model.get_submodule(OUTPUT_LAYERS[name]).out_features == 3, 'Factory output mismatch')
    expected = json.loads((info['root'] / f'results/baselines/setup/{name}_setup.json').read_text())['summary']['Parameters']
    require(sum(p.numel() for p in model.parameters()) == expected, 'Phase 8 parameter count mismatch')
    return model.to(device)


def build_loaders(info, fold):
    """There is intentionally no test-loader option."""
    require(type(fold) is int and fold in range(1, 6), 'Invalid fold')
    cfg = info['config']['training']
    generators = {part: torch.Generator().manual_seed(42) for part in ('train', 'val')}
    datasets = {part: MRIOnlyBaselineDataset(info['root'] / f'splits/fold_{fold}_{part}.csv', part,
                                            config=info['preprocessing'], seed=42) for part in ('train', 'val')}
    loaders = {part: DataLoader(data, batch_size=32, shuffle=part == 'train', num_workers=cfg['num_workers'],
                                persistent_workers=False, drop_last=False, pin_memory=torch.cuda.is_available(),
                                generator=generators[part]) for part, data in datasets.items()}
    return datasets, loaders, generators


def training_objects(model, info, device):
    cfg = info['config']['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['amp'], init_scale=cfg['grad_scaler_init_scale'])
    return optimizer, scheduler, scaler, nn.CrossEntropyLoss()


def forward_loss(model, batch, criterion, device, amp=True):
    require(set(batch) == {'image', 'label', 'pid', 'sample_id'}, 'Only MRI, class label and identifiers are allowed')
    images = batch['image'].to(device, non_blocking=True)
    labels = batch['label'].to(device, non_blocking=True)
    require(tuple(images.shape[1:]) == (1, 256, 256) and images.dtype == torch.float32, 'Invalid MRI input')
    context = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' and amp else nullcontext()
    with context:
        logits = model(images)
        require(tuple(logits.shape) == (len(labels), 3), 'Invalid logits shape')
        loss = criterion(logits, labels)
    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)), 'Nonfinite logits/loss')
    return logits, loss


def optimizer_step(model, loss, optimizer, scaler):
    """Let enabled GradScaler skip overflowed steps and adapt its scale normally.

    Finite forward values are checked separately. Missing gradients and nonfinite
    FP32 gradients still abort. No clipping, LR changes, or architecture exception
    is introduced; the same implementation applies to every baseline.
    """
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nonfinite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise FloatingPointError(f'Missing gradient: {name}')
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
    if nonfinite and not scaler.is_enabled():
        raise FloatingPointError(f'Nonfinite FP32 gradient: {nonfinite[0]}')
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    require(np.isfinite(scale_after) and scale_after > 0, 'Invalid AMP scale; numerical failure requires review')
    if nonfinite:
        require(scale_after < scale_before, 'GradScaler did not reduce the overflowed scale')
    return dict(optimizer_step_applied=not nonfinite, scale_before=scale_before,
                scale_after=scale_after, first_nonfinite_parameter=nonfinite[0] if nonfinite else None)


def epoch_pass(model, loader, criterion, device, info, name, fold, epoch, optimizer=None, scaler=None, max_batches=None):
    """Whole-split train/validation metrics; bounded calls are diagnostic only."""
    training = optimizer is not None
    mode = 'train' if training else 'val'
    require(loader.dataset.dataset.mode == mode, 'Epoch loop refuses incorrect partitions, including test')
    model.train(training)
    if training:
        loader.dataset.set_epoch(epoch - 1)
    rows, total_loss, count = [], 0., 0
    optimizer_steps, amp_events = 0, []
    with nullcontext() if training else torch.inference_mode():
        batches = loader if max_batches is None else islice(loader, max_batches)
        for batch_index, batch in enumerate(batches, start=1):
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, loss = forward_loss(model, batch, criterion, device, info['config']['training']['amp'])
            if training:
                step = optimizer_step(model, loss, optimizer, scaler)
                optimizer_steps += int(step['optimizer_step_applied'])
                if not step['optimizer_step_applied']:
                    amp_events.append(dict(batch=batch_index, **step))
                    warnings.warn(f'{name} fold {fold} epoch {epoch} batch {batch_index}: '
                                  f'AMP overflow; optimizer update skipped; scale '
                                  f'{step["scale_before"]:g} -> {step["scale_after"]:g}.', RuntimeWarning)
            probability = logits.detach().float().softmax(1).cpu().numpy()
            labels = batch['label'].tolist()
            for index, label in enumerate(labels):
                rows.append(dict(PID=batch['pid'][index], sample_id=batch['sample_id'][index],
                                 true_label=label, predicted_label=int(probability[index].argmax()),
                                 prob_meningioma=float(probability[index, 0]), prob_glioma=float(probability[index, 1]),
                                 prob_pituitary=float(probability[index, 2]), model=name, fold=fold, epoch=epoch, split=mode))
            total_loss += float(loss.detach()) * len(labels)
            count += len(labels)
    require(count > 0, 'Empty epoch')
    predictions = pd.DataFrame(rows)
    require(predictions.sample_id.is_unique, 'Duplicate sample within epoch')
    if max_batches is None:
        require(set(predictions.sample_id) == set(loader.dataset.frame.sample_id), 'Incomplete epoch sample coverage')
    metrics = classification_metrics(predictions)
    metrics.update(loss=total_loss/count, num_samples=count, diagnostic_partial_epoch=max_batches is not None)
    if training:
        require(optimizer_steps > 0, 'No optimizer update succeeded in this epoch; numerical failure requires review')
        metrics.update(optimizer_steps=optimizer_steps, amp_skipped_steps=len(amp_events), amp_events=amp_events)
    return metrics, predictions


def output_paths(root, name, fold):
    require(name in BASELINES and type(fold) is int and fold in range(1, 6), 'Invalid output identity')
    root = Path(root).resolve()
    return dict(checkpoints=root / f'checkpoints/baselines/{name}/fold_{fold}',
                results=root / f'results/baselines/training/{name}/fold_{fold}')


def prepare_paths(root, name, fold, identity, resume=False, overwrite=False):
    """Skip compatible completed runs; protect incomplete runs and archive explicit overwrites."""
    require(not (resume and overwrite), 'Resume and overwrite cannot both be enabled')
    paths = output_paths(root, name, fold)
    marker = paths['results'] / 'training_complete.json'
    lock = paths['results'] / '.training.lock'
    if lock.exists():
        raise RuntimeError('Run lock exists. Verify the previous process has stopped before removing the stale lock.')
    if marker.exists():
        saved = json.loads(marker.read_text())
        require(compatible_identity(saved['identity'], identity), 'Completed run has incompatible settings; refusing to skip or overwrite')
        for filename, digest in saved['checkpoint_sha256'].items():
            require(file_hash(paths['checkpoints'] / filename) == digest, 'Completed checkpoint changed')
        if not overwrite:
            return paths, saved
    occupied = [path for path in paths.values() if path.exists() and any(path.iterdir())]
    if occupied and not (resume or overwrite):
        raise FileExistsError('Incomplete run exists; use RESUME=True. Existing data will not be overwritten.')
    if resume:
        require((paths['checkpoints'] / 'last_checkpoint.pt').is_file(), 'No last checkpoint available for resume')
    if overwrite and occupied:
        root = Path(root).resolve()
        archive = root / 'results/baselines/run_archives' / f'{name}_fold_{fold}_{uuid.uuid4().hex}'
        for source in occupied:
            require(source.resolve().is_relative_to(root) and archive.resolve().is_relative_to(root), 'Archive path escapes workspace')
        archive.mkdir(parents=True, exist_ok=False)
        for key, source in paths.items():
            if source.exists():
                shutil.move(str(source), str(archive / key))
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths, None


def make_checkpoint(model, optimizer, scheduler, scaler, stopper, generators, info, identity, epoch, history, train, validation, elapsed):
    return dict(model_state_dict=cpu_tree(model.state_dict()), optimizer_state_dict=cpu_tree(optimizer.state_dict()),
                scheduler_state_dict=deepcopy(scheduler.state_dict()), scaler_state_dict=deepcopy(scaler.state_dict()),
                early_stopping_state=stopper.state_dict(), rng_state=capture_rng(generators),
                identity=deepcopy(identity), configuration=deepcopy(info['config']), preprocessing=deepcopy(info['preprocessing']),
                preprocessing_version=PREPROCESSING_VERSION, class_mapping=CLASS_MAPPING, seed=42,
                architecture=identity['architecture'], fold=identity['fold'], epoch=epoch,
                best_epoch=stopper.best_epoch, best_validation_macro_f1=stopper.best_metric,
                training_metrics=deepcopy(train), validation_metrics=deepcopy(validation), history=deepcopy(history),
                training_seconds=elapsed, saved_at=timestamp())


def restore_checkpoint(path, model, optimizer, scheduler, scaler, stopper, generators, info, identity, *, allow_diagnostic=False):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(allow_diagnostic or not saved.get('diagnostic_only', False), 'Diagnostic checkpoint cannot resume a real experiment')
    require(compatible_identity(saved['identity'], identity), 'Resume architecture/fold/configuration/source/runtime mismatch')
    require(saved['configuration'] == info['config'] and saved['preprocessing'] == info['preprocessing'], 'Resume configuration mismatch')
    require(saved['class_mapping'] == CLASS_MAPPING and saved['preprocessing_version'] == PREPROCESSING_VERSION, 'Resume preprocessing/class mismatch')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    scheduler.load_state_dict(saved['scheduler_state_dict'])
    scaler.load_state_dict(saved['scaler_state_dict'])
    stopper.load_state_dict(saved['early_stopping_state'])
    require(scheduler.last_epoch == saved['epoch'] == len(saved['history']), 'Resume epoch/scheduler/history mismatch')
    require(optimizer.param_groups[0]['lr'] == scheduler.get_last_lr()[0], 'Resume LR mismatch')
    require([row['epoch'] for row in saved['history']] == list(range(1, saved['epoch']+1)), 'Resume history is not contiguous')
    restore_rng(saved['rng_state'], generators)
    return saved


def save_best(best, predictions, paths, repair=False):
    path = paths['checkpoints'] / 'best_macro_f1.pt'
    unchanged = False
    if repair and path.exists():
        existing = torch.load(path, map_location='cpu', weights_only=False)
        require(compatible_identity(existing['identity'], best['identity']), 'Best checkpoint identity mismatch')
        require(existing['best_validation_macro_f1'] <= best['best_validation_macro_f1'], 'Best checkpoint newer/better than committed last; manual review required')
        if existing['epoch'] == best['epoch']:
            require(all(torch.equal(value, existing['model_state_dict'][key]) for key, value in best['model_state_dict'].items()),
                    'Best checkpoint model-state mismatch')
            unchanged = True
    if not unchanged:
        atomic_torch_save(best, path)
    atomic_json(dict(epoch=best['epoch'], **best['validation_metrics']), paths['results'] / 'best_validation_summary.json')
    atomic_csv(pd.DataFrame(predictions), paths['results'] / 'best_validation_predictions.csv')


def plot_curves(history, path):
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout='constrained')
    for axis, metric, label in zip(axes, ['loss', 'macro_f1'], ['Cross-entropy loss', 'Macro F1']):
        for part in ['train', 'val']:
            axis.plot(frame.epoch, frame[f'{part}_{metric}'], label='Training' if part == 'train' else 'Validation')
        axis.set(xlabel='Epoch', ylabel=label)
        axis.legend(frameon=False)
    figure.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(figure)


def require_prepared(info):
    path = info['root'] / 'results/baselines/training_setup/preparation_audit.json'
    require(path.is_file(), 'Run the Phase 9 preparation cells first')
    audit = json.loads(path.read_text())
    require(audit['configuration_fingerprint'] == info['configuration_fingerprint'], 'Preparation fingerprint changed')
    require(audit['trainer_sources'] == info['trainer_sources'] and audit['runtime'] == info['runtime'], 'Preparation source/runtime changed')
    require(audit['all_adamw_preflights_passed'] and audit['resume_validated'] and audit['overwrite_protection_validated'],
            'Preparation checks not passed')


def train_run(root, name, fold, *, confirm_full_training=False, resume=False, overwrite=False):
    if confirm_full_training is not True:
        raise PermissionError('Full training requires manual CONFIRM_FULL_TRAINING=True')
    info = preflight(root)
    require_prepared(info)
    identity = run_identity(info, name, fold)
    paths, completed = prepare_paths(root, name, fold, identity, resume, overwrite)
    if completed is not None:
        print(f'{name} fold {fold}: compatible completed run skipped.')
        return completed
    lock = paths['results'] / '.training.lock'
    with lock.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), started_at=timestamp()), handle)
    try:
        seed_everything(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = fresh_model(name, info, device)
        datasets, loaders, generators = build_loaders(info, fold)
        optimizer, scheduler, scaler, criterion = training_objects(model, info, device)
        stopper = EarlyStopping(patience=15, min_delta=0.)
        history, best, best_predictions, elapsed, start_epoch = [], None, [], 0., 1
        prior_wall_seconds = 0.
        session_started = time.perf_counter()
        if resume:
            saved = restore_checkpoint(paths['checkpoints'] / 'last_checkpoint.pt', model, optimizer, scheduler, scaler,
                                       stopper, generators, info, identity)
            history, elapsed, start_epoch = saved['history'], saved['training_seconds'], saved['epoch']+1
            prior_wall_seconds = saved['active_wall_seconds']
            best, best_predictions = saved['best_checkpoint'], saved['best_predictions']
            require(best is not None, 'Committed last checkpoint is missing its best snapshot')
            save_best(best, best_predictions, paths, repair=True)
            atomic_csv(pd.DataFrame(history), paths['results'] / 'training_log.csv')
            del saved
        metadata = dict(identity=identity, configuration=info['config'], preprocessing=info['preprocessing'],
                        preprocessing_version=PREPROCESSING_VERSION, class_mapping=CLASS_MAPPING, seed=42,
                        from_scratch=True, weights=None, gradient_accumulation=False, gradient_clipping=False,
                        test_access='Split CSV metadata only; no images or predictions', started_at=timestamp(),
                        checkpoint_selection='Highest slice-level validation macro-F1', resumed=resume,
                        implementation_fix=IMPLEMENTATION_FIX,
                        amp_policy='GradScaler skips nonfinite updates and reduces scale; skipped updates are logged')
        metadata_name = f'resume_{uuid.uuid4().hex}.json' if resume else 'training_metadata.json'
        atomic_json(metadata, paths['results'] / metadata_name)
        for epoch in range(start_epoch, 101):
            if stopper.stopped:
                break
            started = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            train, _ = epoch_pass(model, loaders['train'], criterion, device, info, name, fold, epoch, optimizer, scaler)
            validation, predictions = epoch_pass(model, loaders['val'], criterion, device, info, name, fold, epoch)
            improved = stopper.update(validation['macro_f1'], epoch)
            scheduler.step()
            row = dict(epoch=epoch, learning_rate=lr, next_learning_rate=optimizer.param_groups[0]['lr'],
                       epoch_duration_seconds=time.perf_counter()-started, amp_scale=scaler.get_scale())
            row.update(train_optimizer_steps=train['optimizer_steps'], train_amp_skipped_steps=train['amp_skipped_steps'])
            for part, metrics in [('train', train), ('val', validation)]:
                row.update({f'{part}_{key}': metrics[key] for key in ['loss', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1']})
            history.append(row)
            elapsed += row['epoch_duration_seconds']
            payload = make_checkpoint(model, optimizer, scheduler, scaler, stopper, generators, info, identity,
                                      epoch, history, train, validation, elapsed)
            payload['active_wall_seconds'] = prior_wall_seconds + time.perf_counter() - session_started
            if improved:
                best, best_predictions = deepcopy(payload), predictions.to_dict('records')
            # Commit last first; embedded best repairs an interrupted secondary save.
            atomic_torch_save({**payload, 'best_checkpoint': best, 'best_predictions': best_predictions},
                              paths['checkpoints'] / 'last_checkpoint.pt')
            if improved:
                save_best(best, best_predictions, paths)
            atomic_csv(pd.DataFrame(history), paths['results'] / 'training_log.csv')
            atomic_json(dict(train=train, validation=validation), paths['results'] / f'epoch_metrics/epoch_{epoch:03d}.json')
            print(f'{name} | fold {fold} | epoch {epoch}/100 | LR {lr:.6g} | '
                  f'train loss {train["loss"]:.4f}, F1 {train["macro_f1"]:.4f} | '
                  f'val loss {validation["loss"]:.4f}, F1 {validation["macro_f1"]:.4f} | '
                  f'patience {stopper.bad_epochs}/15 | AMP skipped {train["amp_skipped_steps"]}')
            del payload
        require(best is not None and history, 'No completed epoch; completion marker not written')
        plot_curves(history, paths['results'] / 'training_curves.png')
        marker = dict(identity=identity, model=name, fold=fold, best_epoch=best['epoch'],
                      best_validation_accuracy=best['validation_metrics']['accuracy'],
                      best_validation_macro_f1=stopper.best_metric, total_epochs=len(history), early_stopping=stopper.stopped,
                      training_seconds=elapsed, training_time_definition='Sum of completed epoch train/validation compute durations; excludes downtime and checkpoint I/O',
                      active_wall_seconds=prior_wall_seconds + time.perf_counter() - session_started,
                      checkpoint_path=str(paths['checkpoints'] / 'best_macro_f1.pt'),
                      checkpoint_sha256={filename: file_hash(paths['checkpoints'] / filename) for filename in ['best_macro_f1.pt', 'last_checkpoint.pt']},
                      completed_at=timestamp(), test_evaluated=False)
        atomic_json(marker, paths['results'] / 'training_complete.json')
        return marker
    finally:
        lock.unlink(missing_ok=True)


def completion_summary(root, info=None):
    """Write summary files only for real completed runs; never fabricate empty results."""
    info = preflight(root) if info is None else info
    rows = []
    for name in BASELINES:
        for fold in range(1, 6):
            path = output_paths(root, name, fold)['results'] / 'training_complete.json'
            if not path.exists():
                continue
            marker = json.loads(path.read_text())
            require(compatible_identity(marker['identity'], run_identity(info, name, fold)), 'Incompatible completed run in summary')
            rows.append({'Model': name, 'Fold': fold, 'Best Epoch': marker['best_epoch'],
                         'Best Validation Accuracy': marker['best_validation_accuracy'],
                         'Best Validation Macro-F1': marker['best_validation_macro_f1'],
                         'Total Epochs': marker['total_epochs'], 'Early Stopping': marker['early_stopping'],
                         'Training Time Seconds': marker['training_seconds'], 'Checkpoint Path': marker['checkpoint_path']})
            rows[-1]['Active Run Wall Seconds'] = marker['active_wall_seconds']
    table = pd.DataFrame(rows)
    if rows:
        folder = Path(root) / 'results/baselines/training'
        atomic_csv(table, folder / 'baseline_training_summary.csv')
        model_rows = []
        for name, group in table.groupby('Model'):
            if set(group.Fold) == set(range(1, 6)):
                model_rows.append(dict(Model=name, ValidationAccuracyMean=group['Best Validation Accuracy'].mean(),
                                       ValidationAccuracySD=group['Best Validation Accuracy'].std(ddof=1),
                                       ValidationMacroF1Mean=group['Best Validation Macro-F1'].mean(),
                                       ValidationMacroF1SD=group['Best Validation Macro-F1'].std(ddof=1),
                                       TrainingTimeSeconds=group['Training Time Seconds'].sum()))
        if model_rows:
            atomic_csv(pd.DataFrame(model_rows), folder / 'model_validation_summary.csv')
    return table


def train_selected(root, models=('resnet18',), folds=(1,), *, confirm_full_training=False, resume=False, overwrite=False):
    if confirm_full_training is not True:
        raise PermissionError('Set CONFIRM_FULL_TRAINING=True in the manual cell to start selected runs')
    require(models and len(models) == len(set(models)) and all(name in BASELINES for name in models), 'Select unique supported models')
    require(folds and len(folds) == len(set(folds)) and all(type(fold) is int and fold in range(1, 6) for fold in folds), 'Select unique folds 1-5')
    results = []
    for name in models:
        for fold in folds:
            results.append(train_run(root, name, fold, confirm_full_training=True, resume=resume, overwrite=overwrite))
            completion_summary(root)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results
