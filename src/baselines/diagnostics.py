"""Bounded, disposable Phase 9 checks; never start a full training run."""
from copy import deepcopy
from pathlib import Path
import gc
import json
import math
import random
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.baselines.data import MRIOnlyBaselineDataset
from src.models.baselines import BASELINES, OUTPUT_LAYERS
from src.training.scheduler import EarlyStopping
from src.utils.checkpoint import atomic_torch_save, capture_rng, cpu_tree, restore_rng, seed_everything
from src.utils.training_metrics import classification_metrics
from src.baselines.training import (
    atomic_json, build_loaders, epoch_pass, file_hash, forward_loss, fresh_model,
    make_checkpoint, object_hash, optimizer_step, output_paths, prepare_paths,
    require, restore_checkpoint, run_identity, save_best, training_objects, train_selected,
)


def assert_tree_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    else:
        assert left == right


def protected_hashes(root):
    """Protect checkpoint bytes and frozen sources, without reading test prediction files."""
    paths = list((root / 'checkpoints/main').rglob('*.pt'))
    paths += [root/'configs/default.yaml', root/'configs/baselines.yaml', root/'src/models/baselines.py', root/'src/baselines/data.py']
    paths += list((root / 'src/data').glob('*.py'))
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def adamw_memory_preflights(info):
    """Two temporary batch-32 optimizer steps also measure initialized AdamW states."""
    require(torch.cuda.is_available(), 'CUDA is required for the real optimizer memory preflight')
    dataset = MRIOnlyBaselineDataset(info['root']/'splits/fold_1_train.csv', 'train', info['preprocessing'], 42)
    dataset.set_epoch(0)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    batch = next(iter(loader))
    device, rows = torch.device('cuda'), []
    for name in BASELINES:
        seed_everything(42)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = optimizer = scheduler = scaler = criterion = logits = loss = None
        row = dict(Model=name, BatchSize=32, AMP=True, Optimizer='AdamW', OptimizerSteps=0,
                   FullTraining=False, Data='One frozen training batch, reused only for disposable safety steps')
        try:
            model = fresh_model(name, info, device).train()
            optimizer, scheduler, scaler, criterion = training_objects(model, info, device)
            initial = model.get_submodule(OUTPUT_LAYERS[name]).weight.detach().clone()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                logits, loss = forward_loss(model, batch, criterion, device)
                optimizer_step(model, loss, optimizer, scaler)
                row['OptimizerSteps'] += 1
                optimizer.zero_grad(set_to_none=True)
            require(len(optimizer.state) == len(list(model.parameters())), 'AdamW state was not initialized for all parameters')
            require(all(int(state['step']) == 2 for state in optimizer.state.values()), 'Optimizer step skipped')
            require(not torch.equal(initial, model.get_submodule(OUTPUT_LAYERS[name]).weight), 'Disposable optimizer did not update weights')
            require(all(p.grad is None for p in model.parameters()), 'zero_grad(set_to_none=True) failed')
            row.update(Status='PASS', GradScalerScale=scaler.get_scale(), AdamWStatesAllocated=True)
            del initial
        except torch.cuda.OutOfMemoryError:
            row.update(Status='CUDA OOM', GradScalerScale=None, AdamWStatesAllocated=False)
        finally:
            row['PeakAllocatedMiB'] = torch.cuda.max_memory_allocated()/2**20
            row['PeakReservedMiB'] = torch.cuda.max_memory_reserved()/2**20
            del model, optimizer, scheduler, scaler, criterion, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        result = pd.DataFrame(rows)
        result.to_csv(info['root']/'results/baselines/training_setup/adamw_memory_preflight.csv', index=False)
        print(f'{name}: {row["Status"]}; batch 32; peak allocated {row["PeakAllocatedMiB"]:.1f} MiB; optimizer steps {row["OptimizerSteps"]}.')
        if row['Status'] != 'PASS':
            raise RuntimeError(f'{name}: CUDA OOM; full training blocked. Batch size was not changed.')
    return result


def _random_draws(generators, device):
    return (random.random(), np.random.rand(3), torch.rand(3), torch.rand(3, device=device).cpu(),
            {key: torch.randperm(12, generator=value) for key, value in generators.items()})


def isolated_resume_smoke(info):
    """Real one-batch epoch calls plus exact next-step comparison after checkpoint restore."""
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    identity = run_identity(info, 'resnet18', 1)
    model = fresh_model('resnet18', info, device)
    datasets, loaders, generators = build_loaders(info, 1)
    optimizer, scheduler, scaler, criterion = training_objects(model, info, device)
    stopper = EarlyStopping(patience=15, min_delta=0.)
    train, _ = epoch_pass(model, loaders['train'], criterion, device, info, 'resnet18', 1, 1,
                          optimizer, scaler, max_batches=1)
    validation, predictions = epoch_pass(model, loaders['val'], criterion, device, info, 'resnet18', 1, 1, max_batches=1)
    stopper.update(validation['macro_f1'], 1)
    scheduler.step()
    expected_lr = .001*(1+math.cos(math.pi/100))/2
    require(math.isclose(optimizer.param_groups[0]['lr'], expected_lr), 'Cosine schedule mismatch')
    history = [{'epoch': 1, 'diagnostic_only': True}]
    payload = make_checkpoint(model, optimizer, scheduler, scaler, stopper, generators, info, identity,
                              1, history, train, validation, 0.)
    payload['diagnostic_only'] = True
    payload['active_wall_seconds'] = 0.
    expected_random = _random_draws(generators, device)
    folder = info['root']/'results/baselines/training_setup'
    with tempfile.TemporaryDirectory(prefix='disposable_resume_', dir=folder) as temporary:
        path = Path(temporary)/'diagnostic.pt'
        atomic_torch_save(payload, path)
        clone = fresh_model('resnet18', info, device)
        opt2, sched2, scaler2, criterion2 = training_objects(clone, info, device)
        stop2 = EarlyStopping()
        gen2 = {key: torch.Generator().manual_seed(999) for key in generators}
        restore_checkpoint(path, clone, opt2, sched2, scaler2, stop2, gen2, info, identity, allow_diagnostic=True)
        assert_tree_equal(expected_random, _random_draws(gen2, device))
        for left, right in [(model.state_dict(), clone.state_dict()), (optimizer.state_dict(), opt2.state_dict()),
                            (scheduler.state_dict(), sched2.state_dict()), (scaler.state_dict(), scaler2.state_dict()),
                            (stopper.state_dict(), stop2.state_dict())]:
            assert_tree_equal(left, right)
        # The next update must match an uninterrupted continuation exactly.
        datasets['train'].set_epoch(1)
        batch = next(iter(DataLoader(datasets['train'], batch_size=32, shuffle=False, num_workers=0)))
        continuation_rng = capture_rng(generators)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = forward_loss(model, batch, criterion, device)
        optimizer_step(model, loss, optimizer, scaler)
        restore_rng(continuation_rng, gen2)
        clone.train()
        opt2.zero_grad(set_to_none=True)
        _, loss2 = forward_loss(clone, batch, criterion2, device)
        optimizer_step(clone, loss2, opt2, scaler2)
        assert_tree_equal(model.state_dict(), clone.state_dict())
        assert_tree_equal(optimizer.state_dict(), opt2.state_dict())
        assert_tree_equal(scaler.state_dict(), scaler2.state_dict())
        for key, value in [('configuration_fingerprint', 'incompatible'), ('fold', 2), ('architecture', 'resnet50')]:
            wrong = deepcopy(identity)
            wrong[key] = value
            try:
                restore_checkpoint(path, clone, opt2, sched2, scaler2, stop2, gen2, info, wrong, allow_diagnostic=True)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Resume accepted incompatible {key}')
        try:
            restore_checkpoint(path, clone, opt2, sched2, scaler2, stop2, gen2, info, identity)
        except ValueError:
            pass
        else:
            raise AssertionError('Diagnostic checkpoint accepted for real training')
        # Repair must preserve an already-correct best checkpoint byte-for-byte.
        paths = {'checkpoints': Path(temporary)/'checkpoints', 'results': Path(temporary)/'results'}
        save_best(payload, predictions.to_dict('records'), paths)
        before = file_hash(paths['checkpoints']/'best_macro_f1.pt')
        save_best(payload, predictions.to_dict('records'), paths, repair=True)
        require(file_hash(paths['checkpoints']/'best_macro_f1.pt') == before, 'Resume rewrote an unchanged best checkpoint')
        # Test-loader rejection occurs before any image can be read.
        class ForbiddenLoader:
            class dataset:
                class dataset:
                    mode = 'test'
            def __iter__(self):
                raise AssertionError('Forbidden loader was iterated')
        try:
            epoch_pass(clone, ForbiddenLoader(), criterion2, device, info, 'resnet18', 1, 1)
        except ValueError:
            pass
        else:
            raise AssertionError('Test loader was accepted')
    result = dict(status='PASS', model='resnet18', batch_size=32, num_workers=2, real_train_val_batches=1,
                  model_optimizer_scheduler_scaler_stopper_restored=True, rng_and_loader_generators_restored=True,
                  next_optimizer_step_bitwise_equal=True, incompatible_resume_rejected=True,
                  diagnostic_checkpoint_rejected_for_real_training=True, existing_best_checkpoint_preserved=True,
                  test_loader_rejected_before_iteration=True, diagnostic_checkpoints_deleted=True,
                  full_training_started=False, diagnostic_scores_not_reported=True)
    del model, clone, optimizer, opt2, scheduler, sched2, scaler, scaler2, criterion, criterion2, payload, batch, loss, loss2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def safety_logic_checks(info):
    """Exercise completed skips, incompatible settings, locks, archive and manual guards."""
    root = info['root']
    identity = run_identity(info, 'resnet18', 1)
    with tempfile.TemporaryDirectory(prefix='disposable_safety_', dir=root/'results/baselines/training_setup') as temporary:
        project = Path(temporary)
        paths, completed = prepare_paths(project, 'resnet18', 1, identity)
        sentinel = paths['checkpoints']/'sentinel.txt'
        sentinel.write_text('preserve', encoding='utf-8')
        try:
            prepare_paths(project, 'resnet18', 1, identity)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Incomplete run overwrite was accepted')
        marker = dict(identity=identity, checkpoint_sha256={'sentinel.txt': file_hash(sentinel)})
        atomic_json(marker, paths['results']/'training_complete.json')
        _, skipped = prepare_paths(project, 'resnet18', 1, identity)
        require(skipped == marker, 'Completed run was not skipped')
        wrong = deepcopy(identity)
        wrong['configuration_fingerprint'] = 'changed'
        try:
            prepare_paths(project, 'resnet18', 1, wrong)
        except ValueError:
            pass
        else:
            raise AssertionError('Incompatible completed run was silently skipped')
        lock = paths['results']/'.training.lock'
        lock.write_text('diagnostic', encoding='utf-8')
        try:
            prepare_paths(project, 'resnet18', 1, identity, overwrite=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Locked run was overwritten')
        lock.unlink()
        prepare_paths(project, 'resnet18', 1, identity, overwrite=True)
        archived = list((project/'results/baselines/run_archives').rglob('sentinel.txt'))
        require(len(archived) == 1 and archived[0].read_text() == 'preserve', 'Explicit overwrite did not preserve archived files')
    try:
        train_selected(root)
    except PermissionError:
        pass
    else:
        raise AssertionError('Full training started without manual confirmation')
    predictions = pd.DataFrame({'true_label': [0, 0, 1, 2], 'predicted_label': [0, 1, 1, 2]})
    require(math.isclose(classification_metrics(predictions)['macro_f1'], 7/9), 'Macro-F1 formula mismatch')
    stopper = EarlyStopping(patience=2, min_delta=0.)
    require(stopper.update(.5, 1), 'First score not accepted')
    require(not stopper.update(.5, 2) and stopper.bad_epochs == 1, 'Tie reset patience')
    require(not stopper.update(.4, 3) and stopper.stopped, 'Early stopping failed')
    changed = deepcopy(info['contract'])
    changed['training']['optimizer']['learning_rate'] = .002
    require(object_hash(changed) != info['configuration_fingerprint'], 'Fingerprint ignores learning rate')
    fingerprints = {run_identity(info, name, fold)['configuration_fingerprint'] for name in BASELINES for fold in range(1, 6)}
    require(len(fingerprints) == 1, 'Runs do not share one configuration fingerprint')
    return dict(status='PASS', incomplete_run_protected=True, completed_run_skipped=True,
                incompatible_completed_run_rejected=True, active_lock_protected=True, explicit_overwrite_archives=True,
                manual_confirmation_required=True, early_stopping_ties_and_patience=True, macro_f1_formula=True,
                changed_learning_rate_changes_fingerprint=True, identical_configuration_fingerprint_for_25_runs=True)
