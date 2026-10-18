"""Reload-safe manual launch of all folds, preserving completed experiments."""
import gc
import importlib
import json
from pathlib import Path

import pandas as pd
import torch


def load_current_training(root):
    """Replace stale notebook module definitions with the verified on-disk code."""
    root = Path(root).resolve()
    importlib.invalidate_caches()
    module = importlib.import_module('src.baselines.training')
    if Path(module.__file__).resolve() != root / 'src/baselines/training.py':
        raise RuntimeError('A training module from another project is loaded; restart the kernel')
    module = importlib.reload(module)
    if getattr(module, 'IMPLEMENTATION_FIX', None) != 'AMP GradScaler overflow handling, 2026-10-06':
        raise RuntimeError('The required AMP correction is missing')
    info = module.preflight(root)
    module.require_prepared(info)
    return module, info


def fold_plan(module, info, model_name, folds):
    """Select actions from durable files, never a global overwrite switch."""
    rows = []
    for fold in folds:
        identity = module.run_identity(info, model_name, fold)
        paths = module.output_paths(info['root'], model_name, fold)
        if (paths['results'] / '.training.lock').exists():
            raise RuntimeError(f'Fold {fold} is locked; verify the old process has stopped before reviewing its lock')
        marker = paths['results'] / 'training_complete.json'
        last = paths['checkpoints'] / 'last_checkpoint.pt'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if not module.compatible_identity(saved['identity'], identity):
                raise ValueError(f'Fold {fold} completed under incompatible settings')
            for name, digest in saved['checkpoint_sha256'].items():
                if module.file_hash(paths['checkpoints'] / name) != digest:
                    raise ValueError(f'Fold {fold} completed checkpoint changed')
            action = 'Skip completed'
        elif last.exists():
            action = 'Resume checkpoint'
        else:
            checkpoint_files = list(paths['checkpoints'].iterdir()) if paths['checkpoints'].exists() else []
            result_files = list(paths['results'].iterdir()) if paths['results'].exists() else []
            if not checkpoint_files and not result_files:
                action = 'Start new'
            elif not checkpoint_files and {path.name for path in result_files} == {'training_metadata.json'}:
                saved = json.loads((paths['results'] / 'training_metadata.json').read_text())
                if not module.compatible_identity(saved['identity'], identity):
                    raise ValueError(f'Fold {fold} metadata has incompatible settings')
                action = 'Archive metadata-only attempt; start new'
            else:
                raise RuntimeError(f'Fold {fold} has partial artifacts but no last checkpoint; review required')
        rows.append({'Model': model_name, 'Fold': fold, 'Action': action})
    return pd.DataFrame(rows)


def train_model_folds(root, model_name='efficientnet_b0', folds=(1, 2, 3, 4, 5), *, confirm_full_training=False):
    """Preview by default; explicit confirmation launches selected folds sequentially.

    Completed checkpoints are never overwritten. Only a failed metadata-only
    attempt is automatically archived. Real partial runs resume their last epoch.
    """
    module, info = load_current_training(root)
    folds = tuple(folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError('Select unique fold numbers')
    plan = fold_plan(module, info, model_name, folds)
    print('Loaded corrected training code from:', module.__file__)
    print(plan.to_string(index=False))
    if confirm_full_training is not True:
        print('Preview only. No run was started, resumed, or archived.')
        return plan
    for row in plan.itertuples(index=False):
        module.train_run(
            info['root'], model_name, row.Fold, confirm_full_training=True,
            resume=row.Action == 'Resume checkpoint',
            overwrite=row.Action == 'Archive metadata-only attempt; start new',
        )
        module.completion_summary(info['root'])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return module.completion_summary(info['root'])
