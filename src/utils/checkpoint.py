"""Atomic checkpoints with optimizer/scaler/scheduler/RNG and epoch-boundary resume."""
from pathlib import Path
import os
import uuid
import random
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng(generators):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loaders={key:value.get_state() for key,value in generators.items()})


def restore_rng(state, generators):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA checkpoint cannot resume exactly on this CPU-only host')
        torch.cuda.set_rng_state_all(state['cuda'])
    for key,value in state['loaders'].items(): generators[key].set_state(value)


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value


def atomic_torch_save(payload, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        torch.save(payload,temporary)
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()


def load_checkpoint(path, model, optimizer, scheduler, scaler, stopper, generators,
                    config, fold, architecture_fingerprint, split_hashes):
    # Only load the project's own trusted checkpoints: RNG/NumPy states require full pickle.
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['configuration'] != config or saved['fold'] != fold:
        raise ValueError('Resume configuration/fold mismatch; refusing to restart or alter the schedule')
    if saved['architecture_fingerprint'] != architecture_fingerprint or saved['split_hashes'] != split_hashes:
        raise ValueError('Checkpoint architecture or split fingerprint mismatch')
    if saved['preprocessing_version'] != 'MorphoNet-BT Preprocessing v1.0':
        raise ValueError('Preprocessing version mismatch')
    model.load_state_dict(saved['model_state_dict'],strict=True)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    scheduler.load_state_dict(saved['scheduler_state_dict'])
    scaler.load_state_dict(saved['scaler_state_dict'])
    stopper.load_state_dict(saved['early_stopping_state'])
    if scheduler.last_epoch != saved['epoch']:
        raise ValueError('Scheduler epoch does not match completed checkpoint epoch')
    if optimizer.param_groups[0]['lr'] != scheduler.get_last_lr()[0]:
        raise ValueError('Optimizer and scheduler LR disagree')
    if len(saved['history']) != saved['epoch']:
        raise ValueError('Checkpoint log history is incomplete')
    restore_rng(saved['rng_state'],generators)
    return saved
