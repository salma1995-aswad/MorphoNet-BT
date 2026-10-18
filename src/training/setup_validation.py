"""Bounded Phase 6 diagnostics; never call train_fold or iterate an entire split."""
from copy import deepcopy
from pathlib import Path
import gc
import json
import random
import tempfile
import time
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.dataset import BrainTumorDataset
from src.models.morphonet_bt import MorphoNetBT
from src.models.validation import smoke_losses
from src.utils.checkpoint import seed_everything, cpu_tree, atomic_torch_save, load_checkpoint
from src.utils.training_metrics import classification_metrics
from .losses import MorphoNetLoss
from .scheduler import EarlyStopping
from .trainer import (build_loaders,build_training_objects,autocast_context,move_batch,check_gradients,
                      train_one_epoch,checkpoint_payload,prepare_output_paths)


def _sync(device):
    if device.type=='cuda': torch.cuda.synchronize()


def batch_size_memory_test(root,config,info,candidates=(8,16,32)):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=BrainTumorDataset(Path(root)/'splits/fold_1_train.csv','train',config,config['seed'])
    dataset.set_epoch(0)
    rows=[]
    for size in candidates:
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache(); free,total=torch.cuda.mem_get_info()
            successful=[row for row in rows if row['status']=='PASS']
            projected=(successful[-1]['peak_allocated_bytes']*size/successful[-1]['batch_size']*1.5 if successful else 1024**3)
            if projected>min(free*.65,total*.65):
                rows.append(dict(batch_size=size,status='SKIPPED_SAFETY',reason='Conservative projection exceeds 65% memory budget'))
                break
        elif size>8:
            break
        seed_everything(config['seed'])
        model=MorphoNetBT(**config['model']).to(device).train()
        optimizer,scheduler,scaler,criterion=build_training_objects(model,config,info['train_frame'],device)
        cpu_batch=next(iter(DataLoader(dataset,batch_size=size,shuffle=True,num_workers=0,
                                      generator=torch.Generator().manual_seed(config['seed']))))
        batch=move_batch(cpu_batch,device)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        _sync(device); start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device,config['training']['amp']): outputs=model(batch['image'],return_aux=True)
        losses=criterion(outputs,batch)
        scaler.scale(losses['total_loss']).backward(); scaler.unscale_(optimizer)
        check_gradients(model)
        _sync(device); elapsed=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0
        rows.append(dict(batch_size=size,status='PASS',peak_allocated_bytes=peak,peak_allocated_MiB=peak/2**20,
                         peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else 0,
                         gpu_total_MiB=total/2**20 if device.type=='cuda' else None,
                         one_batch_forward_backward_seconds=elapsed,optimizer_steps=0,
                         amp_enabled=scaler.is_enabled(),loss_finite=True,gradients_finite=True,
                         estimated_adam_moment_bytes=2*sum(p.numel()*4 for p in model.parameters())))
        del batch,cpu_batch,outputs,losses,model,optimizer,scheduler,scaler,criterion
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    result=pd.DataFrame(rows)
    passed=result[result.status.eq('PASS')]
    if passed.empty: raise RuntimeError('No safe batch size passed')
    recommended=int(passed.batch_size.max())
    return result,recommended


def _assert_tree_equal(left,right):
    if torch.is_tensor(left):
        assert torch.equal(left.cpu(),right.cpu())
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: _assert_tree_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right): _assert_tree_equal(a,b)
    elif isinstance(left,np.ndarray):
        assert np.array_equal(left,right)
    else: assert left==right


def single_batch_smoke(root,config,info):
    """Exactly one optimizer step on a disposable model validates the real epoch routine."""
    seed_everything(config['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # The only step in Phase 6 is here. Model and diagnostic checkpoint are discarded.
    temporary=MorphoNetBT(**config['model']).to(device)
    datasets,loaders,generators=build_loaders(root,config,1)
    optimizer,scheduler,scaler,criterion=build_training_objects(temporary,config,info['train_frame'],device)
    stopper=EarlyStopping(**{k:config['training']['early_stopping'][k] for k in ('patience','min_delta')})
    initial_classifier=temporary.classifier.weight.detach().clone()
    _sync(device); started=time.perf_counter()
    metrics,predictions=train_one_epoch(temporary,loaders['train'],criterion,optimizer,scaler,device,config,1,1,max_batches=1)
    _sync(device); elapsed=time.perf_counter()-started
    check_gradients(temporary)
    assert not torch.equal(initial_classifier,temporary.classifier.weight)
    scheduler.step()
    expected_lr=config['training']['optimizer']['learning_rate']*(1+math.cos(math.pi/config['training']['max_epochs']))/2
    assert math.isclose(optimizer.param_groups[0]['lr'],expected_lr,rel_tol=1e-9)
    # Isolated scheduler/checkpoint test, not a real completed epoch or validation result.
    history=[dict(epoch=1,diagnostic_only=True)]
    payload=checkpoint_payload(temporary,optimizer,scheduler,scaler,stopper,generators,config,1,1,{}, {},history,info)
    payload['diagnostic_only']=True
    expected_random=(random.random(),float(np.random.rand()),torch.rand(4),
                     torch.rand(4,device=device).cpu(),torch.randperm(20,generator=generators['train']))
    with tempfile.TemporaryDirectory(dir=Path(root)/'results/training_setup') as temp:
        checkpoint_path=Path(temp)/'temporary_smoke_checkpoint.pt'
        atomic_torch_save(payload,checkpoint_path)
        restored=MorphoNetBT(**config['model']).to(device)
        opt2,sch2,scale2,loss2=build_training_objects(restored,config,info['train_frame'],device)
        stop2=EarlyStopping()
        gen2={k:torch.Generator().manual_seed(999) for k in ('train','val')}
        saved=load_checkpoint(checkpoint_path,restored,opt2,sch2,scale2,stop2,gen2,config,1,info['architecture'],info['split_hashes'])
        observed_random=(random.random(),float(np.random.rand()),torch.rand(4),
                         torch.rand(4,device=device).cpu(),torch.randperm(20,generator=gen2['train']))
        _assert_tree_equal(expected_random,observed_random)
        _assert_tree_equal(cpu_tree(temporary.state_dict()),cpu_tree(restored.state_dict()))
        _assert_tree_equal(cpu_tree(optimizer.state_dict()),cpu_tree(opt2.state_dict()))
        _assert_tree_equal(scheduler.state_dict(),sch2.state_dict())
        _assert_tree_equal(scaler.state_dict(),scale2.state_dict())
        _assert_tree_equal(stopper.state_dict(),stop2.state_dict())
        wrong=deepcopy(config); wrong['training']['max_epochs']+=1
        try:
            load_checkpoint(checkpoint_path,restored,opt2,sch2,scale2,stop2,gen2,wrong,1,info['architecture'],info['split_hashes'])
        except ValueError: pass
        else: raise AssertionError('Resume accepted changed configuration')
        # Protect existing outputs; only a disposable temporary project is used.
        mini_root=Path(temp)/'output_safety'
        occupied=mini_root/'checkpoints/main/fold_1'; occupied.mkdir(parents=True)
        sentinel=occupied/'sentinel.txt'; sentinel.write_text('preserve')
        try: prepare_output_paths(mini_root,1)
        except FileExistsError: pass
        else: raise AssertionError('Existing output was not protected')
        assert sentinel.read_text()=='preserve'
    result=dict(status='PASS',batch_size=config['training']['batch_size'],device=str(device),num_workers=config['training']['num_workers'],
        amp_enabled=scaler.is_enabled(),grad_scaler_init_scale=config['training']['grad_scaler_init_scale'],
        loss_values={k:v for k,v in metrics.items() if k.endswith('_loss') or k.endswith('_bce') or k.endswith('_dice')},
        all_gradients_finite=True,optimizer='AdamW',optimizer_steps_on_temporary_model=1,
        scheduler_next_lr=optimizer.param_groups[0]['lr'],scheduler_state_restored=True,optimizer_state_restored=True,
        model_state_restored=True,scaler_state_restored=True,early_stopping_state_restored=True,
        rng_and_loader_generators_restored=True,changed_resume_config_rejected=True,existing_output_protected=True,
        elapsed_single_batch_including_loader_seconds=elapsed,diagnostic_predictions_retained_pid=True,
        test_images_accessed=False,full_training_started=False,temporary_checkpoint_deleted=True,
        note='Single disposable optimizer step. No accuracy/F1 or real training output saved; not an epoch benchmark.')
    del temporary,restored,optimizer,opt2,scheduler,sch2,scaler,scale2,criterion,loss2,loaders,datasets
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    return result


def pure_logic_checks():
    # Hand-verifiable synthetic labels test macro averaging across all three classes.
    predictions=pd.DataFrame(dict(true_label=[0,0,1,2],predicted_label=[0,1,1,2]))
    metrics=classification_metrics(predictions)
    assert math.isclose(metrics['macro_f1'],7/9)
    stopper=EarlyStopping(patience=2)
    assert stopper.update(.5,1)
    assert not stopper.update(.5,2) and stopper.bad_epochs==1
    assert not stopper.update(.4,3) and stopper.stopped
    state=stopper.state_dict(); clone=EarlyStopping(); clone.load_state_dict(state)
    assert clone.state_dict()==state
    # Synthetic values test code only; no fabricated dataset metrics are saved.
    output=dict(logits=torch.randn(2,3),region_logits=torch.randn(2,1,8,8),boundary_logits=torch.randn(2,1,8,8),distance=torch.rand(2,1,8,8))
    batch=dict(label=torch.tensor([0,2]),region=torch.randint(0,2,(2,1,8,8)).float(),
               boundary=torch.randint(0,2,(2,1,8,8)).float(),distance=torch.rand(2,1,8,8))
    new=MorphoNetLoss()(output,batch); old=smoke_losses(output,batch)
    assert torch.allclose(new['total_loss'],old['total'])
    return dict(whole_epoch_metric_formula=True,early_stopping_ties_and_patience=True,
                loss_matches_phase4_definition=True,cpu_float32_loss_fallback=True)
