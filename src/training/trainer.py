"""Main-model-only training. Test images are never loaded by this module."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import uuid
from itertools import islice
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .losses import MorphoNetLoss, train_only_class_weights
from .scheduler import EarlyStopping, build_scheduler
from src.data.dataset import BrainTumorDataset, verify_frozen_splits
from src.data.kfold import load_manifest, reload_splits, audit_splits
from src.models.morphonet_bt import MorphoNetBT
from src.utils.complexity import fingerprint
from src.utils.checkpoint import seed_everything, capture_rng, cpu_tree, atomic_torch_save, load_checkpoint
from src.utils.training_metrics import classification_metrics, plot_training_curves

TRAINING_DEFAULTS = dict(max_epochs=100,batch_size=8,num_workers=2,persistent_workers=False,
    optimizer=dict(name='AdamW',learning_rate=.001,weight_decay=.0001),
    scheduler=dict(name='CosineAnnealingLR',eta_min=0.),
    early_stopping=dict(monitor='val_macro_f1',mode='max',patience=15,min_delta=0.),
    amp=True,grad_scaler_init_scale=1024.,max_grad_norm=None,
    loss=dict(lambda_region=1.,lambda_boundary=1.,lambda_distance=.5,lambda_morphology=.3),
    class_weights=None)


def timestamp():
    return datetime.now(timezone(timedelta(hours=3))).isoformat()


def validate_training_config(config):
    cfg=config['training']
    if cfg['optimizer']['name']!='AdamW' or cfg['scheduler']['name']!='CosineAnnealingLR':
        raise ValueError('Unexpected optimizer/scheduler')
    if cfg['max_epochs']<1 or cfg['batch_size']<1 or cfg['num_workers']<0 or cfg['persistent_workers']:
        raise ValueError('Invalid DataLoader or epoch configuration')
    stop=cfg['early_stopping']
    if stop['monitor']!='val_macro_f1' or stop['mode']!='max' or stop['patience']<1 or stop['min_delta']<0:
        raise ValueError('Checkpoint selection must use validation Macro-F1 only')
    if cfg['max_grad_norm'] is not None and cfg['max_grad_norm']<=0:
        raise ValueError('Gradient clipping must be positive or disabled')
    if cfg['class_weights'] not in (None,'balanced_train'):
        raise ValueError('Class weights must be null or explicitly balanced_train')


def preflight(root,config,fold):
    root=Path(root).resolve()
    if fold not in range(1,6): raise ValueError('Fold must be 1..5')
    validate_training_config(config)
    frozen=verify_frozen_splits(root)
    prior=json.loads((root/'results/architecture/frozen_input_hashes.json').read_text())
    for name,digest in prior['preprocessing_sources'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Preprocessing source changed: {name}')
    import yaml
    approved=yaml.safe_load((root/'results/architecture/preprocessing_v1_approved.yaml').read_text())
    if {k:v for k,v in config.items() if k not in ('model','training')}!=approved:
        raise ValueError('Preprocessing configuration mismatch')
    expected=json.loads((root/'results/complexity/architecture_fingerprint.json').read_text())
    # Keep the preflight construction from consuming the training RNG stream.
    with torch.random.fork_rng(devices=[]):
        model=MorphoNetBT(**config['model'])
    actual=fingerprint(model,config['model'],root)
    if actual!=expected: raise ValueError('Architecture v1.0 fingerprint mismatch')
    del model
    # Read test CSV metadata for required separation checks; no test Dataset or images.
    splits=reload_splits(root/'splits')
    frame=load_manifest(root/'results/dataset_statistics.csv')
    audit,coverage=audit_splits(splits,frame)
    return dict(architecture=expected['fingerprint_sha256'],split_hashes=frozen,
                leakage_audit=audit.to_dict('records'),coverage=coverage,
                train_frame=splits[fold]['train'],val_frame=splits[fold]['val'])


def build_loaders(root,config,fold):
    cfg=config['training']
    generators={part:torch.Generator().manual_seed(config['seed']) for part in ('train','val')}
    datasets={part:BrainTumorDataset(Path(root)/f'splits/fold_{fold}_{part}.csv',part,config,config['seed'])
              for part in ('train','val')}
    loaders={part:DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=part=='train',
                             num_workers=cfg['num_workers'],persistent_workers=False,drop_last=False,
                             pin_memory=torch.cuda.is_available(),generator=generators[part])
             for part,dataset in datasets.items()}
    return datasets,loaders,generators


def build_training_objects(model,config,train_frame,device):
    cfg=config['training']; opt=cfg['optimizer']
    optimizer=torch.optim.AdamW(model.parameters(),lr=opt['learning_rate'],weight_decay=opt['weight_decay'])
    scheduler=build_scheduler(optimizer,cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and cfg['amp'],
                                init_scale=cfg['grad_scaler_init_scale'])
    weights=train_only_class_weights(train_frame,cfg['class_weights'])
    criterion=MorphoNetLoss(**cfg['loss'],class_weights=weights).to(device)
    return optimizer,scheduler,scaler,criterion


def autocast_context(device,enabled):
    return torch.autocast(device_type='cuda',dtype=torch.float16) if device.type=='cuda' and enabled else nullcontext()


def move_batch(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def check_gradients(model):
    for name,parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f'Missing/nonfinite gradient: {name}')


def _epoch(model,loader,criterion,device,config,fold,epoch,optimizer=None,scaler=None,max_batches=None):
    training=optimizer is not None
    expected='train' if training else 'val'
    if loader.dataset.mode!=expected:
        raise ValueError(f'{expected} loop refuses any other dataset, including test')
    model.train(training)
    if training: loader.dataset.set_epoch(epoch-1)
    totals={}; rows=[]; samples=0
    context=nullcontext() if training else torch.inference_mode()
    with context:
        for cpu_batch in (loader if max_batches is None else islice(loader,max_batches)):
            batch=move_batch(cpu_batch,device)
            n=len(batch['label'])
            if training: optimizer.zero_grad(set_to_none=True)
            with autocast_context(device,config['training']['amp']):
                outputs=model(batch['image'],return_aux=True)
            losses=criterion(outputs,batch)
            if training:
                scaler.scale(losses['total_loss']).backward()
                scaler.unscale_(optimizer)
                check_gradients(model)
                clip=config['training']['max_grad_norm']
                if clip is not None: torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update()
            probability=outputs['logits'].detach().float().softmax(1).cpu().numpy()
            if not np.isfinite(probability).all(): raise FloatingPointError('Nonfinite probabilities')
            true=batch['label'].detach().cpu().tolist()
            for i in range(n):
                rows.append(dict(PID=cpu_batch['pid'][i],sample_id=cpu_batch['sample_id'][i],fold=fold,epoch=epoch,
                    split=expected,true_label=true[i],predicted_label=int(probability[i].argmax()),
                    prob_meningioma=float(probability[i,0]),prob_glioma=float(probability[i,1]),prob_pituitary=float(probability[i,2]),
                    model_name='MorphoNet-BT',experiment_name='main_v1'))
            for key,value in losses.items(): totals[key]=totals.get(key,0.)+float(value.detach().cpu())*n
            samples+=n
    if not samples: raise ValueError('Empty epoch')
    predictions=pd.DataFrame(rows)
    if not predictions.sample_id.is_unique: raise ValueError('Duplicate samples within epoch')
    if max_batches is None and set(predictions.sample_id)!=set(loader.dataset.frame.sample_id):
        raise ValueError('Epoch does not cover its entire split exactly once')
    metrics=classification_metrics(predictions)
    metrics.update({key:value/samples for key,value in totals.items()})
    metrics['num_samples']=samples
    metrics['diagnostic_partial_epoch']=max_batches is not None
    return metrics,predictions


def train_one_epoch(model,loader,criterion,optimizer,scaler,device,config,fold,epoch,max_batches=None):
    return _epoch(model,loader,criterion,device,config,fold,epoch,optimizer,scaler,max_batches)


def validate_one_epoch(model,loader,criterion,device,config,fold,epoch,max_batches=None):
    return _epoch(model,loader,criterion,device,config,fold,epoch,max_batches=max_batches)


def atomic_json(data,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temp,path)


def atomic_csv(data,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.tmp')
    data.to_csv(temp,index=False); os.replace(temp,path)


def output_paths(root,fold):
    root=Path(root).resolve()
    return dict(checkpoints=root/f'checkpoints/main/fold_{fold}',results=root/f'results/main/fold_{fold}',
                log=root/f'logs/main/fold_{fold}_training.csv')


def prepare_output_paths(root,fold,resume=False,overwrite=False):
    root=Path(root).resolve(); paths=output_paths(root,fold)
    if resume and overwrite: raise ValueError('Choose resume or overwrite, not both')
    occupied=[p for p in paths.values() if p.exists() and (p.is_file() or any(p.iterdir()))]
    if occupied and not (resume or overwrite):
        raise FileExistsError('Existing run found. Explicit resume=True or overwrite=True is required')
    if resume:
        if not (paths['checkpoints']/'last_checkpoint.pt').is_file(): raise FileNotFoundError('No last checkpoint to resume')
        if (paths['results']/'.training.lock').exists(): raise RuntimeError('Run lock exists; confirm no training process remains before removing it')
    if overwrite and occupied:
        if (paths['results']/'.training.lock').exists(): raise RuntimeError('Cannot archive an active/locked run')
        # Preserve prior outputs by archiving, never deleting them. Validate every move target.
        archive=root/'results/run_archives'/f'fold_{fold}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
        archive.mkdir(parents=True,exist_ok=False)
        for key,source in paths.items():
            if not source.exists(): continue
            destination=archive/key
            if not source.resolve().is_relative_to(root) or not destination.resolve().is_relative_to(root):
                raise ValueError('Archive path escaped project root')
            if (source/'.training.lock').exists(): raise RuntimeError('Cannot archive an active/locked run')
            shutil.move(str(source),str(destination))
    for key,path in paths.items(): (path.parent if key=='log' else path).mkdir(parents=True,exist_ok=True)
    return paths


def checkpoint_payload(model,optimizer,scheduler,scaler,stopper,generators,config,fold,epoch,validation,training,history,info):
    return dict(model_state_dict=cpu_tree(model.state_dict()),optimizer_state_dict=cpu_tree(optimizer.state_dict()),
        scheduler_state_dict=deepcopy(scheduler.state_dict()),scaler_state_dict=deepcopy(scaler.state_dict()),
        epoch=epoch,validation_macro_f1=validation.get('macro_f1'),validation_metrics=deepcopy(validation),
        training_metrics=deepcopy(training),configuration=deepcopy(config),seed=config['seed'],fold=fold,
        architecture_fingerprint=info['architecture'],preprocessing_version='MorphoNet-BT Preprocessing v1.0',
        split_hashes=info['split_hashes'],early_stopping_state=stopper.state_dict(),history=deepcopy(history),
        best_metric=stopper.best_metric,best_epoch=stopper.best_epoch,rng_state=capture_rng(generators),
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=str(next(model.parameters()).device)),
        saved_at=timestamp())


def _write_best(best,predictions,paths):
    if best is None: return
    atomic_torch_save(best,paths['checkpoints']/'best_macro_f1.pt')
    atomic_json(best['validation_metrics'],paths['results']/'best_validation_metrics.json')
    atomic_csv(pd.DataFrame(predictions),paths['results']/'best_validation_predictions.csv')


def train_fold(root,config,fold,*,confirm_full_training=False,resume=False,overwrite=False):
    if confirm_full_training is not True:
        raise PermissionError('Manual full training requires confirm_full_training=True')
    root=Path(root).resolve(); config=deepcopy(config)
    info=preflight(root,config,fold)
    paths=prepare_output_paths(root,fold,resume,overwrite)
    complete=paths['results']/'training_complete.json'
    if resume and complete.exists():
        return json.loads(complete.read_text(encoding='utf-8'))
    lock=paths['results']/'.training.lock'
    with lock.open('x',encoding='utf-8') as handle: json.dump(dict(pid=os.getpid(),started=timestamp()),handle)
    try:
        seed_everything(config['seed'])
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model=MorphoNetBT(**config['model']).to(device)
        datasets,loaders,generators=build_loaders(root,config,fold)
        optimizer,scheduler,scaler,criterion=build_training_objects(model,config,info['train_frame'],device)
        stop_cfg=config['training']['early_stopping']
        stopper=EarlyStopping(stop_cfg['patience'],stop_cfg['min_delta'])
        history=[]; best=None; best_predictions=[]; start_epoch=1
        if resume:
            saved=load_checkpoint(paths['checkpoints']/'last_checkpoint.pt',model,optimizer,scheduler,scaler,stopper,
                                  generators,config,fold,info['architecture'],info['split_hashes'])
            current_runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=str(next(model.parameters()).device))
            if saved['runtime']!=current_runtime: raise ValueError('Runtime changed; exact resume requires review')
            history=saved['history']; start_epoch=saved['epoch']+1
            best=saved['best_checkpoint']; best_predictions=saved['best_predictions']
            atomic_csv(pd.DataFrame(history),paths['log'])
            _write_best(best,best_predictions,paths)
        metadata=dict(started=timestamp(),fold=fold,seed=config['seed'],configuration=config,
            architecture_fingerprint=info['architecture'],preprocessing_version='MorphoNet-BT Preprocessing v1.0',
            split_files={part:f'splits/fold_{fold}_{part}.csv' for part in ('train','val','test')},
            split_hashes=info['split_hashes'],torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
            test_access='CSV metadata only for PID checks; no test images/metrics',
            checkpoint_selection='slice-level validation Macro-F1',resumed=resume,
            deterministic='seed 42; cuDNN deterministic=True benchmark=False; TF32=False; bitwise GPU determinism not guaranteed')
        atomic_json(metadata,paths['results']/('resume_metadata.json' if resume else 'run_metadata.json'))
        max_epochs=config['training']['max_epochs']
        for epoch in range(start_epoch,max_epochs+1):
            if stopper.stopped: break
            started=time.perf_counter(); lr=optimizer.param_groups[0]['lr']
            train_metrics,train_predictions=train_one_epoch(model,loaders['train'],criterion,optimizer,scaler,device,config,fold,epoch)
            val_metrics,val_predictions=validate_one_epoch(model,loaders['val'],criterion,device,config,fold,epoch)
            is_best=stopper.update(val_metrics['macro_f1'],epoch)
            scheduler.step()  # Exactly once after this completed epoch; checkpoint stores next-epoch LR.
            row=dict(fold=fold,epoch=epoch,learning_rate=lr,next_learning_rate=optimizer.param_groups[0]['lr'],
                     epoch_time_seconds=time.perf_counter()-started,amp_scale=scaler.get_scale(),
                     early_stopping_bad_epochs=stopper.bad_epochs)
            for prefix,metrics in [('train',train_metrics),('val',val_metrics)]:
                row.update({f'{prefix}_{k}':v for k,v in metrics.items() if isinstance(v,(float,int)) and k!='diagnostic_partial_epoch'})
            history.append(row)
            atomic_csv(train_predictions,paths['results']/f'predictions/train_epoch_{epoch:03d}.csv')
            atomic_csv(val_predictions,paths['results']/f'predictions/val_epoch_{epoch:03d}.csv')
            atomic_json(dict(train=train_metrics,validation=val_metrics),paths['results']/f'metrics/epoch_{epoch:03d}.json')
            payload=checkpoint_payload(model,optimizer,scheduler,scaler,stopper,generators,config,fold,epoch,
                                       val_metrics,train_metrics,history,info)
            if is_best:
                best=deepcopy(payload); best_predictions=val_predictions.to_dict('records')
            # Last is authoritative; embeds best snapshot to repair interrupted secondary writes on resume.
            atomic_torch_save({**payload,'best_checkpoint':best,'best_predictions':best_predictions},paths['checkpoints']/'last_checkpoint.pt')
            if is_best: _write_best(best,best_predictions,paths)
            atomic_csv(pd.DataFrame(history),paths['log'])
            print(f"Fold {fold} | epoch {epoch}/{max_epochs} | LR={lr:.6g}\n"
                  f"Training: loss={train_metrics['total_loss']:.4f}, accuracy={train_metrics['accuracy']:.4f}, Macro-F1={train_metrics['macro_f1']:.4f}\n"
                  f"Validation: loss={val_metrics['total_loss']:.4f}, accuracy={val_metrics['accuracy']:.4f}, Macro-F1={val_metrics['macro_f1']:.4f}\n"
                  f"Best Validation Macro-F1={stopper.best_metric:.4f} | patience {stopper.bad_epochs}/{stopper.patience}")
        if not history or best is None: raise RuntimeError('No completed epoch; no completion marker created')
        plot_training_curves(paths['log'],paths['results']/'training_curves.png')
        marker=dict(best_epoch=stopper.best_epoch,best_validation_macro_f1=stopper.best_metric,
                    total_epochs=len(history),early_stopping=stopper.stopped,completed_at=timestamp(),
                    checkpoint_path=str(paths['checkpoints']/'best_macro_f1.pt'),test_evaluated=False)
        atomic_json(marker,complete)
        return marker
    finally:
        if lock.exists(): lock.unlink()


def train_folds(root,config,folds=(1,),*,confirm_full_training=False,resume=False,overwrite=False):
    if confirm_full_training is not True:
        raise PermissionError('The researcher must explicitly start full training')
    if not folds or len(set(folds))!=len(folds): raise ValueError('Select unique fold numbers')
    results=[]
    for fold in folds:
        results.append(train_fold(root,config,fold,confirm_full_training=True,resume=resume,overwrite=overwrite))
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results
