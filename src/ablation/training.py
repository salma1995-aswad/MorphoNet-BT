"""Manual-only full epochs on TRAIN/VAL; never invoked during preparation."""
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import gc
import json
import math
import os
import shutil
import time
import uuid
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.models.ablation import build_ablation, AblationLoss, VARIANTS
from src.training.scheduler import EarlyStopping
from src.utils.checkpoint import seed_everything,capture_rng,restore_rng,cpu_tree,atomic_torch_save
from src.utils.training_metrics import classification_metrics
from .data import AblationDataset
from .protocol import (TRAIN_VARIANTS,CLASS_MAPPING,VERSION,require,file_hash,object_hash,preflight,
    run_identity,validate_selection,require_prepared,check_protection)

_PERMIT=object()

def timestamp():return datetime.now(timezone.utc).isoformat()

def atomic_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)

def atomic_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.tmp');frame.to_csv(temp,index=False,float_format='%.17g');os.replace(temp,path)

def run_path(root,variant,fold):
    validate_selection([variant],[fold]);root=Path(root).resolve()
    path=root/f'results/ablation/training/{variant}/fold_{fold}'
    require(path.resolve().is_relative_to(root/'results/ablation/training'),'Output path escapes training directory')
    return path

def build_loaders(info,variant,fold):
    validate_selection([variant],[fold]);require(info['test_lock'],'Test lock inactive')
    generators={part:torch.Generator().manual_seed(42) for part in ('train','val')}
    datasets={part:AblationDataset(info['root']/f'splits/fold_{fold}_{part}.csv',part,VARIANTS[variant]['use_morphology_supervision'],info['preprocessing']) for part in ('train','val')}
    loaders={part:DataLoader(data,batch_size=32,shuffle=part=='train',num_workers=2,persistent_workers=False,drop_last=False,
        pin_memory=True,generator=generators[part]) for part,data in datasets.items()}
    return datasets,loaders,generators

def training_objects(model,variant,device):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=1024.)
    return optimizer,scheduler,scaler,AblationLoss(variant).to(device)

def forward_loss(model,batch,criterion,variant,device):
    morphology=VARIANTS[variant]['use_morphology_supervision']
    expected={'image','label','pid','sample_id'}|({'region','boundary','distance'} if morphology else set())
    require(set(batch)==expected,'Batch contains unexpected/missing targets')
    moved={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
    image=moved['image'];require(image.dtype==torch.float32 and tuple(image.shape[1:])==(1,256,256),'Invalid MRI input')
    require(bool(torch.isfinite(image).all()),'Nonfinite MRI')
    with torch.autocast('cuda',dtype=torch.float16):outputs=model(image,return_aux=morphology)
    logits=outputs['logits'] if isinstance(outputs,dict) else outputs
    require(tuple(logits.shape)==(len(image),3) and bool(torch.isfinite(logits).all()),'Nonfinite/invalid logits')
    losses=criterion(outputs,moved if morphology else {'label':moved['label']})
    if not all(bool(torch.isfinite(v).all()) for v in losses.values()):raise FloatingPointError('Nonfinite loss; run stopped without protocol changes')
    return logits,losses

def _epoch(model,loader,criterion,variant,fold,epoch,device,optimizer=None,scaler=None,*,permit=None):
    require(permit is _PERMIT,'Epoch execution requires the manual launcher')
    training=optimizer is not None;part='train' if training else 'val'
    require(loader.dataset.mode==part,'Test data are locked; wrong epoch partition')
    model.train(training)
    if training:loader.dataset.set_epoch(epoch-1) # Same zero-based augmentation schedule as frozen Phase 6.
    totals={};rows=[];samples=0
    with nullcontext() if training else torch.inference_mode():
        for batch in loader:
            if training:optimizer.zero_grad(set_to_none=True)
            logits,losses=forward_loss(model,batch,criterion,variant,device)
            if training:
                scaler.scale(losses['total_loss']).backward();scaler.unscale_(optimizer)
                for name,p in model.named_parameters():
                    if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all())):
                        raise FloatingPointError(f'Missing/nonfinite gradient: {name}; stopping without clipping or hyperparameter changes')
                scaler.step(optimizer);scaler.update()
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite model state after optimizer update')
            probability=logits.detach().float().softmax(1).cpu().numpy();labels=batch['label'].tolist()
            for i,label in enumerate(labels):
                rows.append(dict(variant=variant,fold=fold,epoch=epoch,split=part,PID=batch['pid'][i],sample_id=batch['sample_id'][i],
                    true_label=label,predicted_label=int(probability[i].argmax()),prob_meningioma=float(probability[i,0]),prob_glioma=float(probability[i,1]),prob_pituitary=float(probability[i,2])))
            for key,value in losses.items():totals[key]=totals.get(key,0.)+float(value.detach())*len(labels)
            samples+=len(labels)
    predictions=pd.DataFrame(rows)
    require(samples==len(loader.dataset) and predictions.sample_id.is_unique and set(predictions.sample_id)==set(loader.dataset.frame.sample_id),'Incomplete/duplicate epoch coverage')
    metrics=classification_metrics(predictions);metrics.update({k:v/samples for k,v in totals.items()});metrics['num_samples']=samples
    return metrics,predictions

CHECKPOINT_KEYS={'variant','fold','epoch','best_epoch','best_validation_macro_f1','model_state_dict','optimizer_state_dict','scheduler_state_dict',
 'scaler_state_dict','early_stopping_state','early_stopping_counter','training_configuration','architecture_configuration','loss_configuration',
 'seed','class_mapping','preprocessing_version','preprocessing','augmentation_configuration','split_identity','split_fingerprint',
 'training_configuration_fingerprint','identity','rng_state','history','training_seconds','active_wall_seconds','training_metrics','validation_metrics'}

def make_checkpoint(model,optimizer,scheduler,scaler,stopper,generators,info,identity,epoch,history,train,val,seconds,wall):
    variant=identity['variant'];morphology=VARIANTS[variant]['use_morphology_supervision']
    return dict(variant=variant,fold=identity['fold'],epoch=epoch,best_epoch=stopper.best_epoch,best_validation_macro_f1=stopper.best_metric,
        model_state_dict=cpu_tree(model.state_dict()),optimizer_state_dict=cpu_tree(optimizer.state_dict()),scheduler_state_dict=deepcopy(scheduler.state_dict()),
        scaler_state_dict=deepcopy(scaler.state_dict()),early_stopping_state=stopper.state_dict(),early_stopping_counter=stopper.bad_epochs,
        training_configuration=deepcopy(info['config']['training']),architecture_configuration={**info['config']['model'],**VARIANTS[variant]},
        loss_configuration=deepcopy(info['config']['training']['loss']) if morphology else {'name':'CrossEntropyLoss','class_weights':None},
        seed=42,class_mapping=CLASS_MAPPING,preprocessing_version=VERSION,preprocessing=deepcopy(info['preprocessing']),augmentation_configuration=deepcopy(info['preprocessing']['augmentation']),
        split_identity=deepcopy(info['split_hashes']),split_fingerprint=object_hash(info['split_hashes']),training_configuration_fingerprint=info['configuration_fingerprint'],
        identity=deepcopy(identity),rng_state=capture_rng(generators),history=deepcopy(history),training_seconds=seconds,active_wall_seconds=wall,
        training_metrics=deepcopy(train),validation_metrics=deepcopy(val),saved_at=timestamp())

def validate_checkpoint(saved,info,identity,allow_diagnostic=False):
    require(CHECKPOINT_KEYS<=set(saved),'Missing checkpoint state keys')
    require(allow_diagnostic or not saved.get('diagnostic_only',False),'Diagnostic state cannot resume an experiment')
    require(saved['identity']==identity and saved['variant']==identity['variant'] and saved['fold']==identity['fold'],'Resume identity/source/runtime mismatch')
    require(saved['training_configuration']==info['config']['training'],'Resume training configuration mismatch')
    require(saved['architecture_configuration']=={**info['config']['model'],**VARIANTS[identity['variant']]},'Resume architecture mismatch')
    expected_loss=info['config']['training']['loss'] if VARIANTS[identity['variant']]['use_morphology_supervision'] else {'name':'CrossEntropyLoss','class_weights':None}
    require(saved['loss_configuration']==expected_loss,'Resume loss mismatch')
    require(saved['preprocessing']==info['preprocessing'] and saved['augmentation_configuration']==info['preprocessing']['augmentation'],'Resume preprocessing mismatch')
    require(saved['seed']==42 and saved['class_mapping']==CLASS_MAPPING and saved['preprocessing_version']==VERSION,'Resume labels/seed mismatch')
    require(saved['split_identity']==info['split_hashes'] and saved['split_fingerprint']==object_hash(info['split_hashes']),'Resume split mismatch')
    require(saved['training_configuration_fingerprint']==info['configuration_fingerprint'],'Resume contract mismatch')
    epoch=saved['epoch'];state=saved['early_stopping_state']
    require(type(epoch) is int and 1<=epoch<=100 and len(saved['history'])==epoch,'Resume epoch mismatch')
    require([r['epoch'] for r in saved['history']]==list(range(1,epoch+1)),'Noncontiguous history')
    require(saved['scheduler_state_dict']['last_epoch']==epoch and saved['scheduler_state_dict']['T_max']==100 and saved['scheduler_state_dict']['eta_min']==0.,'Scheduler epoch/config mismatch')
    expected_lr=.001*.5*(1+math.cos(math.pi*epoch/100))
    require(all(math.isclose(g['lr'],expected_lr,rel_tol=1e-12,abs_tol=1e-15) and g['weight_decay']==.0001 for g in saved['optimizer_state_dict']['param_groups']),'Saved optimizer LR/weight decay mismatch')
    require(state['best_epoch']==saved['best_epoch'] and state['best_metric']==saved['best_validation_macro_f1'] and state['bad_epochs']==saved['early_stopping_counter'],'Early stopping state mismatch')
    require(1<=saved['best_epoch']<=epoch and math.isfinite(saved['best_validation_macro_f1']),'Invalid best selection state')
    require(state['patience']==15 and state['min_delta']==0.,'Patience configuration mismatch')
    require({'python','numpy','torch','cuda','loaders'}<=saved['rng_state'].keys(),'Missing RNG state')
    require(set(saved['rng_state']['loaders'])=={'train','val'},'Missing loader generator states')

def restore_checkpoint(saved,model,optimizer,scheduler,scaler,stopper,generators,info,identity,*,allow_diagnostic=False):
    validate_checkpoint(saved,info,identity,allow_diagnostic)
    model.load_state_dict(saved['model_state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer_state_dict'])
    scheduler.load_state_dict(saved['scheduler_state_dict']);scaler.load_state_dict(saved['scaler_state_dict']);stopper.load_state_dict(saved['early_stopping_state'])
    require(optimizer.param_groups[0]['lr']==scheduler.get_last_lr()[0],'Restored optimizer/scheduler LR mismatch')
    restore_rng(saved['rng_state'],generators)
    return saved['epoch']+1

def inspect_run(root,variant,fold,info,*,resume=False,overwrite=False):
    require(type(resume) is bool and type(overwrite) is bool and not(resume and overwrite),'Choose resume or overwrite, not both')
    path=run_path(root,variant,fold);identity=run_identity(info,variant,fold)
    if (path/'.training.lock').exists():raise RuntimeError('Run lock exists; verify the previous process stopped before removing a stale lock')
    marker=path/'training_complete.json'
    if marker.exists():
        done=json.loads(marker.read_text());require(done['identity']==identity,'Completed run identity mismatch')
        for name,digest in done['checkpoint_sha256'].items():require(file_hash(path/name)==digest,'Completed checkpoint changed')
        if not overwrite:return dict(action='skip',path=path,completed=done)
    occupied=path.exists() and any(path.iterdir())
    if resume:
        require((path/'last_checkpoint.pt').is_file(),'No last checkpoint available for resume')
        saved=torch.load(path/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        validate_checkpoint(saved,info,identity)
        require(saved.get('best_checkpoint') is not None,'Last checkpoint lacks committed best snapshot')
        validate_checkpoint(saved['best_checkpoint'],info,identity)
        require(saved['best_checkpoint']['epoch']==saved['best_epoch'],'Embedded best epoch mismatch')
        return dict(action='resume',path=path,saved=saved)
    if occupied and not overwrite:raise FileExistsError('Existing incomplete run preserved. Set RESUME=True to resume its committed last epoch.')
    return dict(action='archive_then_new' if occupied else 'new',path=path)

def save_best(best,predictions,path):
    atomic_torch_save(best,path/'best_macro_f1.pt')
    atomic_json(dict(epoch=best['epoch'],**best['validation_metrics']),path/'best_validation_summary.json')
    atomic_csv(pd.DataFrame(predictions),path/'best_validation_predictions.csv')

def plot_curves(history,path,morphology):
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(history);require(not frame.empty,'No experimental curves without completed epochs')
    plots=[('total_loss','Total loss'),('macro_f1','Macro F1'),('learning_rate','Learning rate')]
    if morphology:plots += [('region_loss','Region loss'),('boundary_loss','Boundary loss'),('distance_loss','Distance loss'),('weighted_morphology_loss','Weighted morphology contribution')]
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        for key,label in plots:
            fig,ax=plt.subplots(figsize=(5.5,3.5),layout='constrained')
            if key=='learning_rate':ax.plot(frame.epoch,frame.learning_rate,label='LR used')
            else:
                for part,name in [('train','Training'),('val','Validation')]:ax.plot(frame.epoch,frame[f'{part}_{key}'],label=name)
            ax.set(xlabel='Epoch',ylabel=label);ax.legend(frameon=False)
            for ext in ['png','pdf']:fig.savefig(path/f'{key}.{ext}',dpi=300,bbox_inches='tight')
            plt.close(fig)

def verify_completed(path):
    path=Path(path);marker=json.loads((path/'training_complete.json').read_text())
    for name in ['best_macro_f1.pt','last_checkpoint.pt','history.csv','run_metadata.json']:require((path/name).is_file(),f'Missing completed output: {name}')
    frame=pd.read_csv(path/'history.csv',float_precision='round_trip')
    require(len(frame)==marker['total_epochs'],'History count mismatch')
    numeric=frame.select_dtypes('number');require(np.isfinite(numeric.to_numpy()).all(),'Nonfinite epoch history')
    expected=[.001*.5*(1+math.cos(math.pi*(e-1)/100)) for e in frame.epoch]
    require(np.allclose(frame.learning_rate,expected,rtol=1e-12,atol=1e-15),'LR schedule mismatch')
    best=frame.loc[frame.val_macro_f1.idxmax()]
    require(int(best.epoch)==marker['best_epoch'] and best.val_macro_f1==marker['best_validation_macro_f1'],'Best selection mismatch')
    for name,digest in marker['checkpoint_sha256'].items():require(file_hash(path/name)==digest,'Checkpoint hash mismatch')
    return marker

def completion_summary(root,info,*,write=False):
    rows=[]
    for variant in TRAIN_VARIANTS:
        for fold in range(1,6):
            path=run_path(root,variant,fold)
            if not(path/'training_complete.json').exists():continue
            marker=verify_completed(path);require(marker['identity']==run_identity(info,variant,fold),'Summary identity mismatch')
            rows.append({'Variant':variant,'Fold':fold,'Best Epoch':marker['best_epoch'],'Best Validation Accuracy':marker['best_validation_accuracy'],
                'Best Validation Macro-F1':marker['best_validation_macro_f1'],'Total Epochs':marker['total_epochs'],'Early Stopping':marker['early_stopping'],
                'Training Time Seconds':marker['training_seconds'],'Checkpoint Path':marker['checkpoint_path'],'Active Run Wall Seconds':marker['active_wall_seconds']})
    table=pd.DataFrame(rows)
    if write and rows:atomic_csv(table,Path(root)/'results/ablation/training/ablation_training_summary.csv')
    return table

def _train_run(info,variant,fold,*,resume,overwrite,permit=None):
    require(permit is _PERMIT,'Real training requires manual confirmation')
    plan=inspect_run(info['root'],variant,fold,info,resume=resume,overwrite=overwrite);path=plan['path'];identity=run_identity(info,variant,fold)
    if plan['action']=='skip':print(f'{variant} fold {fold}: completed run preserved; skipped.');return plan['completed']
    seed_everything(42);device=torch.device('cuda')
    model=build_ablation(variant,info['config']['model']).to(device)
    datasets,loaders,generators=build_loaders(info,variant,fold)
    optimizer,scheduler,scaler,criterion=training_objects(model,variant,device);stopper=EarlyStopping(patience=15,min_delta=0.)
    history=[];best=None;best_predictions=[];seconds=0.;prior_wall=0.;start_epoch=1
    if resume:
        saved=plan.pop('saved');start_epoch=restore_checkpoint(saved,model,optimizer,scheduler,scaler,stopper,generators,info,identity)
        history=saved['history'];best=saved['best_checkpoint'];best_predictions=saved['best_predictions'];seconds=saved['training_seconds'];prior_wall=saved['active_wall_seconds'];del saved
    if plan['action']=='archive_then_new':
        archive=info['root']/f'results/ablation/run_archives/{variant}_fold_{fold}_{uuid.uuid4().hex}'
        require(path.resolve().is_relative_to(info['root']/'results/ablation/training') and archive.resolve().is_relative_to(info['root']/'results/ablation/run_archives'),'Unsafe archive target')
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(path),str(archive))
    path.mkdir(parents=True,exist_ok=True);lock=path/'.training.lock'
    with lock.open('x') as handle:json.dump(dict(pid=os.getpid(),started_at=timestamp()),handle)
    session_started=time.perf_counter()
    try:
        metadata=dict(identity=identity,training_configuration=info['config']['training'],architecture_configuration={**info['config']['model'],**VARIANTS[variant]},
            test_lock=True,test_access='CSV identity metadata and opaque protection hashes only; no test images or performance',
            configuration_fingerprint=info['configuration_fingerprint'],preprocessing=info['preprocessing'],seed=42,class_mapping=CLASS_MAPPING,
            source='Explicit manual notebook launch',started_at=timestamp(),resumed=resume)
        atomic_json(metadata,path/(f'resume_{uuid.uuid4().hex}.json' if resume else 'run_metadata.json'))
        if resume:
            save_best(best,best_predictions,path);atomic_csv(pd.DataFrame(history),path/'history.csv')
        for epoch in range(start_epoch,101):
            if stopper.stopped:break
            started=time.perf_counter();lr=optimizer.param_groups[0]['lr']
            train,_=_epoch(model,loaders['train'],criterion,variant,fold,epoch,device,optimizer,scaler,permit=permit)
            val,predictions=_epoch(model,loaders['val'],criterion,variant,fold,epoch,device,permit=permit)
            improved=stopper.update(val['macro_f1'],epoch);scheduler.step()
            row=dict(epoch=epoch,learning_rate=lr,next_learning_rate=optimizer.param_groups[0]['lr'],epoch_duration_seconds=time.perf_counter()-started,amp_scale=scaler.get_scale())
            for part,values in [('train',train),('val',val)]:
                row.update({f'{part}_{k}':v for k,v in values.items() if isinstance(v,(int,float))})
            history.append(row);seconds+=row['epoch_duration_seconds']
            payload=make_checkpoint(model,optimizer,scheduler,scaler,stopper,generators,info,identity,epoch,history,train,val,seconds,prior_wall+time.perf_counter()-session_started)
            if improved:best=deepcopy(payload);best_predictions=predictions.to_dict('records')
            atomic_torch_save({**payload,'best_checkpoint':best,'best_predictions':best_predictions},path/'last_checkpoint.pt')
            if improved:save_best(best,best_predictions,path)
            atomic_csv(pd.DataFrame(history),path/'history.csv')
            atomic_json(dict(train=train,validation=val),path/f'epoch_metrics/epoch_{epoch:03d}.json')
            print(f'{variant} | fold {fold} | epoch {epoch}/100 | LR {lr:.6g} | train loss {train["total_loss"]:.4f}, F1 {train["macro_f1"]:.4f} | val loss {val["total_loss"]:.4f}, F1 {val["macro_f1"]:.4f} | patience {stopper.bad_epochs}/15',flush=True)
            del payload
        require(history and best is not None,'No complete epoch to finalize')
        plot_curves(history,path,VARIANTS[variant]['use_morphology_supervision'])
        check_protection(info['root'])
        marker=dict(identity=identity,variant=variant,fold=fold,best_epoch=best['epoch'],best_validation_accuracy=best['validation_metrics']['accuracy'],
            best_validation_macro_f1=stopper.best_metric,total_epochs=len(history),early_stopping=stopper.stopped,training_seconds=seconds,
            active_wall_seconds=prior_wall+time.perf_counter()-session_started,checkpoint_path=str(path/'best_macro_f1.pt'),
            checkpoint_sha256={name:file_hash(path/name) for name in ['best_macro_f1.pt','last_checkpoint.pt']},completed_at=timestamp(),test_evaluated=False)
        atomic_json(marker,path/'training_complete.json');verify_completed(path)
        return marker
    except Exception as exc:
        atomic_json(dict(variant=variant,fold=fold,error_type=type(exc).__name__,message=str(exc),failed_at=timestamp(),
            recovery='Run stopped. No hyperparameters changed. Resume only from the last fully committed epoch after reviewing the cause.'),path/f'failure_{uuid.uuid4().hex}.json')
        raise
    finally:
        lock.unlink(missing_ok=True)

def train_selected(root,variants=('A0',),folds=(1,),*,confirm_full_training=False,resume=False,overwrite=False):
    # This guard returns before filesystem reads, preflight, directory creation or summaries.
    if 'A5' in variants:raise ValueError('A5 is frozen and must not be retrained')
    if confirm_full_training is not True:
        print('Training is disabled. Set CONFIRM_FULL_TRAINING = True manually to start the selected experiment.')
        return []
    validate_selection(variants,folds);require(not(resume and overwrite),'Resume and overwrite cannot both be enabled')
    info=preflight(root);require_prepared(info);require(torch.cuda.is_available(),'Frozen protocol requires CUDA AMP; CPU fallback is disabled')
    # Inspect every selected output/resume identity before any epoch begins.
    for variant in sorted(variants):
        for fold in sorted(folds):inspect_run(root,variant,fold,info,resume=resume,overwrite=overwrite)
    results=[]
    for variant in sorted(variants):
        for fold in sorted(folds):
            results.append(_train_run(info,variant,fold,resume=resume,overwrite=overwrite,permit=_PERMIT))
            completion_summary(root,info,write=True);gc.collect();torch.cuda.empty_cache()
    print('Selected runs finished. Control returned to the user; no unselected run will start.')
    return results
