"""Non-training preparation: schema/state tests only, no epochs or optimizer steps."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
import ast
import json
import math
import random
import numpy as np
import pandas as pd
import torch
from src.models.ablation import build_ablation,AblationLoss,VARIANTS
from src.training.losses import MorphoNetLoss
from src.training.scheduler import EarlyStopping
from src.utils.checkpoint import capture_rng
from . import training as tr
from .protocol import preflight,run_identity,protected_inventory,check_protection,require,file_hash
from .data import AblationDataset


def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def state_roundtrip(info):
    """Serialize explicit artificial state in RAM; no data, backward or step calls."""
    generators={p:torch.Generator().manual_seed(42+i) for i,p in enumerate(['train','val'])}
    original_rng=capture_rng(generators)
    try:
        model=torch.nn.Linear(2,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=0.)
        scaler=torch.amp.GradScaler('cuda',enabled=torch.cuda.is_available(),init_scale=1024.)
        # Populate artificial moments directly. No optimizer update is performed.
        for p in model.parameters():optimizer.state[p]={'step':torch.tensor(7.),'exp_avg':torch.full_like(p,.125),'exp_avg_sq':torch.full_like(p,.25)}
        lr=.001*.5*(1+math.cos(math.pi*7/100));optimizer.param_groups[0]['lr']=lr
        state=scheduler.state_dict();state.update(last_epoch=7,_step_count=8,_last_lr=[lr]);scheduler.load_state_dict(state)
        stopper=EarlyStopping();stopper.update(.6,3);stopper.update(.5,7)
        identity=run_identity(info,'A0',1)
        history=[{'epoch':i} for i in range(1,8)]
        saved=tr.make_checkpoint(model,optimizer,scheduler,scaler,stopper,generators,info,identity,7,history,{}, {'macro_f1':.5},12.,15.)
        saved['diagnostic_only']=True
        stream=BytesIO();torch.save(saved,stream);stream.seek(0);loaded=torch.load(stream,map_location='cpu',weights_only=False)
        require(same(saved,loaded),'In-memory serialization mismatch')
        restored=torch.nn.Linear(2,2)
        opt2=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.0001)
        sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=100,eta_min=0.)
        scaler2=torch.amp.GradScaler('cuda',enabled=torch.cuda.is_available(),init_scale=1024.)
        stop2=EarlyStopping();gens2={p:torch.Generator().manual_seed(99) for p in ['train','val']}
        next_epoch=tr.restore_checkpoint(loaded,restored,opt2,sched2,scaler2,stop2,gens2,info,identity,allow_diagnostic=True)
        require(next_epoch==8 and stop2.best_epoch==3 and stop2.bad_epochs==1,'Resume resets epoch or patience')
        for a,b in [(model.state_dict(),restored.state_dict()),(optimizer.state_dict(),opt2.state_dict()),(scheduler.state_dict(),sched2.state_dict()),
                    (scaler.state_dict(),scaler2.state_dict()),(stopper.state_dict(),stop2.state_dict()),(saved['rng_state'],capture_rng(gens2))]:
            require(same(a,b),'Restored state mismatch')
        # Compare the next random draws on the original and restored streams.
        from src.utils.checkpoint import restore_rng
        def draws(gs):return (random.random(),np.random.random(),torch.rand(3),{k:torch.rand(3,generator=g) for k,g in gs.items()})
        expected=draws(gens2);restore_rng(saved['rng_state'],generators);actual=draws(generators)
        require(same(expected,actual),'RNG continuation mismatch')
        rejected=0
        cases=[deepcopy(saved),deepcopy(saved),deepcopy(saved)]
        cases[0].pop('rng_state');cases[1]['identity']['fold']=2;cases[1]['diagnostic_only']=False
        for case in cases:
            try:tr.validate_checkpoint(case,info,identity)
            except ValueError:rejected+=1
        require(rejected==3,'Incomplete, mismatched or diagnostic state accepted')
        return dict(serialization='BytesIO only; nothing saved as a checkpoint',required_keys=sorted(tr.CHECKPOINT_KEYS),
            optimizer_moments_restored=True,scheduler_restored=True,scaler_restored=True,early_stopper_restored=True,
            RNG_and_loader_streams_restored=True,next_epoch=next_epoch,invalid_states_rejected=rejected,optimizer_steps=0)
    finally:
        from src.utils.checkpoint import restore_rng
        restore_rng(original_rng,generators)


def validate_pipeline(info):
    root=info['root'];checks={}
    def forbidden_step(*args, **kwargs):
        raise AssertionError('Optimizer step forbidden during preparation')
    # Patch all experimental entry points and optimizer updates to fail if called.
    with patch.object(tr,'_epoch',side_effect=AssertionError('Epoch forbidden during preparation')), \
         patch.object(torch.optim.AdamW,'step',new=forbidden_step), \
         patch.object(tr,'_train_run',side_effect=AssertionError('Experiment forbidden during preparation')):
        with patch.object(tr,'preflight',side_effect=AssertionError('Disabled guard must exit before preflight')):
            require(tr.train_selected(root,confirm_full_training=False)==[],'Disabled guard failed')
            try:tr.train_selected(root,variants=['A5'],confirm_full_training=False)
            except ValueError:checks['A5_rejected']=True
        try:AblationDataset(root/'splits/fold_1_test.csv','test',False,info['preprocessing'])
        except ValueError:checks['test_dataset_rejected']=True
        checks['resume_state']=state_roundtrip(info)
        rows=[]
        # Static loss dispatch and model parameter counts; reuse Phase 11 forward/backward evidence.
        for variant in ['A0','A1','A2','A3','A4']:
            model=build_ablation(variant,info['config']['model']);criterion=AblationLoss(variant)
            expected=VARIANTS[variant]['use_morphology_supervision']
            require(criterion.morphology_enabled==expected,'Loss dispatch mismatch')
            if expected:
                require(type(criterion.morphology_loss) is MorphoNetLoss and criterion.morphology_loss.weights==info['config']['training']['loss'],'Frozen loss mismatch')
            else:require(criterion.morphology_loss is None and model.morphology_head is None,'CE-only model has supervised head')
            rows.append(dict(Variant=variant,Parameters=sum(p.numel() for p in model.parameters()),Loss='Frozen Phase 6 morphology' if expected else 'CE only',Status='PASS'))
            del model,criterion
        expected=pd.read_csv(root/'results/ablation/setup/ablation_architecture_summary.csv').set_index('Variant')
        require(all(row['Parameters']==expected.loc[row['Variant'],'Parameters'] for row in rows),'Frozen parameter mismatch')
        # Mocked files exercise skip/refuse decisions without writing to experimental directories.
        marker=dict(identity=run_identity(info,'A0',1),checkpoint_sha256={})
        with patch.object(Path,'exists',lambda p:p.name=='training_complete.json'),patch.object(Path,'read_text',return_value=json.dumps(marker)):
            require(tr.inspect_run(root,'A0',1,info)['action']=='skip','Completed run not preserved')
        with patch.object(Path,'exists',lambda p:p.name.startswith('fold_')),patch.object(Path,'iterdir',return_value=iter([Path('occupied')])):
            try:tr.inspect_run(root,'A0',1,info)
            except FileExistsError:checks['incomplete_overwrite_rejected']=True
        try:tr.inspect_run(root,'A0',1,info,resume=True,overwrite=True)
        except ValueError:checks['conflicting_controls_rejected']=True
    require(checks.get('A5_rejected') and checks.get('test_dataset_rejected') and checks.get('incomplete_overwrite_rejected') and checks.get('conflicting_controls_rejected'),'Safety decision checks failed')
    checks.update(disabled_guard_no_preflight_or_writes=True,completed_run_skip=True,experimental_epochs_executed=0,optimizer_steps_executed=0,real_images_loaded=0)
    return checks,pd.DataFrame(rows)


def prepare(root):
    info=preflight(root);folder=info['root']/'results/ablation/training_setup'
    experimental=info['root']/'results/ablation/training'
    before={p.relative_to(info['root']).as_posix():file_hash(p) for p in experimental.rglob('*') if p.is_file()}
    manifest_path=folder/'protected_artifacts.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text());check_protection(info['root'],manifest)
    else:manifest=protected_inventory(info['root'])
    checks,models=validate_pipeline(info)
    after={p.relative_to(info['root']).as_posix():file_hash(p) for p in experimental.rglob('*') if p.is_file()}
    require(before==after,'Preparation changed experimental artifacts')
    require(check_protection(info['root'],manifest)==len(manifest),'Protection failed')
    folder.mkdir(parents=True,exist_ok=True)
    if not manifest_path.exists():tr.atomic_json(manifest,manifest_path)
    tr.atomic_json(dict(contract=info['contract'],fingerprint=info['configuration_fingerprint']),folder/'training_configuration_fingerprint.json')
    tr.atomic_json(checks,folder/'non_training_validation.json');tr.atomic_csv(models,folder/'variant_readiness.csv')
    tr.atomic_csv(info['split_audit'],folder/'split_separation_audit.csv')
    audit=dict(passed=True,configuration_fingerprint=info['configuration_fingerprint'],sources=info['sources'],runtime=info['runtime'],
        supported_variants=list(TRAIN_VARIANTS_LOCAL),A5_training_blocked=True,training_disabled_by_default=True,
        real_training_runs_started=0,experimental_epochs_executed=0,optimizer_steps_executed=0,new_experimental_checkpoints=0,
        fake_summary_rows=0,test_lock=True,test_inference=0,test_performance_inspected=False,split_metadata_only=True,
        exact_resume_state_roundtrip_passed=True,overwrite_protection=True,protected_artifacts_unchanged=len(manifest),
        phase11_memory_results_reused=True,manual_launch_cell_not_executed=True)
    tr.atomic_json(audit,folder/'preparation_audit.json')
    print('Safe preparation passed: zero epochs, zero optimizer steps, zero experimental checkpoints; Test remains locked.')
    return info,audit,models

TRAIN_VARIANTS_LOCAL=('A0','A1','A2','A3','A4')
