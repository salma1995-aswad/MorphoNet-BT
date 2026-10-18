"""Phase 11 diagnostic validation only; no experimental training or test inference."""
from copy import deepcopy
from pathlib import Path
import gc
import json
import platform
import time
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from src.models.ablation import FLAGS, VARIANTS, build_ablation, validate_flags, AblationLoss
from src.models.morphonet_bt import MorphoNetBT
from src.data.dataset import BrainTumorDataset, verify_frozen_splits
from src.data.kfold import reload_splits, load_manifest, audit_splits
from src.utils.complexity import fingerprint, trace_layers, thop_measure
from src.utils.checkpoint import seed_everything
from src.evaluation.protocol import require, sha256, save_json


def prepare(root):
    root=Path(root).resolve(); output=root/'results/ablation/setup'; output.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((root/'configs/default.yaml').read_text())
    frozen=json.loads((root/'results/architecture/frozen_input_hashes.json').read_text())
    hashes=verify_frozen_splits(root); require(hashes==frozen['splits'],'Frozen splits changed')
    for path,digest in frozen['preprocessing_sources'].items(): require(sha256(root/path)==digest,'Frozen preprocessing changed')
    preprocessing=yaml.safe_load((root/'results/architecture/preprocessing_v1_approved.yaml').read_text())
    require({key:config[key] for key in preprocessing}==preprocessing,'Preprocessing mismatch')
    audits,coverage=audit_splits(reload_splits(root/'splits'),load_manifest(root/'results/dataset_statistics.csv'))
    expected=json.loads((root/'results/complexity/architecture_fingerprint.json').read_text())
    a5=build_ablation('A5',config['model'])
    require(fingerprint(a5,config['model'],root)==expected,'A5 frozen fingerprint mismatch')
    require(sum(p.numel() for p in a5.parameters())==579599,'A5 parameter count mismatch')
    checkpoints=[]
    for fold in range(1,6):
        path=root/f'checkpoints/main/fold_{fold}/best_macro_f1.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        require(checkpoint['architecture_fingerprint']==expected['fingerprint_sha256'],'Frozen checkpoint fingerprint mismatch')
        require(checkpoint['fold']==fold and checkpoint['configuration']==config,'Frozen checkpoint configuration mismatch')
        a5.load_state_dict(checkpoint['model_state_dict'],strict=True)
        checkpoints.append(dict(fold=fold,path=str(path),sha256=sha256(path),best_epoch=checkpoint['best_epoch']))
        del checkpoint
    del a5
    protected={}
    for folder in ['results/main','results/baselines/evaluation','results/complexity','results/architecture','splits','checkpoints/main']:
        for path in (root/folder).rglob('*'):
            if path.is_file(): protected[path.relative_to(root).as_posix()]=sha256(path)
    sources=['src/models/'+x for x in ['blocks.py','attention.py','fusion.py','morphonet_bt.py','__init__.py']]+['configs/default.yaml','src/training/losses.py']+list(frozen['preprocessing_sources'])
    for path in sources: protected[path]=sha256(root/path)
    invalid=0
    for flags in [dict(VARIANTS['A0'],use_baa=True),dict(VARIANTS['A0'],use_amgf=True),dict(VARIANTS['A0'],use_mss=1)]:
        try: validate_flags(flags)
        except ValueError: invalid+=1
    require(invalid==3,'Invalid flags accepted')
    save_json(output/'protected_artifacts_sha256.json',protected)
    future=dict(phase=12,training_enabled=False,requires_explicit_approval=True,seed=42,variants=VARIANTS,
        train_variants=['A0','A1','A2','A3','A4'],reuse_frozen_variants=['A5'],folds=[1,2,3,4,5],expected_new_runs=25,
        model=config['model'],preprocessing=preprocessing,class_mapping={0:'Meningioma',1:'Glioma',2:'Pituitary'},
        training=deepcopy(config['training']),a5_reference_checkpoints=checkpoints,
        target_policy='Masks may preserve frozen training augmentation; A0/A1 losses receive labels only. No GT model inputs.',
        training_start_policy='No training launcher in Phase 11. Require separately approved Phase 12 implementation.')
    future['training']['scheduler']['T_max']=100
    (root/'configs/ablation.yaml').write_text(yaml.safe_dump(future,sort_keys=False),encoding='utf-8')
    return dict(root=root,output=output,config=config,preprocessing=preprocessing,protected=protected,
        fingerprint=expected,checkpoints=checkpoints,split_hashes=hashes,invalid_flags_rejected=invalid)


def gradient_rows(model,variant,test):
    rows=[]
    for name,p in model.named_parameters():
        g=p.grad
        rows.append(dict(Variant=variant,Test=test,Parameter=name,Elements=p.numel(),Present=g is not None,
            Finite=g is not None and bool(torch.isfinite(g).all()),Nonzero=g is not None and bool(g.ne(0).any()),
            Norm=float(g.float().norm()) if g is not None else 0.))
    require(all(r['Present'] and r['Finite'] and r['Nonzero'] for r in rows),f'Gradient audit failed: {variant} {test}')
    return rows


def validate(context):
    root=context['root']; output=context['output']; config=context['config']
    seed_everything(42); torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available(),'CUDA required for the requested AMP batch-32 preflight')
    device=torch.device('cuda')
    dataset=BrainTumorDataset(root/'splits/fold_1_train.csv','train',config=context['preprocessing'],seed=42,include_targets=True)
    real=next(iter(DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0)))
    require(len(real['label'])==32,'Real batch must contain 32 training examples')
    pd.DataFrame({'sample_id':real['sample_id'],'PID':real['pid'],'label':real['label'].tolist()}).to_csv(output/'smoke_training_batch.csv',index=False)
    shapes=[]; gradients=[]; summaries=[]; complexity=[]; losses=[]; memory=[]; checks=[]
    expected={'stem':[2,32,128,128],'stage1':[2,48,64,64],'stage2':[2,80,32,32],'stage3':[2,128,16,16],
              'local_path':[2,128,16,16],'stage4':[2,192,8,8],'classifier':[2,3]}
    for variant,flags in VARIANTS.items():
        seed_everything(42); model=build_ablation(variant,config['model']).to(device)
        shapes_before={k:tuple(v.shape) for k,v in model.state_dict().items()}
        if not flags['use_morphology_supervision']: require(model.morphology_head is None,'Disabled heads allocated')
        require([len(getattr(model,f'stage{i}')) for i in range(1,5)]==[2,2,3,2],'Encoder depth changed')
        handles=[]; observed={}; morphology_calls=[]
        for name in [*expected,'global_path','fusion','pool']:
            module=getattr(model,name,None)
            if module is None: continue
            def hook(module,args,result,name=name):
                value=result[0] if isinstance(result,tuple) else result
                observed[name]=list(value.shape)
            handles.append(module.register_forward_hook(hook))
        if model.morphology_head is not None:
            handles.append(model.morphology_head.register_forward_hook(lambda *args:morphology_calls.append(1)))
        image=torch.randn(2,1,256,256,device=device)
        model.eval()
        with torch.inference_mode(): logits=model(image)
        for handle in handles: handle.remove()
        require(list(logits.shape)==[2,3] and bool(torch.isfinite(logits).all()),'Invalid MRI-only inference')
        for name,shape in expected.items(): require(observed.get(name)==shape,f'Shape changed: {variant}/{name}')
        require(bool(morphology_calls)==flags['use_baa'],'Unneeded morphology inference or missing guidance')
        for name,shape in observed.items(): shapes.append(dict(Variant=variant,Module=name,Shape=json.dumps(shape)))
        if variant=='A5':
            reference=MorphoNetBT(**config['model']).to(device).eval(); reference.load_state_dict(model.state_dict(),strict=True)
            with torch.inference_mode(): require(torch.equal(reference(image),logits),'A5 full-forward mismatch')
            del reference
        # FP32 real-batch backward checks all intended trainable tensors.
        model.train(); small={k:v[:2].to(device) for k,v in real.items() if torch.is_tensor(v)}
        loss_batch=small if flags['use_morphology_supervision'] else {'label':small['label']}
        outputs=model(small['image'],return_aux=True)
        if flags['use_morphology_supervision']:
            for key in ['region_logits','boundary_logits','distance']:
                require(list(outputs[key].shape)==[2,1,256,256] and bool(torch.isfinite(outputs[key]).all()),'Auxiliary output mismatch')
                shapes.append(dict(Variant=variant,Module=key,Shape=json.dumps(list(outputs[key].shape))))
        else:
            require(not any(k in outputs for k in ['region_logits','boundary_logits','distance']),'Disabled auxiliary head executed')
        criterion=AblationLoss(variant).to(device); loss=criterion(outputs,loss_batch)
        if flags['use_morphology_supervision']:
            expected_loss=loss['classification_loss']+.3*(loss['region_loss']+loss['boundary_loss']+.5*loss['distance_loss'])
            require(torch.equal(expected_loss,loss['total_loss']),'Frozen loss mismatch')
        else:
            require(set(loss)=={'classification_loss','total_loss'} and torch.equal(loss['total_loss'],torch.nn.functional.cross_entropy(outputs['logits'].float(),small['label'])),'Not CE-only')
        losses.append(dict(Variant=variant,BatchSize=2,**{k:float(v.detach()) for k,v in loss.items()}))
        loss['total_loss'].backward(); gradients.extend(gradient_rows(model,variant,'FP32 real batch'))
        model.zero_grad(set_to_none=True); model.eval()
        del outputs,loss,small,loss_batch,logits,image
        # Count only executed Conv/Linear operations, with two independent methods.
        image=torch.randn(1,1,256,256,device=device)
        for auxiliary in ([False,True] if flags['use_morphology_supervision'] else [False]):
            layers=trace_layers(model,image,auxiliary=auxiliary)
            measured=thop_measure(model,image,auxiliary=auxiliary)
            macs=int(layers.conv_linear_macs.sum())
            require(macs==measured['thop_conv_linear_macs'],'Complexity cross-check mismatch')
            layers.to_csv(output/f'{variant}_{"auxiliary" if auxiliary else "inference"}_layers.csv',index=False)
            complexity.append(dict(Variant=variant,Path='Auxiliary forward' if auxiliary else 'MRI-only inference',MACs=macs,FLOPs=2*macs))
            if not auxiliary: inference_macs=macs
        count=sum(p.numel() for p in model.parameters())
        summaries.append(dict(Variant=variant,**dict(zip(['MSS','Morphology Supervision','BAA','Dual Path','AMGF'],flags.values())),
            Parameters=count,TrainableParameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            **{'Parameters (M)':count/1e6,'Inference MACs':inference_macs,'Inference FLOPs':2*inference_macs}))
        if variant=='A5':
            require(inference_macs==275715968 and count==579599,'A5 frozen complexity mismatch')
        require(shapes_before=={k:tuple(v.shape) for k,v in model.state_dict().items()},'Profiling changed model state structure')
        del model,criterion,image; gc.collect(); torch.cuda.empty_cache()
        # A fresh disposable model gets exactly one successful AdamW update.
        seed_everything(42); model=build_ablation(variant,config['model']).to(device).train()
        batch={k:v.to(device) for k,v in real.items() if torch.is_tensor(v)}
        if not flags['use_morphology_supervision']: batch={k:batch[k] for k in ['image','label']}
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
        criterion=AblationLoss(variant).to(device)
        before=model.classifier.weight.detach().clone()
        torch.cuda.reset_peak_memory_stats(); start=time.perf_counter(); skipped=0
        for attempt in range(8):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                outputs=model(batch['image'],return_aux=flags['use_morphology_supervision'])
                loss=criterion(outputs,batch)
            scaler.scale(loss['total_loss']).backward(); scaler.unscale_(optimizer)
            finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            if finite: gradients.extend(gradient_rows(model,variant,'AMP batch 32'))
            scale_before=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if finite: break
            skipped+=1; require(scaler.get_scale()<scale_before,'AMP overflow did not lower scale')
        require(finite and not torch.equal(before,model.classifier.weight),'No successful optimizer update')
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Nonfinite parameters after AdamW')
        require(all(int(state['step'])==1 for state in optimizer.state.values()),'Expected exactly one successful update')
        torch.cuda.synchronize()
        memory.append(dict(Variant=variant,BatchSize=32,AMP=True,Optimizer='AdamW',SuccessfulSteps=1,OverflowSkippedSteps=skipped,
            FinalScale=float(scaler.get_scale()),PeakAllocatedMiB=torch.cuda.max_memory_allocated()/2**20,
            PeakReservedMiB=torch.cuda.max_memory_reserved()/2**20,ElapsedSeconds=time.perf_counter()-start,Status='PASS'))
        checks.append(dict(Variant=variant,Construction=True,MRIOnly=True,ForwardBackward=True,AllTrainableGradients=True,AuxiliaryCorrect=True,Batch32AdamW=True))
        del model,optimizer,scaler,criterion,batch,before,outputs,loss; gc.collect(); torch.cuda.empty_cache()
        print(f'{variant}: shapes, FP32 backward, gradients, complexity and temporary AMP batch-32 AdamW PASS.')
    tables={'ablation_architecture_summary':pd.DataFrame(summaries),'ablation_complexity':pd.DataFrame(complexity),
        'ablation_shape_audit':pd.DataFrame(shapes),'ablation_gradient_audit':pd.DataFrame(gradients),
        'ablation_memory_preflight':pd.DataFrame(memory),'ablation_real_batch_losses':pd.DataFrame(losses)}
    for name,frame in tables.items(): frame.to_csv(output/f'{name}.csv',index=False,float_format='%.17g')
    for path,digest in context['protected'].items(): require(sha256(root/path)==digest,f'Protected file changed: {path}')
    audit=dict(A0_A5_validated=True,A5_frozen_fingerprint_exact=True,A5_parameters=579599,MRI_only_inference=True,
        GT_morphology_at_inference=False,CE_only_A0_A1=True,frozen_loss_A2_A5=True,BAA_MRI_predicted_boundary_only=True,
        encoder_dimensions_preserved=True,forward_backward_pass=True,all_trainable_gradients_pass=True,
        independent_complexity_counts_agree=True,batch32_AdamW_all_pass=True,temporary_optimizer_steps=6,
        full_training_runs_started=0,experimental_checkpoints_saved=0,test_images_accessed=0,
        frozen_artifacts_unchanged=len(context['protected']),invalid_configurations_rejected=context['invalid_flags_rejected'])
    save_json(output/'ablation_validation.json',dict(scientific_audit=audit,variants=checks))
    save_json(output/'final_scientific_audit.json',audit)
    import torchvision
    metadata=dict(seed=42,variants=VARIANTS,full_fingerprint=context['fingerprint'],split_hashes=context['split_hashes'],
        preprocessing=context['preprocessing'],a5_reuse=context['checkpoints'],future_training_runs=25,
        primary_inference_shape=[1,1,256,256],MAC_convention='Conv/Linear only; 1 MAC = 2 FLOPs',
        diagnostic_data='First 32 samples of frozen Fold 1 training split, frozen training augmentation at epoch 0',
        software=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda),
        GPU=torch.cuda.get_device_name(0),sources={p:sha256(root/p) for p in ['src/models/ablation.py','src/utils/ablation_setup.py','configs/ablation.yaml','src/training/losses.py']},
        design_limitations=['Sequential effects are order-dependent and conditional, not independent component effects.',
            'Parameter count and compute change with mechanisms; no unused capacity is added.',
            'A3 to A4 adds contextual path and mean fusion as a package; A4 to A5 replaces mean with AMGF.',
            'Frozen training augmentation may use masks internally; A0/A1 model and loss are MRI/label-only.',
            'Global path is dilated contextual convolution, not all-to-all global attention.'])
    save_json(output/'ablation_setup_metadata.json',metadata)
    return tables,audit
