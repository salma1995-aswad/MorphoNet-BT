"""Audited held-out A4 explanation execution. No optimizer is constructed."""
from pathlib import Path
import json
import os
import time
import platform
import numpy as np
import pandas as pd
import torch
from src.phase14_statistics import sha, read, dump, save, verify_manifest
from src.ablation.protocol import preflight as frozen_preflight, run_identity
from src.ablation.training import validate_checkpoint
from src.models.ablation import build_ablation
from src.data.dataset import load_checked_sample
from src.data.preprocessing import preprocess
from .core import LAYERS, GradCAM, state_digest, resize, localization, morphology, synthetic_tests

PROBS=['prob_meningioma','prob_glioma','prob_pituitary']
CLASSES=['Meningioma','Glioma','Pituitary']
SETTINGS=dict(seed=42,batch_size=16,bootstrap_iterations=10000,probability_atol=1e-5,probability_rtol=1e-4,
              primary_layer='stage3',secondary_layer='stage4',heatmap_thresholds=[.3,.5,.7],morphology_threshold=.5,
              boundary_tolerance_pixels=2,epsilon=1e-8,precision='FP32; no autocast; TF32 disabled')
POLICY='''Primary layer stage3 (128x16x16) is the late encoder before both global/local paths and offers more spatial detail than stage4 (192x8x8), which is secondary. Selection is architectural, before mask-overlap analysis. Use predicted-class logits for every slice and additionally true-class logits for errors. Apply spatial mean gradients, weighted feature sum and ReLU, bilinear upsampling with align_corners=False directly to original image size, then divide by positive peak (no minimum subtraction). Zero-positive maps receive zero localization scores and are explicitly counted, not excluded. Peak ties use the first maximum in row-major order. Ratio epsilon is 1e-8. Fixed heatmap thresholds 0.3/0.5/0.7 are descriptive, never selected for best overlap. Original-image masks evaluate Grad-CAM; auxiliary morphology is evaluated on frozen 256x256 targets. Region/boundary logits use sigmoid and threshold 0.5; distance is already sigmoid in the model. Boundary F1 uses Euclidean tolerance 2 pixels on the 256x256 grid, no physical-millimeter interpretation. Primary results are means of per-slice metrics; patient summaries average slices within each patient. Cluster bootstrap resamples patients, seed 42, 10000 iterations. No significance tests.

Representative selection is fixed before Grad-CAM: for each true class, highest-confidence correct, lowest-confidence correct, seeded random correct, highest-confidence error, and seeded random error; numeric Sample ID breaks confidence ties, and previously used patients are avoided when possible. These selections never inspect heatmaps or mask overlap. Additional failure illustrations deliberately show the minimum and maximum stage3 attention mass among errors of each class, with explicit extreme-case labels; these are diagnostic extremes, never representative performance evidence. No PID appears in figure labels. A5 comparison is omitted to keep the analysis focused on A4 and avoid doubling checkpoint/explanation execution for an optional, non-primary question.'''

def log(out,message):
    stamp=time.strftime('%Y-%m-%d %H:%M:%S')
    with (out/'metadata/progress.log').open('a',encoding='utf-8') as f:f.write(f'{stamp} {message}\n')
    print(message,flush=True)

def checkpoint(root,fold,info):
    path=root/f'results/ablation/training/A4/fold_{fold}/best_macro_f1.pt'
    marker=json.loads(path.with_name('training_complete.json').read_text())
    digest=sha(path)
    assert digest==marker['checkpoint_sha256']['best_macro_f1.pt']
    receipt=json.loads((root/f'results/ablation/evaluation/A4/fold_{fold}/evaluation_receipt.json').read_text())
    assert digest==receipt['identity']['checkpoint_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    validate_checkpoint(saved,info,run_identity(info,'A4',fold))
    assert saved['epoch']==saved['best_epoch']==marker['best_epoch']
    model=build_ablation('A4',info['config']['model'])
    model.load_state_dict(saved['model_state_dict'],strict=True)
    assert sum(p.numel() for p in model.parameters())==575359
    assert model.fusion is None and model.use_dual_path and model.use_baa
    for value in model.state_dict().values():assert torch.isfinite(value).all()
    model.eval().requires_grad_(False)
    row=dict(fold=fold,checkpoint=str(path),sha256=digest,best_epoch=saved['epoch'],parameters=575359,strict_load=True,variant='A4',AMGF=False)
    return model,row

def protected_snapshot(root,out):
    prior=json.loads((root/'results/statistical_analysis/completion_receipt.json').read_text())
    verify_manifest(root,prior['input_hashes']);verify_manifest(root,prior['output_hashes'])
    snapshot={}
    for dirname in ['results','checkpoints','configs','splits','logs','notebooks','src','scripts']:
        for path in (root/dirname).rglob('*'):
            if not path.is_file() or out in path.parents or '__pycache__' in path.parts:continue
            if 'explainability' in path.parts or 'phase15' in path.name.lower() or path.name.startswith('15_'):continue
            snapshot[path.relative_to(root).as_posix()]=sha(path)
    # Protect original data containers as well, not just metadata.
    for source in read(root/'results/dataset_statistics.csv').source.unique():snapshot[str(Path(source).resolve())]=sha(source)
    return snapshot

def selected_examples(frozen):
    rng=np.random.default_rng(42);used=set();rows=[]
    frame=frozen.copy();frame['confidence']=frame[PROBS].max(axis=1);frame['numeric_id']=frame.sample_id.astype(int)
    for label in range(3):
        for category,correct,order in [('high_confidence_correct',True,'high'),('low_confidence_correct',True,'low'),('random_correct',True,'random'),('high_confidence_error',False,'high'),('random_error',False,'random')]:
            part=frame[(frame.true_label==label)&((frame.true_label==frame.predicted_label)==correct)]
            distinct=part[~part.PID.isin(used)]
            if not distinct.empty:part=distinct
            if part.empty:continue
            part=part.sort_values('numeric_id')
            row=part.iloc[int(rng.integers(len(part)))] if order=='random' else part.sort_values(['confidence','numeric_id'],ascending=[order=='low',True]).iloc[0]
            used.add(row.PID)
            rows.append(dict(sample_id=row.sample_id,PID=row.PID,fold=int(row.fold),true_class=CLASSES[label],predicted_class=CLASSES[int(row.predicted_label)],confidence=float(row.confidence),selection=category,selection_uses_overlap=False))
    return pd.DataFrame(rows)

def batch_data(frame,config):
    rows=frame.to_dict('records');items=[]
    for row in rows:
        raw,mask=load_checked_sample(row,True)
        arrays,metadata=preprocess(raw,mask,'test',config)
        assert not metadata['augmentation_active']
        # MRI tensors must be exactly independent of whether targets are requested.
        image_only,_=preprocess(raw,None,'test',config)
        np.testing.assert_array_equal(arrays['image'],image_only['image'])
        assert mask.any() and mask.shape==raw.shape and np.isfinite(mask).all()
        items.append((row,raw,mask,arrays))
    return items,torch.from_numpy(np.stack([v[3]['image'] for v in items])).cuda()

def compare(prob,rows,frozen):
    ref=frozen.loc[[str(r['sample_id']) for r in rows]]
    np.testing.assert_array_equal(ref.PID,[str(r['PID']) for r in rows])
    np.testing.assert_array_equal(ref.fold,[int(r['fold']) for r in rows])
    np.testing.assert_array_equal(ref.true_label,[int(r['original_label'])-1 for r in rows])
    np.testing.assert_array_equal(prob.argmax(1),ref.predicted_label)
    np.testing.assert_allclose(prob,ref[PROBS].to_numpy(),atol=SETTINGS['probability_atol'],rtol=SETTINGS['probability_rtol'])
    return float(np.abs(prob-ref[PROBS].to_numpy()).max())

def run(root):
    root=Path(root).resolve();out=root/'results/explainability'
    for name in ['metadata','tables','figures','maps']:(out/name).mkdir(parents=True,exist_ok=True)
    source_hashes={p.relative_to(root).as_posix():sha(p) for p in (root/'src/explainability').glob('*.py')}
    completed=out/'metadata/completion_receipt.json'
    if completed.exists():
        done=json.loads(completed.read_text());assert done['source_hashes']==source_hashes
        verify_manifest(root,done['protected']);verify_manifest(root,done['outputs'])
        log(out,'Verified completed Phase 15 outputs reused; no new inference.');return out
    started=time.perf_counter()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    dump(out/'metadata/analysis_policy.json',dict(settings=SETTINGS,policy=POLICY,source_hashes=source_hashes))
    dump(out/'metadata/synthetic_validation.json',synthetic_tests())
    log(out,'Synthetic tests passed. Verifying previous-phase artifacts and frozen protocol.')
    protected=protected_snapshot(root,out);dump(out/'metadata/protected_sha256.json',protected)
    info=frozen_preflight(root)
    frozen=read(root/'results/ablation/evaluation/A4/oof_predictions.csv')
    assert len(frozen)==3064 and frozen.sample_id.is_unique and frozen.PID.nunique()==233
    selection=selected_examples(frozen);save(selection,out/'tables/representative_sample_manifest.csv')
    frozen=frozen.set_index('sample_id')
    inventory=[];shapes={}
    for fold in range(1,6):
        model,row=checkpoint(root,fold,info);before=state_digest(model)
        model.cuda()
        with GradCAM(model) as explainer:
            smoke=explainer.compute(torch.randn(2,1,256,256,device='cuda'),torch.tensor([0,1],device='cuda'))
        assert before==state_digest(model)
        assert smoke['feature_shapes']=={'stage3':[128,16,16],'stage4':[192,8,8]}
        assert all(np.isfinite(v).all() for v in smoke['cam_pred'].values())
        row.update(smoke_test='PASS',state_unchanged=True);inventory.append(row);shapes=smoke['feature_shapes']
        del model;torch.cuda.empty_cache()
    save(pd.DataFrame(inventory),out/'tables/checkpoint_audit.csv')
    log(out,'All five A4 checkpoints strictly loaded; parameter count and forward/backward smoke tests passed.')
    # All held-out predictions must pass before any full-dataset explanations are generated.
    reproduction=[];raw_hashes=[]
    for fold in range(1,6):
        model,_=checkpoint(root,fold,info);model.cuda();before=state_digest(model)
        frame=read(root/f'splits/fold_{fold}_test.csv');maxdelta=0.
        for start in range(0,len(frame),16):
            items,images=batch_data(frame.iloc[start:start+16],info['preprocessing'])
            with torch.inference_mode():prob=model(images).softmax(1).cpu().numpy()
            maxdelta=max(maxdelta,compare(prob,[v[0] for v in items],frozen))
            for row,raw,mask,arrays in items:
                raw_hashes.append(dict(sample_id=str(row['sample_id']),PID=str(row['PID']),fold=fold,raw_height=raw.shape[0],raw_width=raw.shape[1],mask_sha256=__import__('hashlib').sha256(mask.tobytes()).hexdigest()))
        assert before==state_digest(model)
        reproduction.append(dict(fold=fold,slices=len(frame),patients=frame.PID.nunique(),class_mismatches=0,max_absolute_probability_error=maxdelta,atol=1e-5,rtol=1e-4,state_unchanged=True))
        log(out,f'Prediction reproducibility fold {fold}: {len(frame)} slices, zero class mismatches, max probability difference {maxdelta:.3g}.')
        del model;torch.cuda.empty_cache()
    save(pd.DataFrame(reproduction),out/'tables/prediction_reproducibility.csv');save(pd.DataFrame(raw_hashes),out/'metadata/annotation_inventory.csv')
    assert len(raw_hashes)==3064 and len({r['sample_id'] for r in raw_hashes})==3064
    annotations={r['sample_id']:r for r in raw_hashes}
    log(out,'Full 3064-slice reproducibility and annotation preflight passed. Starting explanations.')
    camrows=[];ba_rows=[];regionrows=[];boundaryrows=[];distancerows=[]
    for fold in range(1,6):
        model,_=checkpoint(root,fold,info);model.cuda();before=state_digest(model)
        frame=read(root/f'splits/fold_{fold}_test.csv')
        with GradCAM(model) as explainer:
            for start in range(0,len(frame),16):
                items,images=batch_data(frame.iloc[start:start+16],info['preprocessing'])
                labels=torch.tensor([int(v[0]['original_label'])-1 for v in items],device='cuda')
                result=explainer.compute(images,labels)
                compare(result['probabilities'],[v[0] for v in items],frozen)
                for i,(row,raw,mask,arrays) in enumerate(items):
                    sid=str(row['sample_id']);true=int(row['original_label'])-1;pred=int(result['predicted'][i])
                    assert __import__('hashlib').sha256(mask.tobytes()).hexdigest()==annotations[sid]['mask_sha256']
                    identity=dict(sample_id=sid,PID=str(row['PID']),fold=fold,true_label=true,predicted_label=pred,true_class=CLASSES[true],predicted_class=CLASSES[pred],correct=int(true==pred),confidence=float(result['probabilities'][i,pred]))
                    maps={f'{key}_logits_low':value[i] for key,value in result['native'].items()}
                    maps['baa']=result['baa'][i]
                    for target in ['pred']+(['true'] if pred!=true else []):
                        for layer in LAYERS:
                            native=result[f'cam_{target}'][layer][i];maps[f'cam_{target}_{layer}']=native
                            metrics=localization(resize(native,raw.shape),mask)
                            camrows.append(dict(**identity,target='predicted' if target=='pred' else 'true',layer=layer,native_height=native.shape[0],native_width=native.shape[1],**metrics))
                    ba_rows.append(dict(**identity,**localization(resize(result['baa'][i],raw.shape),mask)))
                    reg,bound,dist=morphology(result['region'][i],result['boundary'][i],result['distance'][i],arrays)
                    regionrows.append(dict(**identity,**reg));boundaryrows.append(dict(**identity,**bound));distancerows.append(dict(**identity,**dist))
                    # Native FP32 maps retain exact reconstruction; no redundant full-size data copies.
                    np.savez_compressed(out/f'maps/{sid}.npz',**maps)
                if start%160==0:log(out,f'Grad-CAM fold {fold}: {min(start+16,len(frame))}/{len(frame)} slices.')
        assert state_digest(model)==before and all(p.grad is None for p in model.parameters())
        log(out,f'Fold {fold} explanations complete; all weights and buffers unchanged.')
        del model;torch.cuda.empty_cache()
    cams=pd.DataFrame(camrows)
    save(cams,out/'tables/gradcam_localization_per_slice.csv')
    save(pd.DataFrame(ba_rows),out/'tables/baa_localization_per_slice.csv')
    for name,rows in [('region',regionrows),('boundary',boundaryrows),('distance',distancerows)]:save(pd.DataFrame(rows),out/f'tables/morphology_{name}_metrics.csv')
    for name,df in [('cam',cams),('region',pd.DataFrame(regionrows)),('boundary',pd.DataFrame(boundaryrows)),('distance',pd.DataFrame(distancerows))]:
        assert np.isfinite(df.select_dtypes('number').to_numpy()).all(),name
    for layer in LAYERS:
        primary=cams[(cams.target=='predicted')&(cams.layer==layer)]
        assert len(primary)==3064 and primary.sample_id.is_unique and primary.PID.nunique()==233
        assert len(cams[(cams.target=='true')&(cams.layer==layer)])==227
    from .reporting import build_report
    build_report(root,out,cams,selection)
    log(out,'Quantitative summaries, correlations, representative panels and publication figures completed.')
    verify_manifest(root,protected)
    audit=dict(status='PASS',slices=3064,patients=233,folds=5,parameters=575359,classification_errors=227,correct_slices=2837,
               predicted_class_maps=6128,true_class_error_maps=454,feature_shapes=shapes,primary_layer='stage3',secondary_layer='stage4',
               prediction_class_mismatches=0,max_absolute_probability_error=max(r['max_absolute_probability_error'] for r in reproduction),
               probabilities_within_predeclared_tolerance=True,MRI_only=True,mask_input_invariance=True,all_annotations_available=True,
               all_maps_and_metrics_finite=True,patient_leakage=False,held_out_alignment=True,optimizer_steps=0,training_runs=0,
               weights_and_buffers_unchanged=True,AMGF_added=False,checkpoints_reselected=False,protected_files_unchanged=True,protected_file_count=len(protected),
               full_coverage=True,omitted_samples=0,bootstrap_iterations=10000,seed=42,synthetic_tests='PASS',A5_comparison='skipped: optional secondary scope and additional checkpoint/compute complexity',
               phase16_started=False,elapsed_seconds=time.perf_counter()-started,
               software=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0)))
    dump(out/'final_scientific_audit.json',audit)
    (out/'PHASE15_EXPLAINABILITY_REPORT.md').write_text((out/'PHASE15_EXPLAINABILITY_REPORT.md').read_text(encoding='utf-8')+'\n\n## Scientific audit\n\n```json\n'+json.dumps(audit,indent=2)+'\n```\n',encoding='utf-8')
    outputs={p.relative_to(root).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['progress.log','completion_receipt.json','notebook_execution_audit.json']}
    dump(completed,dict(status='complete',source_hashes=source_hashes,protected=protected,outputs=outputs))
    log(out,'Phase 15 completed. Scientific integrity PASS. STOP before Phase 16.')
    return out
