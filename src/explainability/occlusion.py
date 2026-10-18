"""Phase 15.1: fixed, MRI-only diagnostic perturbations of frozen A4 models."""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label
from scipy.signal import fftconvolve
from src.phase14_statistics import sha, read, save, dump, verify_manifest
from src.ablation.protocol import preflight as frozen_preflight
from src.data.dataset import load_checked_sample
from src.data.preprocessing import preprocess
from .pipeline import checkpoint, compare, CLASSES, PROBS
from .core import state_digest

SETTINGS=dict(seed=42,n_controls=10,bootstrap_iterations=10000,original_batch_size=16,inference_batch_size=11,
    probability_atol=1e-5,probability_rtol=1e-4,head_signal_fraction_of_p99=0.05,head_inner_margin_pixels=8,
    head_envelope='Otsu on p99-clipped raw MRI, Gaussian sigma 1.5, 5-pixel elliptical closing, largest component, hole filling',
    minimum_nonair_control_fraction=0.95,local_ring_radius_pixels=5,primary_replacement=0.,secondary='local_nontumor_ring_median',
    strategies=['zero','local_median'],control_response_sd_ddof=0,formal_pvalues=False)
POLICY='''All settings are fixed before perturbed-model responses are inspected. The primary intervention replaces annotated tumor pixels or a translated, identical-shaped control with zero AFTER frozen 256x256 per-image z-score normalization. No renormalization, augmentation, cropping, rotation, or resizing follows occlusion. The secondary method uses the median normalized intensity in an external five-pixel Euclidean ring, excluding the original tumor and the target region, identically for tumor and controls. Both methods use the same control locations and the complete available cohort. No generative inpainting is used.

No validated intracranial/brain mask is supplied in the original dataset (available annotations are tumor mask/border). Controls therefore use a deliberately labelled HEAD-INTERIOR NON-AIR PROXY, not a brain segmentation. On the linearly resized raw MRI, signal pixels exceed 5% of the image 99th percentile. The head envelope is constructed separately using Otsu thresholding of a p99-clipped, Gaussian-smoothed (sigma 1.5 pixels) raw image, a 5-pixel elliptical closing, the largest 8-connected foreground component, and hole filling. Pixels at Euclidean distance greater than 8 pixels from its exterior form the conservative head interior. This gross foreground step was corrected after geometry-only inspection detected background noise in the initial low-threshold envelope, before any perturbed-model inference. Every translated tumor-shaped control must lie entirely in this interior, overlap the original tumor at zero pixels, and have at least 95% signal pixels. This avoids empty-air controls and suppresses the peripheral rim, but cannot guarantee exclusion of skull, CSF, face, or non-brain structures. Anatomical homology and intensity matching are not claimed. No relaxed background fallback is allowed.

All integer translations that keep the original binary mask inside the 256x256 image are enumerated by FFT cross-correlation and then verified with exact Boolean checks. Up to 10 unique eligible translations are sampled uniformly without replacement using a stable per-Sample-ID seed derived from seed 42. Control regions may overlap EACH OTHER; this is not independence, and their responses are first averaged within a slice. A sample with fewer than 10 controls is flagged, never silently dropped. Zero-control samples retain original/tumor inference; paired fields explicitly read unavailable_no_controls and summaries report available matched samples. There is no zero imputation. An empty local ring triggers an integrity stop rather than an invented replacement.

Primary effect is (original true-class logit - tumor-occluded true-class logit) minus the within-slice mean control drop. Negative changes are retained. True-probability, ORIGINAL predicted-class logit, true-vs-strongest-competitor margin, flips, originally-correct losses and originally-incorrect recoveries are secondary. Patients are primary analysis units. Means are accompanied by 10000 shared patient-bootstrap percentile 95% intervals, seed 42. Slice-weighted intervals preserve every sampled patient's slices and paired controls. Correct/incorrect slice groups can share patients. No p-value is reported because tumor and non-tumor anatomical interventions are not assumed exchangeable; the audit is exploratory, post-hoc and conditional on frozen trained models. Correlations use patient summaries and Spearman rho with patient-bootstrap intervals.

Qualitative selection before perturbed responses: reuse the three Phase 15 seeded random-correct examples (Sample IDs 66, 2570, 1315) and three high-confidence error examples (36, 2172, 1566), with control index 0, never the most favorable response. Show every selected sample even if no control exists. The intracranial-proxy limitation, class-dependent tumor size, intervention artifacts, and lack of independent clinical/external validation preclude causal-reliance or clinical-trust claims. No frozen model or preprocessing is tuned, and Phase 16 is not started.'''

def log(out,text):
    with (out/'metadata/progress.log').open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+text+'\n')
    print(text,flush=True)

def control_domain(raw):
    raw=cv2.resize(np.asarray(raw,np.float32),(256,256),interpolation=cv2.INTER_LINEAR)
    cutoff=.05*float(np.percentile(raw,99))
    assert cutoff>0
    signal=raw>cutoff
    # Otsu separates gross head foreground from acquisition-background noise;
    # this is still explicitly not an intracranial segmentation.
    scaled=np.rint(np.clip(raw/(cutoff/.05),0,1)*255).astype(np.uint8)
    smoothed=cv2.GaussianBlur(scaled,(0,0),1.5)
    _,foreground=cv2.threshold(smoothed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    foreground=cv2.morphologyEx(foreground,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    components,n=label(foreground>0,structure=np.ones((3,3)))
    sizes=np.bincount(components.ravel());sizes[0]=0
    assert n>0
    envelope=binary_fill_holes(components==sizes.argmax())
    # Padding makes the image exterior background even for a head touching the edge.
    depth=distance_transform_edt(np.pad(envelope,1))[1:-1,1:-1]
    interior=depth>8
    return interior,signal,cutoff

def controls(mask,interior,signal,sample_id):
    mask=np.asarray(mask,bool);ys,xs=np.where(mask);assert len(ys)>0
    y0,y1,x0,x1=int(ys.min()),int(ys.max()+1),int(xs.min()),int(xs.max()+1)
    template=mask[y0:y1,x0:x1];area=int(mask.sum());kernel=template[::-1,::-1].astype(float)
    def corr(a):return np.rint(fftconvolve(a.astype(float),kernel,mode='valid')).astype(int)
    inside,overlap,signal_count=corr(interior),corr(mask),corr(signal)
    valid=(inside==area)&(overlap==0)&(signal_count>=int(np.ceil(.95*area)))
    positions=np.argwhere(valid)
    identity=int.from_bytes(hashlib.sha256(str(sample_id).encode()).digest()[:8],'little')
    rng=np.random.default_rng(np.random.SeedSequence([42,identity]))
    chosen=positions[rng.choice(len(positions),size=min(10,len(positions)),replace=False)] if len(positions) else np.empty((0,2),int)
    result=[]
    for y,x in chosen:
        region=np.zeros_like(mask);region[y:y+template.shape[0],x:x+template.shape[1]]=template
        assert int(region.sum())==area and not (region&mask).any() and interior[region].all() and signal[region].mean()>=.95
        result.append(dict(top_y=int(y),top_x=int(x),dy=int(y-y0),dx=int(x-x0),signal_fraction=float(signal[region].mean())))
    return result,dict(eligible_locations=len(positions),n_controls=len(result),tumor_pixels=area,tumor_fraction=area/mask.size,
        tumor_centroid_y=float(ys.mean()),tumor_centroid_x=float(xs.mean()),bbox_y=y0,bbox_x=x0,bbox_h=y1-y0,bbox_w=x1-x0,
        controls_shortfall=len(result)<10,paired_available=len(result)>0)

def translate(mask,dy,dx):
    y,x=np.where(mask);out=np.zeros_like(mask,bool);yy,xx=y+int(dy),x+int(dx)
    assert yy.min()>=0 and xx.min()>=0 and yy.max()<256 and xx.max()<256
    out[yy,xx]=True
    assert out.sum()==mask.sum()
    return out

def local_value(image,region,tumor):
    ring=(distance_transform_edt(~region)<=5)&~region&~tumor
    assert ring.any(), 'Empty local non-tumor ring; stop rather than fabricate a value.'
    return float(np.median(image[ring])),int(ring.sum())

def perturb(image,region,value):
    out=image.copy();out[region]=np.float32(value)
    assert np.array_equal(out[~region],image[~region]) and np.all(out[region]==np.float32(value))
    assert np.isfinite(out).all()
    return out

def response(logits,true,pred):
    logits=np.asarray(logits,float);prob=np.exp(logits-logits.max());prob/=prob.sum()
    return dict(**{f'logit_{k}':float(v) for k,v in enumerate(logits)},**{f'prob_{k}':float(v) for k,v in enumerate(prob)},
        predicted_label=int(logits.argmax()),confidence=float(prob.max()),true_logit=float(logits[true]),
        original_predicted_logit=float(logits[pred]),true_probability=float(prob[true]),margin=float(logits[true]-np.max(np.delete(logits,true))),correct=int(logits.argmax()==true))

def differences(original,perturbed):
    return dict(true_logit_drop=original['true_logit']-perturbed['true_logit'],
        true_probability_drop=original['true_probability']-perturbed['true_probability'],
        original_predicted_logit_drop=original['original_predicted_logit']-perturbed['original_predicted_logit'],
        margin_drop=original['margin']-perturbed['margin'],flip=int(original['predicted_label']!=perturbed['predicted_label']))

def synthetic_tests():
    mask=np.zeros((256,256),bool);mask[100:106,110:117]=True
    domain=np.zeros_like(mask);domain[20:235,20:235]=True
    records,audit=controls(mask,domain,domain,'synthetic');records2,_=controls(mask,domain,domain,'synthetic')
    assert records==records2 and len(records)==10 and audit['tumor_pixels']==42
    for r in records:
        c=translate(mask,r['dy'],r['dx']);assert c.sum()==42 and not (c&mask).any() and domain[c].all()
    none,a=controls(mask,np.zeros_like(mask),domain,'none');assert not none and a['n_controls']==0
    image=np.arange(256*256,dtype=np.float32).reshape(256,256)/1000
    o=perturb(image,mask,0);assert np.array_equal(o[~mask],image[~mask])
    v,n=local_value(image,mask,mask);assert np.isfinite(v) and n>0
    original=response([2.,1.,0.],0,0);changed=response([3.,1.,0.],0,0)
    assert differences(original,changed)['true_logit_drop']==-1
    return dict(status='PASS',checks=['deterministic matched translations','zero-control handling','unaltered outside pixels','finite local replacement','negative signed drops retained'])

def snapshot(root,out):
    prior=json.loads((root/'results/explainability/metadata/completion_receipt.json').read_text())
    verify_manifest(root,prior['protected']);verify_manifest(root,prior['outputs'])
    hashes={}
    for folder in ['results','checkpoints','configs','splits','logs','src','scripts','notebooks']:
        for p in (root/folder).rglob('*'):
            if not p.is_file() or out in p.parents or '__pycache__' in p.parts:continue
            if p.name.startswith('15_1_') or 'phase15_1' in p.name or p.name.startswith('occlusion'):continue
            hashes[p.relative_to(root).as_posix()]=sha(p)
    for source in read(root/'results/dataset_statistics.csv').source.unique():hashes[str(Path(source).resolve())]=sha(source)
    return hashes

def run(root):
    root=Path(root).resolve();out=root/'results/explainability/occlusion_audit'
    for name in ['tables','figures','metadata','cache']:(out/name).mkdir(parents=True,exist_ok=True)
    code={p.relative_to(root).as_posix():sha(p) for p in (root/'src/explainability').glob('occlusion*.py')}
    receipt_path=out/'metadata/completion_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text());assert receipt['source_hashes']==code
        verify_manifest(root,receipt['protected']);verify_manifest(root,receipt['outputs'])
        log(out,'Complete Phase 15.1 results verified and reused; no new inference.');return out
    start_time=time.perf_counter()
    inference_receipt=out/'metadata/inference_completion_receipt.json'
    if inference_receipt.exists():
        sealed=json.loads(inference_receipt.read_text())
        assert sealed['settings']==SETTINGS and sealed['status']=='complete'
        verify_manifest(root,sealed['outputs'])
        protected=json.loads((out/'metadata/protected_sha256.json').read_text())
        verify_manifest(root,protected)
        log(out,'Verified sealed inference outputs; resuming statistics and reporting without new model inference.')
        def t(name):return read(out/'tables'/name)
        return finish_analysis(root,out,t('occlusion_per_control.csv'),t('occlusion_per_slice.csv'),t('original_predictions.csv'),
            t('occlusion_sampling_audit.csv'),protected,code,t('prediction_reproducibility.csv').to_dict('records'),t('control_locations.csv'),start_time)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    dump(out/'metadata/predeclared_policy.json',dict(settings=SETTINGS,policy=POLICY,source_hashes=code))
    dump(out/'metadata/synthetic_tests.json',synthetic_tests())
    log(out,'Synthetic checks passed; verifying all frozen Phase 1-15 artifacts.')
    protected=snapshot(root,out);dump(out/'metadata/protected_sha256.json',protected)
    info=frozen_preflight(root)
    frozen=read(root/'results/ablation/evaluation/A4/oof_predictions.csv').set_index('sample_id')
    assert len(frozen)==3064 and frozen.index.is_unique and frozen.PID.nunique()==233
    sampling=[];locations=[];originals=[];inventory=[];repro=[]
    # Reproduce the entire original cohort before any perturbed-model forward pass.
    for fold in range(1,6):
        model,inv=checkpoint(root,fold,info);model.cuda();before=state_digest(model);inventory.append(inv)
        frame=read(root/f'splits/fold_{fold}_test.csv');maxdiff=0.
        for offset in range(0,len(frame),16):
            rows=frame.iloc[offset:offset+16].to_dict('records');images=[];prepared=[]
            for row in rows:
                raw,rawmask=load_checked_sample(row,True);arrays,meta=preprocess(raw,rawmask,'test',info['preprocessing'])
                no_targets,_=preprocess(raw,None,'test',info['preprocessing'])
                np.testing.assert_array_equal(arrays['image'],no_targets['image']);assert not meta['augmentation_active']
                image=arrays['image'][0];mask=arrays['region'][0].astype(bool)
                interior,signal,cutoff=control_domain(raw)
                controls_here,s=controls(mask,interior,signal,row['sample_id'])
                identity=dict(sample_id=str(row['sample_id']),PID=str(row['PID']),fold=fold,true_label=int(row['original_label'])-1,true_class=row['class_name'])
                sampling.append(dict(**identity,**s,raw_signal_cutoff=cutoff,head_interior_pixels=int(interior.sum()),tumor_signal_fraction=float(signal[mask].mean()),tumor_mean_normalized_intensity=float(image[mask].mean())))
                locations.extend([dict(**identity,control_index=j,**r,tumor_pixels=s['tumor_pixels']) for j,r in enumerate(controls_here)])
                np.savez_compressed(out/f'cache/{row["sample_id"]}.npz',image=image,mask=mask,interior=interior,signal=signal,
                    translations=np.array([[r['dy'],r['dx']] for r in controls_here],dtype=np.int32).reshape(-1,2))
                images.append(arrays['image']);prepared.append(identity)
            with torch.inference_mode():logits=model(torch.from_numpy(np.stack(images)).cuda());prob=logits.softmax(1).cpu().numpy();logits=logits.cpu().numpy()
            maxdiff=max(maxdiff,compare(prob,rows,frozen))
            for identity,z in zip(prepared,logits):
                pred=int(z.argmax());originals.append(dict(**identity,original_correct=int(pred==identity['true_label']),**response(z,identity['true_label'],pred)))
        assert state_digest(model)==before
        repro.append(dict(fold=fold,slices=len(frame),patients=frame.PID.nunique(),class_mismatches=0,max_probability_error=maxdiff))
        log(out,f'Fold {fold}: original predictions reproduced; {len(frame)} masks and deterministic control sets prepared; max probability error {maxdiff:.3g}.')
        del model;torch.cuda.empty_cache()
    sampling=pd.DataFrame(sampling);originals=pd.DataFrame(originals);locations=pd.DataFrame(locations)
    assert len(sampling)==3064 and sampling.sample_id.is_unique and originals.PID.nunique()==233
    save(sampling,out/'tables/occlusion_sampling_audit.csv');save(locations,out/'tables/control_locations.csv')
    save(originals,out/'tables/original_predictions.csv');save(pd.DataFrame(inventory),out/'tables/checkpoint_audit.csv');save(pd.DataFrame(repro),out/'tables/prediction_reproducibility.csv')
    log(out,f'All original predictions passed. Controls: {len(locations)}; shortfall slices: {int(sampling.controls_shortfall.sum())}; zero-control slices: {int((sampling.n_controls==0).sum())}.')
    control_rows=[];slice_rows=[]
    originals=originals.set_index('sample_id');sampling=sampling.set_index('sample_id')
    for fold in range(1,6):
        model,_=checkpoint(root,fold,info);model.cuda();before=state_digest(model)
        part=originals[originals.fold==fold]
        for position,(sid,original) in enumerate(part.iterrows()):
            data=np.load(out/f'cache/{sid}.npz',allow_pickle=False);image=data['image'];mask=data['mask'];trans=data['translations']
            regions=[mask]+[translate(mask,*v) for v in trans]
            values=[local_value(image,r,mask) for r in regions]
            identity={k:original[k] for k in ['PID','fold','true_label','true_class','original_correct']};identity['sample_id']=sid
            for strategy in ['zero','local_median']:
                repl=[0.]*len(regions) if strategy=='zero' else [v[0] for v in values]
                stack=np.stack([perturb(image,r,v)[None] for r,v in zip(regions,repl)])
                with torch.inference_mode():z=model(torch.from_numpy(stack).cuda()).cpu().numpy()
                responses=[response(t,int(original.true_label),int(original.predicted_label)) for t in z]
                drops=[differences(original,r) for r in responses]
                tumor=responses[0];td=drops[0];controls_list=[]
                for j,(r,d) in enumerate(zip(responses[1:],drops[1:])):
                    c=regions[j+1];delta=image[c]-repl[j+1]
                    item=dict(**identity,strategy=strategy,control_index=j,dy=int(trans[j,0]),dx=int(trans[j,1]),pixels=int(c.sum()),
                        replacement_value=repl[j+1],local_ring_pixels=values[j+1][1],mean_absolute_intensity_change=float(np.abs(delta).mean()),
                        rms_intensity_change=float(np.sqrt(np.square(delta).mean())),**r,**d)
                    control_rows.append(item);controls_list.append(item)
                row=dict(**identity,strategy=strategy,n_controls=len(controls_list),paired_available=bool(controls_list),
                    tumor_pixels=int(mask.sum()),tumor_area_fraction=float(mask.mean()),tumor_centroid_y=sampling.loc[sid,'tumor_centroid_y'],tumor_centroid_x=sampling.loc[sid,'tumor_centroid_x'],
                    tumor_replacement_value=repl[0],tumor_local_ring_pixels=values[0][1],
                    tumor_mean_absolute_intensity_change=float(np.abs(image[mask]-repl[0]).mean()),
                    **{f'original_{k}':original[k] for k in ['predicted_label','confidence','true_logit','original_predicted_logit','true_probability','margin']+[f'logit_{k}' for k in range(3)]+[f'prob_{k}' for k in range(3)]},
                    **{f'tumor_{k}':v for k,v in tumor.items()},**{f'tumor_{k}':v for k,v in td.items()})
                for metric in ['true_logit_drop','true_probability_drop','original_predicted_logit_drop','margin_drop','flip','correct','mean_absolute_intensity_change']:
                    row[f'control_{metric}']=float(np.mean([r[metric] for r in controls_list])) if controls_list else 'unavailable_no_controls'
                row['control_median_true_logit_drop']=float(np.median([r['true_logit_drop'] for r in controls_list])) if controls_list else 'unavailable_no_controls'
                row['control_sd_true_logit_drop']=float(np.std([r['true_logit_drop'] for r in controls_list],ddof=0)) if controls_list else 'unavailable_no_controls'
                for metric in ['true_logit_drop','true_probability_drop','original_predicted_logit_drop','margin_drop','flip']:
                    row['delta_'+metric]=row['tumor_'+metric]-row['control_'+metric] if controls_list else 'unavailable_no_controls'
                slice_rows.append(row)
            if position%100==0:log(out,f'Perturbations fold {fold}: {position+1}/{len(part)} slices, both fixed replacement strategies.')
        assert state_digest(model)==before and all(p.grad is None for p in model.parameters())
        del model;torch.cuda.empty_cache();log(out,f'Fold {fold} perturbations complete; weights and buffers unchanged.')
    controls_df=pd.DataFrame(control_rows);slices_df=pd.DataFrame(slice_rows)
    save(controls_df,out/'tables/occlusion_per_control.csv');save(slices_df,out/'tables/occlusion_per_slice.csv')
    seal_names=['occlusion_per_control.csv','occlusion_per_slice.csv','original_predictions.csv','occlusion_sampling_audit.csv',
                'prediction_reproducibility.csv','control_locations.csv','checkpoint_audit.csv']
    seal_paths=[out/'tables'/n for n in seal_names]+list((out/'cache').glob('*.npz'))+[out/'metadata/protected_sha256.json']
    dump(inference_receipt,dict(status='complete',settings=SETTINGS,source_hashes=code,
        outputs={p.relative_to(root).as_posix():sha(p) for p in seal_paths}))
    return finish_analysis(root,out,controls_df,slices_df,originals.reset_index(),sampling.reset_index(),protected,code,repro,locations,start_time)

def finish_analysis(root,out,controls_df,slices_df,originals,sampling,protected,code,repro,locations,start_time):
    receipt_path=out/'metadata/completion_receipt.json'
    from .occlusion_reporting import analyze
    analyze(root,out,slices_df,controls_df,originals,sampling)
    verify_manifest(root,protected)
    for df in [controls_df,slices_df]:assert np.isfinite(df.select_dtypes('number').to_numpy()).all()
    audit=dict(status='PASS',slices=3064,patients=233,paired_slices=int((sampling.n_controls>0).sum()),
        paired_patients=int(sampling[sampling.n_controls>0].PID.nunique()),checkpoints=5,parameters=575359,original_class_mismatches=0,
        maximum_probability_error=max(r['max_probability_error'] for r in repro),control_locations=len(locations),control_inferences=len(controls_df),
        tumor_inferences=len(slices_df),zero_control_slices=int((sampling.n_controls==0).sum()),shortfall_slices=int(sampling.controls_shortfall.sum()),
        strict_load=True,no_patient_leakage=True,MRI_only=True,post_normalization_occlusion=True,no_renormalization=True,
        exact_area_and_shape_match=True,no_tumor_control_overlap=True,control_constraints_verified=True,
        control_domain_is_validated_brain_mask=False,patient_bootstrap=True,bootstrap_iterations=10000,seed=42,
        optimizer_steps=0,training_runs=0,weights_and_buffers_unchanged=True,previous_phases_unchanged=True,protected_files=len(protected),
        all_numeric_results_finite=True,unavailable_results_explicit=True,formal_pvalues_generated=False,phase16_started=False,
        elapsed_current_execution_seconds=time.perf_counter()-start_time,torch=torch.__version__,GPU=torch.cuda.get_device_name(0))
    dump(out/'final_scientific_audit.json',audit)
    (out/'PHASE15_1_OCCLUSION_REPORT.md').write_text((out/'PHASE15_1_OCCLUSION_REPORT.md').read_text(encoding='utf-8')+'\n\n## Scientific audit\n\n```json\n'+json.dumps(audit,indent=2)+'\n```',encoding='utf-8')
    outputs={p.relative_to(root).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['completion_receipt.json','progress.log','notebook_execution_audit.json']}
    dump(receipt_path,dict(status='complete',source_hashes=code,protected=protected,outputs=outputs))
    log(out,'Phase 15.1 complete. Scientific audit PASS. STOP before Phase 16.');return out

