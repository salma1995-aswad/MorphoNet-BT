"""Phase 13 metadata-only preflight and explicitly manual held-out inference."""
import json
import os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.dataset import BrainTumorDataset
from src.data.kfold import reload_splits,load_manifest
from src.models.ablation import build_ablation,VARIANTS
from src.ablation.protocol import preflight as training_preflight,run_identity,require_prepared,file_hash,protected_inventory,check_protection
from src.ablation.training import validate_checkpoint
from .protocol import CLASSES,PROBS,VERSION,require,save_json,validate_predictions
from .analysis import aggregate_patients,metrics

VARIANT_IDS=('A0','A1','A2','A3','A4')
CODE_FILES=['src/evaluation/ablation_protocol.py','src/evaluation/ablation_analysis.py','src/evaluation/ablation_comparison.py','src/evaluation/ablation_figures.py',
 'src/evaluation/analysis.py','src/evaluation/protocol.py','src/models/ablation.py','src/models/morphonet_bt.py','src/models/blocks.py','src/models/attention.py','src/models/fusion.py',
 'src/data/dataset.py','src/data/preprocessing.py','src/data/morphology.py','src/data/inspection.py']
_MANUAL=object()

def read(path):return pd.read_csv(path,dtype={'PID':str,'sample_id':str},keep_default_na=False,float_precision='round_trip')
def write(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');frame.to_csv(temp,index=False,float_format='%.17g');temp.replace(path)

def guard(confirmation):
    if confirmation is not True:
        raise RuntimeError('Held-out Test evaluation is disabled. The user must manually set CONFIRM_TEST_EVALUATION = True.')

def reference(context):
    """Read frozen A5 predictions; no model construction or inference."""
    main=context['root']/'results/main';frame=read(main/'oof_predictions.csv')
    require(len(frame)==3064 and frame.sample_id.is_unique and frame.PID.nunique()==233,'A5 coverage mismatch')
    for fold in range(1,6):
        part=frame[frame.fold.eq(fold)];validate_predictions(part,context['splits'][fold]['test'],fold)
        saved=read(main/f'fold_{fold}/test_predictions.csv')
        receipt=json.loads((main/f'fold_{fold}/test_evaluation_receipt.json').read_text())
        require(file_hash(main/f'fold_{fold}/test_predictions.csv')==receipt['predictions_sha256'],'A5 reference receipt mismatch')
        a=part.set_index('sample_id').sort_index();b=saved.set_index('sample_id').sort_index()
        ids=[k for k in b.columns if k not in PROBS]
        pd.testing.assert_frame_equal(a[ids],b[ids],check_dtype=False,check_exact=True)
        require(np.allclose(a[PROBS],b[PROBS],rtol=0,atol=3e-16),'A5 reference probabilities mismatch')
    patients=read(main/'patient_level_predictions.csv');calculated,_=aggregate_patients(frame)
    pd.testing.assert_frame_equal(calculated.set_index('PID').sort_index()[patients.set_index('PID').columns],patients.set_index('PID').sort_index(),check_dtype=False,rtol=0,atol=1e-15)
    return frame,patients

def preflight(root):
    root=Path(root).resolve();info=training_preflight(root);require_prepared(info)
    splits=reload_splits(root/'splits');out=root/'results/ablation/evaluation'
    frozen=read(root/'results/ablation/setup/ablation_architecture_summary.csv').set_index('Variant')
    rows=[]
    for name in VARIANT_IDS:
        model=build_ablation(name,info['config']['model'])
        require(sum(p.numel() for p in model.parameters())==frozen.loc[name,'Parameters'],'Frozen parameter mismatch')
        for fold in range(1,6):
            folder=root/f'results/ablation/training/{name}/fold_{fold}'
            path=folder/'best_macro_f1.pt';marker=json.loads((folder/'training_complete.json').read_text())
            identity=run_identity(info,name,fold)
            require(marker['identity']==identity and marker['variant']==name and marker['fold']==fold,f'Completion identity mismatch: {name}/{fold}')
            require(marker['early_stopping'] is True and marker['test_evaluated'] is False,'Unexpected completion state')
            require(Path(marker['checkpoint_path']).resolve()==path,'Completion path mismatch')
            digest=file_hash(path);require(digest==marker['checkpoint_sha256']['best_macro_f1.pt'],'Best checkpoint changed')
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            validate_checkpoint(checkpoint,info,identity)
            require(checkpoint['epoch']==checkpoint['best_epoch']==marker['best_epoch'],'Best epoch mismatch')
            require(checkpoint['best_validation_macro_f1']==checkpoint['validation_metrics']['macro_f1']==marker['best_validation_macro_f1'],'Best selection mismatch')
            require(checkpoint['validation_metrics']['accuracy']==marker['best_validation_accuracy'],'Best validation accuracy mismatch')
            history=read(folder/'history.csv')
            require(len(history)==marker['total_epochs'] and int(history.loc[history.val_macro_f1.idxmax(),'epoch'])==marker['best_epoch'],'Completed history mismatch')
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            require(all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()),'Nonfinite checkpoint tensors')
            rows.append(dict(Variant=name,Fold=fold,Checkpoint=str(path),CheckpointSHA256=digest,BestEpoch=checkpoint['epoch'],
                BestValidationMacroF1=marker['best_validation_macro_f1'],TrainingConfigurationFingerprint=info['configuration_fingerprint'],
                TestSplitSHA256=info['split_hashes'][f'fold_{fold}_test.csv'],TestSlices=len(splits[fold]['test']),TestPatients=int(splits[fold]['test'].PID.nunique())))
            del checkpoint
        del model
    context=dict(root=root,output=out,config=info['preprocessing'],splits=splits,manifest=load_manifest(root/'results/dataset_statistics.csv'),
        checkpoints=pd.DataFrame(rows),complexity=frozen,model_config=info['config']['model'])
    context['reference']=reference(context)
    protected=protected_inventory(root)
    for path in (root/'results/ablation/training').rglob('*'):
        if path.is_file():protected[path.relative_to(root).as_posix()]=file_hash(path)
    manifest_path=out/'protected_artifacts.json'
    if manifest_path.exists():
        prior=json.loads(manifest_path.read_text());require(prior==protected,'Protected artifacts differ from original preparation; review required')
    context['protected']=protected
    import sklearn
    metadata=dict(status='prepared_not_evaluated',prepared_at=datetime.now(timezone.utc).isoformat(),evaluation_date=None,
        dataset=dict(slices=3064,patients=233,manifest_sha256=file_hash(root/'results/dataset_statistics.csv')),
        split_hashes=info['split_hashes'],preprocessing=info['preprocessing'],preprocessing_version=VERSION,class_mapping=dict(enumerate(CLASSES)),
        variants=VARIANTS,checkpoints=rows,complexity=frozen.reset_index().to_dict('records'),software={**info['runtime'],'pandas':pd.__version__,'sklearn':sklearn.__version__},
        seed=42,batch_size=16,precision='FP32; no autocast; TF32 disabled',num_workers=0,augmentation=False,MRI_only=True,
        bootstrap=dict(unit='patient',iterations=2000,seed=42,method='percentile',shared_draws=True,conditional_on_fixed_predictions=True),
        code_sha256={p:file_hash(root/p) for p in CODE_FILES},oof_integrity_results={},
        A5_reference={p:file_hash(root/p) for p in ['results/main/oof_predictions.csv','results/main/patient_level_predictions.csv','results/main/five_fold_test_results.csv']})
    context['metadata']=metadata
    return context

def prepare(root):
    """No Test Dataset construction, image decoding, inference or new test metrics."""
    context=preflight(root);out=context['output'];out.mkdir(parents=True,exist_ok=True)
    path=out/'protected_artifacts.json'
    if not path.exists():save_json(path,context['protected'])
    write(context['checkpoints'],out/'checkpoint_inventory.csv')
    save_json(out/'preparation_metadata.json',context['metadata'])
    if not(out/'ablation_evaluation_metadata.json').exists():save_json(out/'ablation_evaluation_metadata.json',context['metadata'])
    print('All 25 best checkpoints strictly loaded and verified. All frozen folds and A5 reference verified. Zero new Test evaluations.')
    return context

def receipt_identity(context,name,fold):
    row=context['checkpoints'].set_index(['Variant','Fold']).loc[(name,fold)]
    return json.loads(json.dumps(dict(variant=name,fold=fold,checkpoint_sha256=row.CheckpointSHA256,test_split_sha256=row.TestSplitSHA256,
        preprocessing=context['config'],class_names=CLASSES,code_sha256=context['metadata']['code_sha256'],
        version='phase13-v1',batch_size=16,precision='float32',seed=42,software=context['metadata']['software'])))

def cached_fold(context,name,fold,resume=True):
    folder=context['output']/name/f'fold_{fold}';prediction=folder/'predictions.csv';receipt=folder/'evaluation_receipt.json'
    if receipt.exists():
        require(resume,'Existing evaluation preserved. Enable RESUME_EVALUATION to reuse it.')
        saved=json.loads(receipt.read_text());require(saved['identity']==receipt_identity(context,name,fold) and saved['status']=='complete' and saved['inference_passes']==1,'Cached evaluation identity/version mismatch; manual review required')
        require(file_hash(prediction)==saved['predictions_sha256'],'Cached prediction file changed')
        frame=read(prediction);require(frame.variant.eq(name).all(),'Cached variant mismatch')
        validate_predictions(frame,context['splits'][fold]['test'],fold)
        return frame
    require(not folder.exists() or not any(folder.iterdir()),'Incomplete/unreceipted evaluation exists. Stop for explicit recovery review; do not repeat inference automatically.')
    return None

def evaluate_fold(context,name,fold,*,resume=True,permit=None):
    require(permit is _MANUAL,'Held-out inference requires the manual evaluation launcher')
    cached=cached_fold(context,name,fold,resume)
    if cached is not None:
        print(f'{name} fold {fold}: verified predictions reused; no repeated inference.');return cached
    row=context['checkpoints'].set_index(['Variant','Fold']).loc[(name,fold)]
    require(file_hash(row.Checkpoint)==row.CheckpointSHA256,'Checkpoint changed after preflight')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.manual_seed(42);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available(),'Phase 7-matched CUDA environment required')
    model=build_ablation(name,context['model_config'])
    checkpoint=torch.load(row.Checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True);del checkpoint
    model.to('cuda').eval()
    dataset=BrainTumorDataset(context['root']/f'splits/fold_{fold}_test.csv','test',config=context['config'],include_targets=False)
    loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,drop_last=False)
    folder=context['output']/name/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
    identity=receipt_identity(context,name,fold)
    # Exclusive marker prevents concurrent or automatic repeated evaluation.
    with (folder/'evaluation_started.json').open('x') as f:json.dump(dict(identity=identity,started_at=datetime.now(timezone.utc).isoformat()),f)
    probs=[];ids=[];pids=[];labels=[]
    with torch.inference_mode():
        for batch in loader:
            require(set(batch)=={'image','label','pid','sample_id'},'Morphology targets forbidden at inference')
            image=batch['image'].to('cuda');require(image.dtype==torch.float32 and tuple(image.shape[1:])==(1,256,256) and bool(torch.isfinite(image).all()),'Invalid MRI tensor')
            logits=model(image);require(tuple(logits.shape)==(len(image),3) and bool(torch.isfinite(logits).all()),'Invalid logits')
            probs.append(logits.softmax(1).cpu().numpy());ids.extend(batch['sample_id']);pids.extend(batch['pid']);labels.extend(batch['label'].tolist())
    probability=np.concatenate(probs);frame=pd.DataFrame(dict(variant=name,fold=fold,PID=pids,sample_id=ids,true_label=labels))
    frame['predicted_label']=probability.argmax(1);frame[PROBS]=probability
    frame['file_path']=dataset.frame.set_index('sample_id').loc[ids,'file_path'].to_numpy()
    for prefix in ['true','predicted']:frame[prefix+'_class']=[CLASSES[i] for i in frame[prefix+'_label']]
    validate_predictions(frame,dataset.frame,fold)
    write(frame,folder/'predictions.csv')
    save_json(folder/'evaluation_receipt.json',dict(identity=identity,status='complete',inference_passes=1,samples=len(frame),predictions_sha256=file_hash(folder/'predictions.csv'),completed_at=datetime.now(timezone.utc).isoformat()))
    del model;torch.cuda.empty_cache()
    print(f'{name} fold {fold}: one held-out pass completed; {len(frame)} slices.')
    return read(folder/'predictions.csv')

def launch(root,variants=VARIANT_IDS,folds=(1,2,3,4,5),*,confirm_test_evaluation=False,overwrite=False,resume_evaluation=True):
    guard(confirm_test_evaluation) # First operation; no metadata or Test access before confirmation.
    require(variants and len(set(variants))==len(variants) and all(v in VARIANT_IDS for v in variants),'Select unique A0-A4 variants; A5 is a frozen file reference')
    require(folds and len(set(folds))==len(folds) and all(type(f) is int and 1<=f<=5 for f in folds),'Select unique folds 1-5')
    require(overwrite is False,'Automatic overwrite/re-evaluation is prohibited; preserve receipts and request explicit recovery review')
    require(type(resume_evaluation) is bool,'RESUME_EVALUATION must be Boolean')
    context=preflight(root)
    prepared=json.loads((context['output']/'preparation_metadata.json').read_text())
    require(prepared['code_sha256']==context['metadata']['code_sha256'],'Evaluation code changed since preparation')
    # Check every selected cache before allowing the first image to be decoded.
    for name in sorted(variants):
        for fold in sorted(folds):cached_fold(context,name,fold,resume_evaluation)
    frames={name:{} for name in sorted(variants)}
    for name in sorted(variants):
        for fold in sorted(folds):frames[name][fold]=evaluate_fold(context,name,fold,resume=resume_evaluation,permit=_MANUAL)
    check_protection(context['root'],context['protected'])
    if set(variants)!=set(VARIANT_IDS) or set(folds)!=set(range(1,6)):
        print('Selected folds completed. Final six-variant reporting requires all five variants and five folds; verified receipts can be reused.')
        return None
    from .ablation_comparison import report
    return report(context,frames)
