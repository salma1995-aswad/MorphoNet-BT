"""Phase 13 preparation tests: synthetic metrics and hard no-inference assertions."""
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from . import ablation_protocol as protocol
from .ablation_comparison import align,shared_bootstrap,sequential_rows,ALL
from .analysis import metrics,aggregate_patients
from .protocol import CLASSES,PROBS,require,save_json

def synthetic_checks():
    labels=np.repeat(np.arange(233)%3,3);pred=labels.copy();pred[::5]=(pred[::5]+1)%3
    probabilities=np.full((len(labels),3),.1);probabilities[np.arange(len(labels)),pred]=.8
    data=pd.DataFrame(dict(PID=np.repeat([f'synthetic-{i:03d}' for i in range(233)],3),sample_id=[f'synthetic-slice-{i}' for i in range(len(labels))],
        true_label=labels,predicted_label=pred,fold=np.repeat(np.arange(233)%5+1,3),file_path='synthetic://not-an-image'))
    data[PROBS]=probabilities
    for prefix in ['true','predicted']:data[prefix+'_class']=[CLASSES[x] for x in data[prefix+'_label']]
    values,classes,matrix=metrics(data)
    require(np.isclose(values['Accuracy'],accuracy_score(labels,pred)) and np.isclose(values['Macro F1'],f1_score(labels,pred,average='macro')),'Synthetic metric mismatch')
    require(np.isclose(values['Macro ROC-AUC'],np.mean([roc_auc_score(labels==i,probabilities[:,i]) for i in range(3)])),'Synthetic AUC mismatch')
    patients,consistency=aggregate_patients(data)
    for pid,g in data.groupby('PID'):
        actual=patients.set_index('PID').loc[pid,PROBS].to_numpy(dtype=float)
        require(np.allclose(actual,g[PROBS].mean().to_numpy()),'Synthetic patient aggregation mismatch')
    aligned=align({name:data.copy() for name in ALL},'sample_id');require(len(aligned)==len(data),'Synthetic alignment mismatch')
    broken={name:data.copy() for name in ALL};broken['A2'].loc[0,'PID']='wrong'
    rejected=False
    try:align(broken,'sample_id')
    except AssertionError:rejected=True
    require(rejected,'Misalignment accepted')
    ci,reps,draws,pids=shared_bootstrap({n:consistency.copy() for n in ALL})
    c=consistency.sort_values('PID')
    # Independent cluster-multiplicity formula for all 2,000 draws.
    multiplicities=np.array([np.bincount(d,minlength=233) for d in draws])
    expected=(multiplicities@c.n_correct.to_numpy())/(multiplicities@c.n_slices.to_numpy())
    require(np.array_equal(expected,reps.A0.to_numpy()) and all(np.array_equal(reps.A0,reps[n]) for n in ALL),'Paired cluster bootstrap mismatch')
    summary={n:{m:1-i*.01 for m in ['Accuracy','Macro F1','Macro ROC-AUC']} for i,n in enumerate(ALL)}
    delta=sequential_rows(summary,summary)
    require(np.allclose(delta.filter(like='(pp)').to_numpy(),-1),'Signed percentage-point difference mismatch')
    # Explicit undefined precision is NaN when a class has no predicted samples.
    missing=data.copy();missing.predicted_label=0
    require(np.isnan(metrics(missing)[1].loc[1,'Precision']),'Undefined metrics silently replaced')
    return dict(synthetic_only=True,metrics_match_sklearn=True,patient_mean_verified=True,alignment_verified=True,
        mismatched_ID_rejected=True,shared_cluster_bootstrap_verified=True,negative_percentage_points_preserved=True,undefined_metrics_explicit=True)

def safe_prepare(root):
    def forbidden(*a,**k):raise AssertionError('Test image access, forward or optimizer execution forbidden during preparation')
    real_load=torch.load
    loaded=[]
    def best_only(path,*a,**k):
        require(Path(path).name=='best_macro_f1.pt','Only best checkpoint deserialization is allowed')
        loaded.append(str(path));return real_load(path,*a,**k)
    with patch.object(protocol,'BrainTumorDataset',new=forbidden),patch.object(torch.nn.Module,'__call__',new=forbidden), \
         patch.object(torch.optim.AdamW,'step',new=forbidden),patch.object(torch,'load',new=best_only):
        try:protocol.launch(root,confirm_test_evaluation=False)
        except RuntimeError as exc:require('disabled' in str(exc),'Wrong guard error')
        else:raise AssertionError('Guard failed')
        try:protocol.evaluate_fold(None,'A0',1)
        except ValueError:pass
        else:raise AssertionError('Direct inference bypass accepted')
        context=protocol.prepare(root)
    checks=synthetic_checks()
    require(len(loaded)==25 and len(set(loaded))==25,'Preparation did not load exactly 25 distinct best checkpoints')
    from src.ablation.protocol import check_protection
    count=check_protection(context['root'],context['protected'])
    checks.update(best_checkpoints_strictly_loaded=25,completed_runs_verified=25,frozen_folds_verified=5,A5_reference_verified=True,
        protected_artifacts_unchanged=count,confirmation_default=False,new_held_out_evaluations=0,test_images_decoded=0,
        training_runs=0,optimizer_steps=0,manual_cell_executed=False,ready_for_manual_evaluation=True)
    save_json(context['output']/'preparation_audit.json',checks)
    return context,checks
