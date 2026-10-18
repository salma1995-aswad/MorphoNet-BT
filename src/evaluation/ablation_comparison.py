"""Pure aggregation utilities and user-triggered Phase 13 reporting."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.ablation import VARIANTS,build_ablation
from src.ablation.protocol import check_protection,file_hash
from .protocol import CLASSES,PROBS,require,save_json,validate_predictions
from .analysis import METRICS,metrics,aggregate_patients
from .ablation_protocol import read,write,VARIANT_IDS,reference
from .ablation_analysis import save_analysis
from .ablation_figures import save_figures

ALL=(*VARIANT_IDS,'A5')
TRANSITIONS=['MSS','Morphology supervision','BAA','Dual Path with mean fusion','AMGF']

def audit_oof(context,name,frames):
    require(set(frames)==set(range(1,6)),'Five folds required for OOF')
    for fold,frame in frames.items():
        require(frame.variant.eq(name).all(),'Variant mismatch')
        validate_predictions(frame,context['splits'][fold]['test'],fold)
    oof=pd.concat(frames.values(),ignore_index=True);expected=context['manifest']
    audit=dict(predictions=len(oof),unique_samples=int(oof.sample_id.nunique()),unique_patients=int(oof.PID.nunique()),
        duplicate_samples=int(oof.sample_id.duplicated().sum()),missing_samples=len(set(expected.sample_id)-set(oof.sample_id)),
        unexpected_samples=len(set(oof.sample_id)-set(expected.sample_id)),patients_multiple_folds=int((oof.groupby('PID').fold.nunique()!=1).sum()))
    require(list(audit.values())==[3064,3064,233,0,0,0,0],f'OOF integrity failed: {name}: {audit}')
    require(oof.groupby('PID').true_label.nunique().eq(1).all(),'Patient labels inconsistent')
    require(oof.true_label.value_counts().sort_index().tolist()==expected.original_label.value_counts().sort_index().tolist(),'Class support mismatch')
    return oof,audit

def align(frames,key):
    base=frames['A0'].set_index(key).sort_index()
    identity=['PID','fold','true_label','true_class','file_path'] if key=='sample_id' else ['fold','true_label','true_class','n_slices']
    aligned=base[identity].copy()
    for name in ALL:
        frame=frames[name].set_index(key).sort_index()
        require(frame.index.is_unique and frame.index.equals(base.index),'Paired sample/PID alignment mismatch')
        pd.testing.assert_frame_equal(frame[identity],base[identity],check_exact=True,check_dtype=False)
        for col in ['predicted_label','predicted_class',*PROBS]:aligned[name+'__'+col]=frame[col]
    return aligned.reset_index()

def shared_bootstrap(consistencies,iterations=2000,seed=42):
    base=consistencies['A0'].sort_values('PID');draws=np.random.default_rng(seed).integers(0,len(base),size=(iterations,len(base)))
    rows=[];replicates={}
    for name in ALL:
        c=consistencies[name].sort_values('PID')
        require(c.PID.tolist()==base.PID.tolist() and np.array_equal(c.n_slices,base.n_slices),'Bootstrap clusters misaligned')
        values=c.n_correct.to_numpy()[draws].sum(1)/c.n_slices.to_numpy()[draws].sum(1)
        lo,hi=np.percentile(values,[2.5,97.5]);replicates[name]=values
        rows.append(dict(Variant=name,Accuracy=c.n_correct.sum()/c.n_slices.sum(),Lower95=lo,Upper95=hi))
    return pd.DataFrame(rows),pd.DataFrame(replicates),draws,base.PID.to_numpy(dtype=str)

def sequential_rows(slice_summary,patient_summary):
    rows=[]
    for earlier,later,component in zip(ALL[:-1],ALL[1:],TRANSITIONS):
        row=dict(Earlier=earlier,Later=later,Component=component)
        for level,values in [('OOF',slice_summary),('Patient',patient_summary)]:
            for metric in ['Accuracy','Macro F1','Macro ROC-AUC']:row[f'{level} {metric} difference (pp)']=100*(values[later][metric]-values[earlier][metric])
        rows.append(row)
    return pd.DataFrame(rows)

def report(context,frames):
    """Called only after the manual launcher obtains all 25 verified prediction sets."""
    require(set(frames)==set(VARIANT_IDS),'All five new variants required')
    root=context['root'];out=context['output'];oofs={};audits={};results={}
    for name in VARIANT_IDS:oofs[name],audits[name]=audit_oof(context,name,frames[name])
    a5,a5patients=reference(context);a5=a5.assign(variant='A5')
    a5frames={f:a5[a5.fold.eq(f)].copy() for f in range(1,6)}
    oofs['A5'],audits['A5']=audit_oof(context,'A5',a5frames)
    # Every coverage audit must pass before generating any comparison table.
    for name in ALL:
        directory='A5_reference' if name=='A5' else name
        result=save_analysis(context,directory,oofs[name],a5frames if name=='A5' else frames[name]);results[name]=result
        write(oofs[name],out/directory/'oof_predictions.csv')
        write(result['tables']['patient_level_predictions'],out/directory/'patient_predictions.csv')
        save_json(out/directory/'oof_integrity_audit.json',audits[name])
        matrix=result['slice_matrix'];matrix.div(matrix.sum(axis=1),axis=0).to_csv(out/directory/'normalized_confusion_matrix.csv')
        save_figures(root,directory,oofs[name],result)
        for level,frame in [('slice',oofs[name]),('patient',result['tables']['patient_level_predictions'])]:
            error=frame[frame.true_label.ne(frame.predicted_label)]
            counts=error.groupby('true_class').size().reindex(CLASSES,fill_value=0).rename('Errors').reset_index()
            write(counts,out/directory/f'{level}_per_class_errors.csv')
    patients={n:r['tables']['patient_level_predictions'] for n,r in results.items()}
    pd.testing.assert_frame_equal(patients['A5'].set_index('PID').sort_index()[a5patients.set_index('PID').columns],a5patients.set_index('PID').sort_index(),check_dtype=False,rtol=0,atol=1e-15)
    # A5 per-fold Test values are read from Phase 7 and cross-checked, never from validation.
    stored=read(root/'results/main/five_fold_test_results.csv')
    require(np.allclose(stored[METRICS],results['A5']['tables']['five_fold_test_results'][METRICS],rtol=0,atol=1e-14),'A5 frozen fold metrics mismatch')
    results['A5']['tables']['five_fold_test_results']=stored
    slice_summary={n:r['slice_summary'] for n,r in results.items()};patient_summary={n:r['patient_summary'] for n,r in results.items()}
    primary=[];expanded=[];patient_rows=[];efficiency=[];classes=[];patient_classes=[];means=[];folds=[];fold_classes=[];mistakes=[]
    for name in ALL:
        c=context['complexity'].loc[name];r=results[name]
        flags=dict(zip(['MSS','Morphology','BAA','Dual Path','AMGF'],VARIANTS[name].values()))
        common=dict(Variant=name,**flags,**{'Params (M)':c['Parameters (M)'],'MACs (G)':c['Inference MACs']/1e9})
        primary.append({**common,**{'OOF '+m:slice_summary[name][m] for m in ['Accuracy','Macro F1','Macro ROC-AUC']}})
        expanded.append({**common,**{'OOF '+m:slice_summary[name][m] for m in METRICS}})
        patient_rows.append(dict(Variant=name,**{'Patient '+m:patient_summary[name][m] for m in METRICS}))
        # Analytical parameter inventory; does not reprofile or modify a model.
        model=build_ablation(name,context['model_config'])
        excluded=[]
        for key,p in model.named_parameters():
            if name=='A2' and key.startswith('morphology_head.') or name in ('A3','A4','A5') and key.startswith(('morphology_head.region_predictor.','morphology_head.distance_predictor.')):excluded.append(p.numel())
        inference_parameters=int(c.Parameters)-sum(excluded);del model
        efficiency.append(dict(Variant=name,TotalTrainingParameters=int(c.Parameters),InferenceRequiredParameters=inference_parameters,
            **{'Params (M)':c['Parameters (M)'],'MACs (G)':c['Inference MACs']/1e9,'Inference FLOPs':int(c['Inference FLOPs']),
            'Total parameter FP32 bytes':4*int(c.Parameters),'Inference-required parameter FP32 bytes':4*inference_parameters,
            'OOF Accuracy':slice_summary[name]['Accuracy'],'OOF Macro F1':slice_summary[name]['Macro F1'],'Patient Macro F1':patient_summary[name]['Macro F1']}))
        classes.append(r['tables']['oof_per_class_metrics'].assign(Variant=name));patient_classes.append(r['tables']['patient_level_per_class_metrics'].assign(Variant=name))
        means.append(r['tables']['final_performance_summary'].assign(Variant=name));folds.append(r['tables']['five_fold_test_results'].assign(Variant=name))
        for fold in range(1,6):fold_classes.append(metrics((a5frames if name=='A5' else frames[name])[fold])[1].assign(Variant=name,Fold=fold))
        mistakes.append(dict(Variant=name,MisclassifiedSlices=int(oofs[name].true_label.ne(oofs[name].predicted_label).sum()),MisclassifiedPatients=int(patients[name].true_label.ne(patients[name].predicted_label).sum())))
    tables=dict(final_ablation_comparison=pd.DataFrame(primary),expanded_ablation_comparison=pd.DataFrame(expanded),patient_level_ablation_comparison=pd.DataFrame(patient_rows),
        accuracy_efficiency=pd.DataFrame(efficiency),per_class_metrics=pd.concat(classes,ignore_index=True),patient_per_class_metrics=pd.concat(patient_classes,ignore_index=True),
        five_fold_mean_sd=pd.concat(means,ignore_index=True),per_fold_comparison=pd.concat(folds,ignore_index=True),per_fold_class_metrics=pd.concat(fold_classes,ignore_index=True),misclassification_counts=pd.DataFrame(mistakes),
        sequential_component_contributions=sequential_rows(slice_summary,patient_summary))
    tables['meningioma_comparison']=tables['per_class_metrics'].query("Class=='Meningioma'").reset_index(drop=True)
    tables['meningioma_by_fold']=tables['per_fold_class_metrics'].query("Class=='Meningioma'")[['Variant','Fold','Sensitivity']]
    fold2=[];fold_differences=[]
    for name in ALL:
        row=tables['per_fold_comparison'].query('Variant==@name and Fold==2').iloc[0]
        per=tables['per_fold_class_metrics'].query('Variant==@name and Fold==2').set_index('Class')
        fold2.append(dict(Variant=name,Accuracy=row.Accuracy,**{'Macro F1':row['Macro F1'],**{c+' Recall':per.loc[c,'Sensitivity'] for c in CLASSES}}))
    for earlier,later in zip(ALL[:-1],ALL[1:]):
        for fold in range(1,6):
            pair=tables['per_fold_comparison'].query('Fold==@fold').set_index('Variant')
            fold_differences.append(dict(Earlier=earlier,Later=later,Fold=fold,**{m+' difference (pp)':100*(pair.loc[later,m]-pair.loc[earlier,m]) for m in ['Accuracy','Macro F1','Macro ROC-AUC']}))
    tables['fold_2_comparison']=pd.DataFrame(fold2);tables['per_fold_sequential_differences']=pd.DataFrame(fold_differences)
    overlaps=[]
    for level,collection,key in [('slice',oofs,'sample_id'),('patient',patients,'PID')]:
        aligned=align(collection,key);write(aligned,out/f'statistical_preparation/aligned_{level}_predictions.csv')
        for earlier,later in zip(ALL[:-1],ALL[1:]):
            a=aligned.true_label.eq(aligned[earlier+'__predicted_label']);b=aligned.true_label.eq(aligned[later+'__predicted_label'])
            overlaps.append(dict(Level=level,Earlier=earlier,Later=later,BothCorrect=int((a&b).sum()),EarlierCorrectOnly=int((a&~b).sum()),LaterCorrectOnly=int((~a&b).sum()),BothWrong=int((~a&~b).sum())))
    tables['paired_error_overlap']=pd.DataFrame(overlaps)
    ci,replicates,draws,pids=shared_bootstrap({n:r['tables']['patient_consistency'] for n,r in results.items()})
    tables['accuracy_cluster_bootstrap_ci']=ci
    original_ci=read(root/'results/main/accuracy_cluster_bootstrap_ci.csv').iloc[0]
    require(np.allclose(ci.set_index('Variant').loc['A5',['Lower95','Upper95']].to_numpy(dtype=float),[original_ci.Lower95,original_ci.Upper95],rtol=0,atol=1e-15),'A5 bootstrap mismatch')
    write(replicates.assign(Iteration=np.arange(1,2001)),out/'statistical_preparation/bootstrap_replicates.csv')
    np.savez_compressed(out/'statistical_preparation/shared_bootstrap_draws.npz',indices=draws,PID=pids)
    for name,frame in tables.items():write(frame,out/(name+'.csv'))
    for name in ['final_ablation_comparison','patient_level_ablation_comparison']:
        df=tables[name];scorecols=[c for c in df if c.startswith(('OOF ','Patient '))]
        style=df.style.format(precision=4).highlight_max(subset=scorecols,color='#dcebdc')
        (out/(name+'.html')).write_text(style.to_html(),encoding='utf-8')
        (out/(name+'.tex')).write_text(df.to_latex(index=False,float_format='%.4f'),encoding='utf-8')
    leaders=[]
    for name in ['final_ablation_comparison','patient_level_ablation_comparison']:
        for col in tables[name]:
            if col.startswith(('OOF ','Patient ')):
                best=tables[name][col].max(skipna=False)
                leaders.append(dict(Table=name,Metric=col,BestValue=best,Variants='; '.join(tables[name].loc[tables[name][col].eq(best),'Variant'])))
    write(pd.DataFrame(leaders),out/'comparison/metric_leaders.csv')
    figures(out,tables)
    check_protection(root,context['protected'])
    receipts=[]
    for name in VARIANT_IDS:
        for fold in range(1,6):
            folder=out/name/f'fold_{fold}';receipt=json.loads((folder/'evaluation_receipt.json').read_text())
            require(receipt['inference_passes']==1 and receipt['predictions_sha256']==file_hash(folder/'predictions.csv'),'Final receipt audit failed');receipts.append(receipt)
    from datetime import datetime,timezone
    context['metadata'].update(status='complete',evaluation_date=datetime.now(timezone.utc).isoformat(),oof_integrity_results=audits,A5_inference_passes=0)
    save_json(out/'ablation_evaluation_metadata.json',context['metadata'])
    audit=dict(best_checkpoints_evaluated_or_verified_reused=len(receipts),last_checkpoints_used=0,retraining=0,optimizer_steps=0,architecture_changes=0,
        MRI_only=True,test_augmentation=False,all_OOF_audits_passed=True,patient_aggregation=True,per_class_metrics=True,fold_sample_SD_ddof=1,
        figures_generated=True,shared_patient_bootstrap=True,paired_alignment=True,sequential_differences_percentage_points=True,
        statistical_significance_tests=0,protected_artifacts_unchanged=True,A5_frozen_reference_verified=True)
    save_json(out/'final_scientific_audit.json',audit)
    (out/'conclusions.md').write_text('Observed results: see the numerical tables and signed sequential differences. Negative changes are retained.\n\nSupported interpretation: conditional cumulative differences under the frozen protocol; not independent component effects or interactions.\n\nUnresolved: statistical reliability and external-institution generalization. Bootstrap intervals condition on fixed predictions and omit retraining variability. No significance, clinical-validation or external-generalization claim is made. No model was changed after Test evaluation.\n',encoding='utf-8')
    return tables

def figures(out,tables):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=400,bbox_inches='tight')
        plt.close(fig)
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        for key,col,label in [('final_ablation_comparison','OOF Macro F1','OOF macro F1'),('final_ablation_comparison','OOF Accuracy','OOF accuracy'),
            ('patient_level_ablation_comparison','Patient Macro F1','Patient macro F1'),('accuracy_efficiency','Params (M)','Total training parameters (millions)'),
            ('accuracy_efficiency','MACs (G)','Inference Conv/Linear MACs (billions)'),('meningioma_comparison','Sensitivity','Meningioma sensitivity')]:
            df=tables[key];fig,ax=plt.subplots(figsize=(6,4),layout='constrained');bars=ax.bar(df.Variant,df[col],color='#0072B2')
            ax.set_ylabel(label);ax.set_ylim(0,1.08 if col not in ('Params (M)','MACs (G)') else df[col].max()*1.2)
            ax.bar_label(bars,fmt='%.4f',padding=3);save(fig,col.replace(' ','_').replace('(','').replace(')',''))
        df=tables['sequential_component_contributions'];values=df['OOF Macro F1 difference (pp)']
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.bar(df.Earlier+' to '+df.Later,values,color=['#0072B2' if v>=0 else '#D55E00' for v in values]);ax.axhline(0,color='black',lw=.8)
        lim=max(float(values.abs().max())*1.3,.1);ax.set_ylim(-lim,lim);ax.set_ylabel('OOF macro F1 difference (percentage points)');save(fig,'sequential_macro_f1')
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for name,g in tables['per_fold_comparison'].groupby('Variant',sort=True):ax.plot(g.Fold,g['Macro F1'],'o-',label=name)
        ax.set(xticks=range(1,6),ylim=(0,1.05),xlabel='Outer Test fold',ylabel='Macro F1');ax.legend(ncol=3,frameon=False);save(fig,'fold_macro_f1')
