"""OOF coverage gates, paired comparisons and shared patient bootstrap."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .protocol import CLASSES, PROBS, require, sha256, save_json, validate_predictions
from .analysis import METRICS, metrics, aggregate_patients
from .baseline_analysis import save_analysis
from .baseline_figures import save_figures
from src.models.baselines import BASELINES, DISPLAY_NAMES


def read(path):
    return pd.read_csv(path, dtype={'PID': str, 'sample_id': str}, keep_default_na=False, float_precision='round_trip')


def write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g')


def audit_oof(context, name, frames):
    require(set(frames) == set(range(1, 6)), 'Exactly five test folds required')
    for fold, frame in frames.items():
        require(frame.model.eq(name).all(), 'Prediction architecture mismatch')
        validate_predictions(frame, context['splits'][fold]['test'], fold)
    oof = pd.concat(frames.values(), ignore_index=True)
    expected = set(context['manifest'].sample_id)
    audit = dict(predictions=len(oof), unique_samples=int(oof.sample_id.nunique()), unique_patients=int(oof.PID.nunique()),
        duplicate_samples=int(oof.sample_id.duplicated().sum()), missing_samples=len(expected-set(oof.sample_id)),
        unexpected_samples=len(set(oof.sample_id)-expected), patients_in_multiple_folds=int((oof.groupby('PID').fold.nunique()!=1).sum()))
    require(list(audit.values()) == [3064,3064,233,0,0,0,0], f'OOF audit failed: {name}: {audit}')
    require(oof.groupby('PID').true_label.nunique().eq(1).all(), 'Patient truth mismatch')
    output = context['root'] / f'results/baselines/evaluation/{name}'
    write(oof, output/'oof_predictions.csv')
    save_json(output/'oof_integrity_audit.json', audit)
    return oof, audit


def analyze(context, all_frames):
    require(set(all_frames) == set(BASELINES), 'All five baselines required before final comparisons')
    oofs, audits, results = {}, {}, {}
    # Gate all comparative analysis on all five successful coverage audits.
    for name in BASELINES:
        oofs[name], audits[name] = audit_oof(context, name, all_frames[name])
    for name in BASELINES:
        results[name] = save_analysis(context, name, oofs[name], all_frames[name])
        matrix = results[name]['slice_matrix']
        matrix.div(matrix.sum(axis=1), axis=0).to_csv(context['root']/f'results/baselines/evaluation/{name}/normalized_confusion_matrix.csv')
        save_figures(context['root'], name, oofs[name], results[name])
    context['metadata']['oof_integrity_results'] = audits
    return oofs, results


def compare(context, oofs, results):
    root = context['root']; output = root/'results/baselines/evaluation'; main = root/'results/main'
    names = ['morphonet_bt', *BASELINES]
    display = {'morphonet_bt':'MorphoNet-BT', **DISPLAY_NAMES}
    reference = read(main/'oof_predictions.csv')
    require(len(reference)==3064 and reference.sample_id.is_unique, 'Reference OOF coverage mismatch')
    for fold in range(1,6):
        receipt = json.loads((main/f'fold_{fold}/test_evaluation_receipt.json').read_text())
        require(sha256(main/f'fold_{fold}/test_predictions.csv') == receipt['predictions_sha256'], 'Reference fold predictions changed')
        saved = read(main/f'fold_{fold}/test_predictions.csv')
        validate_predictions(saved, context['splits'][fold]['test'], fold)
        a=reference[reference.fold.eq(fold)].set_index('sample_id').sort_index()
        b=saved.set_index('sample_id').sort_index()
        identity_columns = [col for col in b.columns if col not in PROBS]
        pd.testing.assert_frame_equal(a[identity_columns], b[identity_columns], check_dtype=False, check_exact=True)
        # Phase 7 OOF was reserialized after pandas' default CSV parser.
        # Enforce exact identities/classes and only sub-femtoscale probability tolerance.
        require(np.allclose(a[PROBS], b[PROBS], atol=3e-16, rtol=0), 'Reference probabilities changed')
    oofs = {'morphonet_bt':reference, **oofs}
    patients = {'morphonet_bt':read(main/'patient_level_predictions.csv'), **{n:r['tables']['patient_level_predictions'] for n,r in results.items()}}
    reference_summary = read(main/'final_performance_summary.csv').set_index('Metric').AggregatedOOFValue.to_dict()
    reference_patient = read(main/'patient_level_metrics.csv').set_index('Metric').Value.to_dict()
    slice_summaries = {'morphonet_bt':reference_summary, **{n:r['slice_summary'] for n,r in results.items()}}
    patient_summaries = {'morphonet_bt':reference_patient, **{n:r['patient_summary'] for n,r in results.items()}}
    for n in names:
        reconstructed, _ = aggregate_patients(oofs[n])
        pd.testing.assert_frame_equal(reconstructed.set_index('PID').sort_index()[patients[n].set_index('PID').columns], patients[n].set_index('PID').sort_index(), check_dtype=False, atol=1e-15, rtol=0)
        for frame, stored in [(oofs[n],slice_summaries[n]), (patients[n],patient_summaries[n])]:
            computed=metrics(frame)[0]
            require(all(np.isclose(computed[m],stored[m],atol=1e-14,rtol=0) for m in METRICS), 'Stored metric mismatch')
    complexity=read(root/'results/baselines/setup/baseline_architecture_summary.csv').set_index('Model')
    phase5=json.loads((root/'results/complexity/complexity_summary.json').read_text())
    require(complexity.loc['MorphoNet-BT','Parameters']==phase5['total_parameters'], 'Phase 5/8 parameter mismatch')
    require(complexity.loc['MorphoNet-BT','ConvLinearMACs']==phase5['profiles']['inference']['thop_conv_linear_macs'], 'Phase 5/8 MAC mismatch')
    primary=[]; patient_rows=[]; efficiency=[]; classes=[]; patient_classes=[]; fold2=[]; means=[]; mistakes=[]
    for n in names:
        label=display[n]; c=complexity.loc[label]; folder=main if n=='morphonet_bt' else output/n
        primary.append({'Model':label,'Parameters (M)':c.ParametersM,'MACs (G)':c.ConvLinearMACs/1e9,**{'OOF '+k:v for k,v in slice_summaries[n].items()}})
        patient_rows.append({'Model':label,**{'Patient '+k:v for k,v in patient_summaries[n].items()}})
        efficiency.append(dict(Model=label, Parameters=int(c.Parameters), **{'Parameters (M)':c.ParametersM,
            'Parameter ratio relative to MorphoNet-BT':c.Parameters/phase5['total_parameters'], 'MACs':int(c.ConvLinearMACs),
            'MAC ratio relative to MorphoNet-BT':c.ConvLinearMACs/complexity.loc['MorphoNet-BT','ConvLinearMACs'],
            'Conv/Linear FLOPs':int(c.ConvLinearFLOPs),'FP32 storage bytes':int(c.FP32Bytes),'FP32 storage MB':c.FP32MB,
            'OOF Accuracy':slice_summaries[n]['Accuracy'],'OOF Macro F1':slice_summaries[n]['Macro F1'],
            'MorphoNet parameter reduction (%)':(1-phase5['total_parameters']/c.Parameters)*100}))
        classes.append(read(folder/'oof_per_class_metrics.csv').assign(Model=label))
        patient_classes.append(read(folder/'patient_level_per_class_metrics.csv').assign(Model=label))
        fold=read(folder/'five_fold_test_results.csv').set_index('Fold').loc[2]
        fc=read(folder/'fold_2/test_per_class_metrics.csv').set_index('Class')
        fold2.append({'Model':label,'Accuracy':fold.Accuracy,'Macro F1':fold['Macro F1'], **{k+' Recall':fc.loc[k,'Sensitivity'] for k in CLASSES}})
        means.append(read(folder/'final_performance_summary.csv').assign(Model=label))
        mistakes.append(dict(Model=label, MisclassifiedSlices=int(oofs[n].true_label.ne(oofs[n].predicted_label).sum()), MisclassifiedPatients=int(patients[n].true_label.ne(patients[n].predicted_label).sum())))
    tables={'final_model_comparison':pd.DataFrame(primary), 'patient_level_model_comparison':pd.DataFrame(patient_rows),
        'efficiency_comparison':pd.DataFrame(efficiency), 'per_class_comparison':pd.concat(classes,ignore_index=True),
        'patient_per_class_comparison':pd.concat(patient_classes,ignore_index=True), 'fold_2_comparison':pd.DataFrame(fold2),
        'five_fold_mean_sd':pd.concat(means,ignore_index=True), 'misclassification_counts':pd.DataFrame(mistakes)}
    tables['meningioma_comparison']=tables['per_class_comparison'].query("Class == 'Meningioma'").reset_index(drop=True)
    overlaps=[]
    for level, frames, key, identitycols in [('slice',oofs,'sample_id',['PID','fold','true_label','true_class','file_path']),('patient',patients,'PID',['fold','true_label','true_class','n_slices'])]:
        base=frames['morphonet_bt'].set_index(key).sort_index()
        aligned=base[identitycols].copy()
        for n in names:
            frame=frames[n].set_index(key).sort_index()
            require(frame.index.is_unique and frame.index.equals(base.index), 'Paired identity mismatch')
            pd.testing.assert_frame_equal(frame[identitycols],base[identitycols],check_dtype=False,check_exact=True)
            for col in ['predicted_label','predicted_class',*PROBS]: aligned[n+'__'+col]=frame[col]
            if n!='morphonet_bt':
                a=base.true_label.eq(base.predicted_label); b=frame.true_label.eq(frame.predicted_label)
                overlaps.append(dict(Level=level,Model=display[n],BothCorrect=int((a&b).sum()),MorphoNetCorrectBaselineWrong=int((a&~b).sum()),MorphoNetWrongBaselineCorrect=int((~a&b).sum()),BothWrong=int((~a&~b).sum())))
        write(aligned.reset_index(),output/f'statistical_preparation/aligned_{level}_predictions.csv')
    tables['error_overlap']=pd.DataFrame(overlaps)
    draws=np.random.default_rng(42).integers(0,233,size=(2000,233))
    np.savez_compressed(output/'statistical_preparation/shared_patient_bootstrap_draws.npz', indices=draws, PID=patients['morphonet_bt'].sort_values('PID').PID.to_numpy(dtype=str))
    replicates={}; cis=[]
    for n in names:
        _,consistency=aggregate_patients(oofs[n]); consistency=consistency.sort_values('PID')
        count=consistency.n_slices.to_numpy(); correct=consistency.n_correct.to_numpy()
        values=correct[draws].sum(axis=1)/count[draws].sum(axis=1)
        low,high=np.percentile(values,[2.5,97.5]); replicates[display[n]]=values
        cis.append(dict(Model=display[n],Accuracy=correct.sum()/count.sum(),Lower95=low,Upper95=high))
    tables['accuracy_cluster_bootstrap_ci']=pd.DataFrame(cis)
    stored=read(main/'accuracy_cluster_bootstrap_ci.csv').iloc[0]
    require(np.allclose([cis[0]['Lower95'],cis[0]['Upper95']],[stored.Lower95,stored.Upper95],rtol=0,atol=1e-15),'Phase 7 bootstrap mismatch')
    require(np.array_equal(replicates['MorphoNet-BT'],read(main/'bootstrap_replicates.csv').Accuracy.to_numpy()), 'Phase 7 bootstrap draws mismatch')
    write(pd.DataFrame(replicates).assign(Iteration=np.arange(1,2001)),output/'statistical_preparation/shared_bootstrap_replicates.csv')
    for name,table in tables.items(): write(table,output/f'{name}.csv')
    # Explicit programmatic best values; these are descriptive ranks, not tests.
    winners=[]
    for label in ['final_model_comparison','patient_level_model_comparison']:
        for col in tables[label].columns:
            if col.startswith(('OOF ','Patient ')):
                best=tables[label][col].max()
                winners.append(dict(Table=label,Metric=col,BestValue=best,Models='; '.join(tables[label].loc[tables[label][col].eq(best),'Model'])))
    write(pd.DataFrame(winners),output/'tables/metric_leaders.csv')
    for label in ['final_model_comparison','patient_level_model_comparison','efficiency_comparison']:
        (output/f'tables/{label}.html').write_text(tables[label].to_html(index=False,float_format=lambda x:f'{x:.4f}'),encoding='utf-8')
        (output/f'tables/{label}.tex').write_text(tables[label].to_latex(index=False,float_format='%.4f'),encoding='utf-8')
    summary_figures(output,tables)
    context['metadata']['bootstrap']['interpretation']='Same 2000 sorted-PID draws across all six models; entire patient clusters retained with multiplicity. Conditional uncertainty of fixed OOF predictions; excludes training variability.'
    context['metadata']['reference_inference_passes']=0
    context['metadata']['paired_alignment']={'slices':3064,'patients':233,'identity_truth_fold_verified':True}
    save_json(output/'baseline_evaluation_metadata.json',context['metadata'])
    return tables


def summary_figures(output,tables):
    folder=output/'figures'; folder.mkdir(exist_ok=True)
    with plt.rc_context({'font.size':10,'axes.titlesize':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        for table,col,unit,filename in [('final_model_comparison','OOF Macro F1','OOF macro F1','oof_macro_f1'),('final_model_comparison','OOF Accuracy','OOF accuracy','oof_accuracy'),('patient_level_model_comparison','Patient Macro F1','Patient macro F1','patient_macro_f1'),('final_model_comparison','Parameters (M)','Parameters (millions)','parameters'),('final_model_comparison','MACs (G)','Conv/Linear MACs (billions)','macs')]:
            data=tables[table]; fig,ax=plt.subplots(figsize=(7.3,4.3),layout='constrained')
            bars=ax.barh(data.Model,data[col],color=['#0072B2']+['#888888']*5)
            ax.invert_yaxis(); ax.set_xlabel(unit); ax.set_xlim(0,1.12 if 'F1' in col or 'Accuracy' in col else data[col].max()*1.2)
            ax.bar_label(bars,labels=[f'{v:.4f}' for v in data[col]],padding=5,fontsize=9)
            for suffix in ['png','pdf']: fig.savefig(folder/f'{filename}.{suffix}',dpi=400,bbox_inches='tight')
            plt.close(fig)


def final_audit(context):
    output=context['root']/'results/baselines/evaluation'
    for relative,digest in context['protected'].items(): require(sha256(context['root']/relative)==digest,f'Protected artifact changed: {relative}')
    receipts=[]
    for n in BASELINES:
        for f in range(1,6):
            folder=output/n/f'fold_{f}'
            receipt=json.loads((folder/'test_evaluation_receipt.json').read_text())
            require(receipt['inference_passes']==1 and receipt['status']=='complete','Inference receipt mismatch')
            require(sha256(folder/'test_predictions.csv')==receipt['predictions_sha256'],'Prediction digest mismatch')
            receipts.append(receipt)
    audit=dict(best_checkpoints_evaluated=len(receipts),last_checkpoints_loaded=0,retraining_runs=0,test_augmentation=False,
        MRI_only=True,all_oof_audits_passed=True,shared_frozen_folds=True,slice_and_patient_metrics_saved=True,
        confusion_and_ROC_figures_saved=True,shared_patient_cluster_bootstrap=True,exact_paired_alignment=True,
        frozen_reference_complexity_training_preserved=True,statistical_significance_tests=0,ablation_runs=0,
        protected_artifacts_verified=len(context['protected']))
    save_json(output/'final_scientific_audit.json',audit)
    return audit
