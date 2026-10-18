"""Publication consolidation from frozen artifacts. No model or MRI loader imports."""
from pathlib import Path
from datetime import datetime, timezone
import json
import platform
import hashlib
import shutil
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score
from src import phase14_statistics as stats

NAMES = {**stats.DISPLAY, 'A4': 'MorphoNet-BT A4', 'A5': 'MorphoNet-BT A5'}
MAIN = ['A4', 'A5'] + stats.BASELINES
METRICS = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Weighted F1', 'Macro ROC-AUC']
SELECTION = ('A4 was selected post hoc after inspection of Phase 13 outer-held-out results. '
             'These results are exploratory and potentially optimistic; they are not independent confirmation after selection.')

def markdown(df, floatfmt='.6f'):
    def fmt(v):
        return format(v,floatfmt) if isinstance(v,(float,np.floating)) else str(v).replace('|',r'\|').replace('\n',' ')
    rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    rows += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def metrics(frame):
    y, p = frame.true_label.to_numpy(int), frame.predicted_label.to_numpy(int)
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    tp = np.diag(cm).astype(float); support = cm.sum(1); predicted = cm.sum(0)
    def div(a, b):
        return np.divide(a, b, out=np.full(3, np.nan), where=b != 0)
    precision, recall = div(tp, predicted), div(tp, support)
    f1 = div(2*tp, support+predicted)
    auc = np.array([roc_auc_score(y == k, frame[stats.PROBS[k]]) if len(np.unique(y == k)) == 2 else np.nan for k in range(3)])
    values = [tp.sum()/len(y), precision.mean(), recall.mean(), f1.mean(), np.sum(f1*support)/len(y), auc.mean()]
    per = pd.DataFrame({'Class': stats.CLASSES, 'Support': support, 'Precision': precision, 'Recall': recall, 'F1': f1, 'ROC-AUC': auc})
    return dict(zip(METRICS, values)), per

def texescape(value):
    table = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(table.get(c, c) for c in str(value))

class Package:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.out = self.root/'results/publication'
        for d in ['tables/csv', 'tables/latex', 'tables/markdown', 'figures/main', 'figures/supplementary', 'captions', 'manuscript_snippets', 'audits', 'metadata']:
            (self.out/d).mkdir(parents=True, exist_ok=True)
        self.sources = {}; self.lineage = []; self.tables = {}; self.inventory = []; self.figures = []

    def source(self, path):
        p = Path(path)
        if not p.is_absolute(): p = self.root/p
        key = p.relative_to(self.root).as_posix()
        if key not in self.sources: self.sources[key] = dict(sha256=stats.sha(p), bytes=p.stat().st_size)
        return key

    def csv(self, path):
        self.source(path); return stats.read(self.root/path)

    def js(self, path):
        self.source(path); return json.loads((self.root/path).read_text(encoding='utf-8'))

    def table(self, name, df, caption, sources, calculation, allocation='supplementary'):
        df = df.reset_index(drop=True); self.tables[name] = df
        source_names = [self.source(s) for s in sources]
        stats.save(df, self.out/f'tables/csv/{name}.csv')
        shown = df.copy().astype(object)
        for c in df:
            for i, value in enumerate(df[c]):
                if isinstance(value, (float, np.floating)):
                    shown.loc[i,c] = f'{value:.6g}' if 'p_' in c else f'{value:.5f}'
                self.lineage.append(dict(table=name, row=i, column=c, value=value.item() if isinstance(value,np.generic) else value,
                                         source_files=source_names, calculation=calculation))
        md = markdown(shown)
        (self.out/f'tables/markdown/{name}.md').write_text(f'# {name}\n\n{caption}\n\n{md}\n', encoding='utf-8')
        # Long supplementary tables are intentionally longtable, with repeated headers.
        long = len(df) > 24
        env = 'longtable' if long else 'tabular'
        cols = 'l' + 'r'*(len(df.columns)-1)
        header = ' & '.join(texescape(c) for c in df.columns) + r' \\'
        lines = [f'% {caption}', f'\\begin{{{env}}}{{{cols}}}', r'\toprule', header, r'\midrule']
        if long: lines += [r'\endhead']
        for row in shown.itertuples(index=False, name=None):
            cells = [texescape(v) for v in row]
            if any(str(v) in ['A4','MorphoNet-BT A4'] for v in row): cells = [r'\textbf{'+v+'}' for v in cells]
            lines.append(' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', f'\\end{{{env}}}', '']
        (self.out/f'tables/latex/{name}.tex').write_text('\n'.join(lines), encoding='utf-8')
        self.inventory.append(dict(name=name, rows=len(df), columns=len(df.columns), allocation=allocation, caption=caption, calculation=calculation))
        return df

def prediction_sources(model):
    base = 'results/main' if model == 'A5' else f'results/baselines/evaluation/{model}' if model in stats.BASELINES else f'results/ablation/evaluation/{model}'
    return [f'{base}/oof_predictions.csv', f'{base}/patient_level_predictions.csv']

def build_tables(pkg, frames):
    patient_manifest = pkg.csv('splits/patient_manifest.csv')
    data = patient_manifest.groupby('class_name').agg(Slices=('num_slices','sum'), Patients=('PID','size')).reindex(stats.CLASSES).reset_index(names='Class')
    data.loc[len(data)] = ['Total', int(data.Slices.sum()), int(data.Patients.sum())]
    data['Slices (%)'] = 100*data.Slices/3064; data['Patients (%)'] = 100*data.Patients/233
    assert data.Slices.tolist()==[708,1426,930,3064] and data.Patients.tolist()==[82,89,62,233]
    pkg.table('T01_dataset', data, 'Class distribution. Percentages use 3,064 slices and 233 patients.', ['splits/patient_manifest.csv'], 'Sum num_slices and count unique patient rows by class; divide by cohort total.')
    pkg.table('S01_fold_splits', pkg.csv('results/kfold/fold_summary.csv'), 'Frozen train, validation and test counts. Patients are disjoint within each fold; fold class distributions are not perfectly balanced.', ['results/kfold/fold_summary.csv'], 'Unchanged frozen Phase 2 split summary.')
    arch = pkg.csv('results/ablation/setup/ablation_architecture_summary.csv').set_index('Variant')
    bas = pkg.csv('results/baselines/setup/baseline_architecture_summary.csv').set_index('Model')
    complexity=[]
    for m in MAIN:
        if m in ['A4','A5']: params, mac = int(arch.loc[m,'Parameters']), int(arch.loc[m,'Inference MACs'])
        else: params, mac = int(bas.loc[NAMES[m],'Parameters']), int(bas.loc[NAMES[m],'ConvLinearMACs'])
        complexity.append({'Model':NAMES[m], 'Parameters (M)':params/1e6, 'MACs (G)':mac/1e9, 'FLOPs (G)':2*mac/1e9, 'FP32 parameter MB':4*params/1e6})
    assert arch.loc['A4','Parameters']==575359 and arch.loc['A4','Inference MACs']==275711872
    pkg.table('T02_complexity',pd.DataFrame(complexity),'MRI input 1 x 256 x 256; convolution/linear MACs only; FLOPs = 2 MACs. FP32 MB is theoretical parameter payload (decimal), not measured checkpoint size. Comparable batch-1 latency is unavailable and is not compared.', ['results/ablation/setup/ablation_architecture_summary.csv','results/baselines/setup/baseline_architecture_summary.csv'], 'Frozen parameter/MAC counts; divide by 1e6/1e9; FP32 bytes = 4 parameters. Historical MorphoNet-BT entry denotes A5.')
    scores={}; foldrows=[]; classrows=[]; checks=[]
    for m in stats.MODELS:
        scores[m]={}
        for level in ['slice','patient']:
            d, per = metrics(frames[m][level]); scores[m][level]=d
            per.insert(0,'Level',level);per.insert(0,'Model',NAMES.get(m,m));classrows.extend(per.to_dict('records'))
            for f in range(1,6):
                vals,_=metrics(frames[m][level].loc[frames[m][level].fold==f]); foldrows.append(dict(Model=NAMES.get(m,m),Level=level,Fold=f,**vals))
            folder=prediction_sources(m)[0].rsplit('/',1)[0]
            frozen=pkg.csv(folder+('/final_performance_summary.csv' if level=='slice' else '/patient_level_metrics.csv'))
            expected=frozen.set_index('Metric')['AggregatedOOFValue' if level=='slice' else 'Value']
            for metric,value in d.items():
                err=abs(float(expected[metric])-value);assert err<1e-12,(m,level,metric,err)
                checks.append(dict(model=m,level=level,metric=metric,recomputed=value,frozen=float(expected[metric]),absolute_error=err,status='PASS'))
    allpred=[p for m in stats.MODELS for p in prediction_sources(m)]
    for level,num in [('slice','03'),('patient','04')]:
        df=pd.DataFrame([dict(Model=NAMES[m],**scores[m][level]) for m in MAIN])
        pkg.table(f'T{num}_{level}_comparison',df, f'Pooled outer-held-out {level} results on '+('3,064 slices.' if level=='slice' else '233 patients. Float64 slice probabilities are averaged within PID, without renormalization, then argmax is applied.')+' Values are proportions. '+SELECTION, [p for m in MAIN for p in prediction_sources(m)], 'Frozen 3-class confusion matrix metrics; mean of three binary one-vs-rest AUCs; undefined ratios NaN. No mean-of-fold substitution.', 'main')
    rows=[]
    for m in [f'A{i}' for i in range(6)]:
        rows.append(dict(Variant=m,Parameters=int(arch.loc[m,'Parameters']),MACs=int(arch.loc[m,'Inference MACs']),Slice_Accuracy=scores[m]['slice']['Accuracy'],Slice_Macro_F1=scores[m]['slice']['Macro F1'],Patient_Accuracy=scores[m]['patient']['Accuracy'],Patient_Macro_F1=scores[m]['patient']['Macro F1'],Patient_Macro_ROC_AUC=scores[m]['patient']['Macro ROC-AUC']))
    pkg.table('T05_sequential_ablation',pd.DataFrame(rows),'Cumulative sequence: A0 core; A1 + MSS; A2 + privileged morphology supervision; A3 + BAA; A4 + global/local paths and fixed mean fusion; A5 + AMGF. These are not independent factorial effects. A3 and A5 do not improve pooled performance. '+SELECTION,allpred+['results/ablation/setup/ablation_architecture_summary.csv'],'Frozen complexity joined to pooled prediction-derived metrics.','main')
    tests=pkg.csv('results/statistical_analysis/exact_mcnemar_holm.csv')
    verified=stats.primary_tests(frames)
    for row in verified.itertuples():
        old=tests[(tests.left==row.left)&(tests.right==row.right)].iloc[0]
        for c in ['left_only','right_only','p_exact','effect_pp','p_holm']:
            assert abs(float(old[c])-float(getattr(row,c)))<1e-12
    effects=pkg.csv('results/statistical_analysis/paired_effects_with_ci.csv')
    pkg.source('results/statistical_analysis/bootstrap_metric_replicates.npz')
    replicates=np.load(pkg.root/'results/statistical_analysis/bootstrap_metric_replicates.npz')
    for row in effects.itertuples():
        key={'Macro-F1':'Macro F1','Macro-OVR-AUC':'Macro ROC-AUC','Accuracy':'Accuracy'}[row.metric]
        actual=100*(scores[row.left][row.level][key]-scores[row.right][row.level][key])
        assert abs(actual-row.effect_pp)<1e-11
        index=stats.METRICS.index(row.metric)
        delta=100*(replicates[f'{row.left}__{row.level}'][:,index]-replicates[f'{row.right}__{row.level}'][:,index])
        np.testing.assert_allclose(np.nanpercentile(delta,[2.5,97.5]),[row.ci_low_pp,row.ci_high_pp],atol=1e-11,rtol=0)
    merged=tests.merge(effects[(effects.level=='patient')&(effects.metric=='Accuracy')][['left','right','ci_low_pp','ci_high_pp']],on=['left','right'],validate='one_to_one')
    cols=['left','right','n','left_only','right_only','effect_pp','ci_low_pp','ci_high_pp','p_exact','p_holm']
    pkg.table('T06_exploratory_statistics',merged[(merged.left=='A4')&merged.right.isin(MAIN[1:])][cols], 'Patient Accuracy difference (percentage points), exact two-sided McNemar and frozen Holm adjustment across TEN unique comparisons. Patient-paired percentile bootstrap 95% CI, 10,000 replicates, seed 42. A5 vs A4 is a reverse display, not an 11th test. McNemar does not test Macro-F1. '+SELECTION,['results/statistical_analysis/exact_mcnemar_holm.csv','results/statistical_analysis/paired_effects_with_ci.csv'],'Import frozen Phase 14 values, independently verify discordance counts, exact p-values, 10-test Holm and point effects.','main')
    seq=[('A1','A0'),('A2','A1'),('A3','A2'),('A4','A3'),('A5','A4')]
    pkg.table('S02_sequential_statistics',pd.concat([merged[(merged.left==l)&(merged.right==r)] for l,r in seq])[cols], 'Sequential exploratory patient Accuracy comparisons; positive effects favor the added component. Same 10-test Holm family as T06, not a separately corrected subset.', ['results/statistical_analysis/exact_mcnemar_holm.csv','results/statistical_analysis/paired_effects_with_ci.csv'],'Select five cumulative pairs without recalculating correction family.')
    for name,filename in [('S03_paired_bootstrap_all_metrics','paired_effects_with_ci.csv'),('S04_absolute_bootstrap_intervals','model_metrics_with_ci.csv'),('S05_fold_robustness','fold_robustness_summary.csv'),('S06_meningioma_recall','meningioma_paired_recall.csv'),('S07_meningioma_tests','meningioma_exact_mcnemar_holm.csv')]:
        pkg.table(name,pkg.csv('results/statistical_analysis/'+filename),'Frozen Phase 14 exploratory results. Patient-level resampling retains pairing; slice-level resampling retains whole patient clusters. CIs are unadjusted percentile intervals. Meningioma tests use a separate 10-test family.', ['results/statistical_analysis/'+filename],'Unchanged Phase 14 table; no new statistical family or inference.')
    explanation=[]
    for source,kind in [('gradcam_localization_summary.csv','Grad-CAM'),('baa_localization_summary.csv','BAA'),('morphology_summary.csv','Morphology')]:
        d=pkg.csv('results/explainability/tables/'+source)
        d=d[(d.group=='overall')&(d.estimand=='slice_weighted')]
        if kind=='Grad-CAM': d=d[(d.layer=='stage3')&(d.target=='predicted')&d.metric.isin(['attention_mass','peak_inside'])]
        if kind=='BAA': d=d[d.metric.isin(['attention_mass','peak_inside'])]
        if kind=='Morphology': d=d[((d['head']=='region')&d.metric.isin(['dice','iou']))|((d['head']=='boundary')&(d.metric=='f1_2px'))|((d['head']=='distance')&d.metric.isin(['mae','rmse','tumor_mae','background_mae']))]
        for _,r in d.iterrows(): explanation.append(dict(Map=kind if kind!='Morphology' else r['head'],Metric=r.metric,Mean=r['mean'],CI_low=r.ci_low,CI_high=r.ci_high,Slices=int(r.slices),Patients=int(r.patients)))
    pkg.table('T07_explainability',pd.DataFrame(explanation),'Slice-weighted means; 95% patient-cluster bootstrap intervals. Primary Grad-CAM targets the predicted class at stage3 (16 x 16). BAA is class-agnostic gating, not equivalent to Grad-CAM. Peak-inside is a fraction (multiply by 100 for percent). Boundary F1 uses 2-pixel tolerance on the 256 grid. Distance target is internal Euclidean distance normalized per mask; full-grid MAE can be background dominated. Auxiliary outputs are not clinical segmentation.', ['results/explainability/tables/'+s for s in ['gradcam_localization_summary.csv','baa_localization_summary.csv','morphology_summary.csv']], 'Select overall slice-weighted frozen estimands; do not mix patient-equal means.')
    oc=pkg.csv('results/explainability/occlusion_audit/tables/all_effect_summaries.csv'); rows=[]
    for strategy in ['zero','local_median']:
        for group in ['overall','Meningioma','Glioma','Pituitary']:
            d=oc[(oc.strategy==strategy)&(oc.group==group)&(oc.estimand=='patient_equal')].set_index('metric')
            r=d.loc['delta_true_logit_drop']; rows.append(dict(Replacement=strategy,Group=group,Slices=int(r.slices),Patients=int(r.patients),Controls=int(r.slices)*10,Tumor_drop=d.loc['tumor_true_logit_drop','mean'],Control_drop=d.loc['control_true_logit_drop','mean'],Delta=r['mean'],CI_low=r.ci_low,CI_high=r.ci_high))
    pkg.table('T08_occlusion_sensitivity',pd.DataFrame(rows),'Patient-equal true-class logit drops and paired DeltaSensitivity = tumor drop minus mean matched-control drop. 95% patient-bootstrap percentile CI (10,000, seed 42). The 30,490 control locations are clustered within 3,049 eligible slices and 233 patients, not independent samples; two replacements reuse the same locations (60,980 responses). Fifteen slices have no eligible control. Geometric matching does not establish anatomical or intensity matching. Exploratory perturbations, without formal p-values.', ['results/explainability/occlusion_audit/tables/all_effect_summaries.csv','results/explainability/occlusion_audit/final_scientific_audit.json'], 'Select patient_equal estimands, retain both replacements and all classes; ten frozen controls per eligible slice.')
    pkg.table('S08_occlusion_all_estimands',oc,'All frozen Phase 15.1 class, correctness, replacement and weighting results. Slice-weighted and patient-equal estimands are explicitly separated.', ['results/explainability/occlusion_audit/tables/all_effect_summaries.csv'],'Unchanged frozen summary, including unfavorable subgroup effects.')
    folds=pd.DataFrame(foldrows)
    pkg.table('S09_fold_metrics',folds,'Metrics within each frozen outer test fold. Results are proportions; all models and both levels are preserved.',allpred,'Same six metrics applied independently to each saved fold.')
    summaries=folds.groupby(['Model','Level'])[METRICS].agg(['mean','std']).reset_index()
    summaries.columns=[' '.join(c).strip() for c in summaries.columns]
    pkg.table('S10_fold_mean_sd',summaries,'Unweighted five-fold mean and sample standard deviation (ddof=1). These are not pooled OOF estimates.',allpred,'Unweighted mean and sample SD of S09, not independent repeated trials.')
    pkg.table('S11_per_class',pd.DataFrame(classrows),'All models, both levels and all classes; proportions. Competing models can outperform A4 on individual classes.',allpred,'Class precision, recall, F1 and binary one-vs-rest AUC from saved predictions.')
    val=pkg.csv('results/ablation/training/ablation_training_summary.csv'); val=val[val.Variant=='A4']; vr=[]; vsources=['results/ablation/training/ablation_training_summary.csv']
    for r in val.itertuples(index=False,name=None):
        row=dict(zip(val.columns,r)); vr.append(dict(Model='A4',Fold=int(row['Fold']),Best_epoch=int(row['Best Epoch']),Validation_Macro_F1=float(row['Best Validation Macro-F1'])))
    for f in range(1,6):
        s=f'results/main/fold_{f}/training_complete.json';j=pkg.js(s);vsources.append(s)
        vr.append(dict(Model='A5',Fold=f,Best_epoch=j['best_epoch'],Validation_Macro_F1=j['best_validation_macro_f1']))
    vdf=pd.DataFrame(vr)
    pkg.table('S12_validation_folds',vdf,'Frozen checkpoint-selection validation Macro-F1. No checkpoint is reselected.',vsources,'Read best-validation metrics and epoch from completed training records.')
    sel=pd.DataFrame([dict(Model=m,Validation_Macro_F1_mean=vdf[vdf.Model==m].Validation_Macro_F1.mean(),Validation_Macro_F1_SD=vdf[vdf.Model==m].Validation_Macro_F1.std(ddof=1),Held_out_slice_Macro_F1=scores[m]['slice']['Macro F1'],Held_out_patient_Macro_F1=scores[m]['patient']['Macro F1']) for m in ['A4','A5']])
    assert sel.iloc[1].Validation_Macro_F1_mean>sel.iloc[0].Validation_Macro_F1_mean
    pkg.table('S13_selection_audit',sel,'A5 has the higher mean validation Macro-F1; A4 has higher held-out pooled Macro-F1. '+SELECTION,vsources+prediction_sources('A4')+prediction_sources('A5'),'Five-fold validation mean/sample SD versus pooled outer-held-out performance.')
    stats.save(pd.DataFrame(checks),pkg.out/'audits/metric_recomputation.csv')
    return scores

def run(root):
    pkg=Package(root)
    code_at_start={p.relative_to(pkg.root).as_posix():stats.sha(p) for p in (pkg.root/'src').glob('phase16_*.py')}
    receipt_path='results/explainability/occlusion_audit/metadata/completion_receipt.json'
    receipt=pkg.js(receipt_path)
    protected={**receipt['protected'],**receipt['outputs'],**receipt['source_hashes']}
    protected[receipt_path]=stats.sha(pkg.root/receipt_path)
    for directory in ['configs','splits','src/models']:
        for p in (pkg.root/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in str(p): protected[p.relative_to(pkg.root).as_posix()]=stats.sha(p)
    print(f'Verifying {len(protected)} frozen artifacts before publication work...',flush=True)
    stats.verify_manifest(pkg.root,protected)
    stats.dump(pkg.out/'metadata/protected_source_hashes.json',protected)
    synthetic=stats.validate_synthetic();stats.dump(pkg.out/'audits/synthetic_validation.json',synthetic)
    frames,alignment=stats.load_aligned(pkg.root);stats.save(alignment,pkg.out/'audits/prediction_alignment.csv')
    print('All 11 models aligned: 3,064 slices, 233 patients, five folds.',flush=True)
    split=pkg.js('results/kfold/split_metadata.json')
    assert split['seed']==42 and split['outer_strategy']=='StratifiedGroupKFold' and split['n_splits']==5
    for file,expected in split['split_sha256'].items(): assert stats.sha(pkg.root/'splits'/file)==expected
    import yaml
    pkg.source('configs/default.yaml');config=yaml.safe_load((pkg.root/'configs/default.yaml').read_text())
    assert config['image_size']==256 and config['normalization']['method']=='zscore' and config['normalization']['clipping'] is False
    for p in ['src/models/ablation.py','src/models/morphonet_bt.py','src/models/blocks.py','src/models/attention.py','results/ablation/setup/ablation_shape_audit.csv']:
        pkg.source(p)
    scores=build_tables(pkg,frames)
    from src.phase16_figures import build_figures
    build_figures(pkg,frames,scores)
    from src.phase16_reporting import build_report
    build_report(pkg,scores,split,config)
    print('Rechecking frozen artifacts after all aggregate analysis and rendering...',flush=True)
    stats.verify_manifest(pkg.root,protected)
    stats.dump(pkg.out/'metadata/source_of_truth_manifest.json',dict(sources=pkg.sources,table_cells=pkg.lineage,figures=pkg.figures))
    stats.dump(pkg.out/'metadata/table_inventory.json',pkg.inventory)
    stats.dump(pkg.out/'metadata/figure_inventory.json',pkg.figures)
    audit=dict(status='PASS',phase=16,training_runs=0,optimizer_steps=0,new_model_inference=0,checkpoints_reselected=0,
               previous_artifacts_unchanged=True,protected_files=len(protected),models_aligned=11,slices=3064,patients=233,
               patient_disjoint_folds=5,metric_recomputation='PASS (132 comparisons)',McNemar_and_10_test_Holm='PASS',
               A4_parameters=575359,A4_inference_MACs=275711872,A4_AMGF=False,A4_MRI_only=True,
               post_hoc_selection_disclosed=True,A5_validation_mean_higher=True,tables=len(pkg.inventory),
               figures=len(pkg.figures),unresolved_numerical_discrepancies=[],external_validation=False,clinical_validation=False,
               latex_snippets='booktabs/longtable source exported; full manuscript compilation is outside this phase',phase17_started=False)
    stats.dump(pkg.out/'final_scientific_audit.json',audit)
    stats.dump(pkg.out/'metadata/publication_metadata.json',dict(project_version='Phase16-frozen-publication-v1',generated_utc=datetime.now(timezone.utc).isoformat(),
        model='A4 (post-hoc principal); A5 (historical full comparator)',parameters=575359,MACs=275711872,split=split,
        preprocessing=config['normalization'],class_order=stats.CLASSES.tolist(),patient_aggregation='Float64 mean of saved slice probabilities within PID, no renormalization, argmax',
        metric_definitions='Frozen confusion-matrix ratios; equal class macro means; support-weighted F1; binary OVR AUC mean; undefined ratios NaN',
        statistics=dict(iterations=10000,seed=42,interval='percentile 2.5/97.5',resampling='paired patients; whole-patient clusters for slices',Holm_unique_tests=10,scope='exploratory'),
        explainability=pkg.js('results/explainability/final_scientific_audit.json'),occlusion=pkg.js('results/explainability/occlusion_audit/final_scientific_audit.json'),
        figures=dict(dpi=600,font='DejaVu Sans',font_pt=8,PDF_embedded_type=42,column_mm=84,double_column_mm=174),
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        journal_guidelines='https://link.springer.com/journal/11760/submission-guidelines',journal_checked='2026-10-10',
        source_code={p.name:stats.sha(p) for p in (pkg.root/'src').glob('phase16_*.py')}))
    stats.verify_manifest(pkg.root,code_at_start)
    stats.dump(pkg.out/'metadata/completion_receipt.json',dict(status='complete',outputs={p.relative_to(pkg.root).as_posix():stats.sha(p) for p in pkg.out.rglob('*') if p.is_file() and p.name not in ['completion_receipt.json','notebook_execution_audit.json']},source_hashes=pkg.sources,executed_code_hashes=code_at_start))
    print(json.dumps(audit,indent=2));return audit
