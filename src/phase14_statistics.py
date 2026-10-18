"""Phase 14: statistics on frozen predictions only; no model dependencies."""
from pathlib import Path
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sklearn

PROBS = ['prob_meningioma', 'prob_glioma', 'prob_pituitary']
CLASSES = np.array(['Meningioma', 'Glioma', 'Pituitary'])
BASELINES = ['resnet18', 'resnet50', 'densenet121', 'mobilenet_v3_small', 'efficientnet_b0']
MODELS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5'] + BASELINES
DISPLAY = dict(zip(BASELINES, ['ResNet18', 'ResNet50', 'DenseNet121', 'MobileNetV3-Small', 'EfficientNet-B0']))
PAIRS = [('A4', 'A5')] + [('A4', b) for b in BASELINES] + [('A1', 'A0'), ('A2', 'A1'), ('A3', 'A2'), ('A4', 'A3')]
METRICS = ['Accuracy', 'Macro-F1', 'Macro-OVR-AUC']
SETTINGS = dict(CONFIRM_STATISTICAL_ANALYSIS=True, OVERWRITE=False, RESUME_ANALYSIS=True, BOOTSTRAP_ITERATIONS=10000, RANDOM_SEED=42)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def read(path):
    return pd.read_csv(path, dtype={'PID': str, 'sample_id': str}, keep_default_na=False, float_precision='round_trip')

def save(df, path):
    df.to_csv(path, index=False, float_format='%.17g')

def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty_like(p)
    out[order] = adjusted
    return out

def mcnemar(left, right):
    left, right = np.asarray(left, bool), np.asarray(right, bool)
    b, c = int(np.sum(left & ~right)), int(np.sum(~left & right))
    p = float(binomtest(b, b+c, .5, alternative='two-sided').pvalue) if b+c else 1.
    return dict(both_correct=int(np.sum(left & right)), left_only=b, right_only=c,
                both_wrong=int(np.sum(~left & ~right)), discordant=b+c, p_exact=p)

def draws(n, iterations, rng):
    indices = rng.integers(0, n, size=(iterations, n))
    counts = np.zeros((iterations, n), dtype=np.int32)
    np.add.at(counts, (np.arange(iterations)[:, None], indices), 1)
    return counts

def weighted_auc(y, scores, weights):
    """Batched weighted binary AUC, with half credit for tied scores."""
    order = np.argsort(scores, kind='stable')
    values, labels = np.asarray(scores)[order], np.asarray(y)[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    w = weights[:, order]
    positive = np.add.reduceat(w * labels, starts, axis=1)
    negative = np.add.reduceat(w * (1-labels), starts, axis=1)
    numerator = np.sum(positive * (np.cumsum(negative, axis=1) - .5 * negative), axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)

def bootstrap_metrics(frame, pids, counts):
    mapping = {pid: i for i, pid in enumerate(pids)}
    cluster = frame.PID.map(mapping).to_numpy(dtype=int)
    y, pred = frame.true_label.to_numpy(int), frame.predicted_label.to_numpy(int)
    contribution = np.zeros((len(pids), 9), dtype=np.float64)
    np.add.at(contribution, (cluster, 3*y+pred), 1)
    cm = (counts @ contribution).reshape(-1, 3, 3)
    tp = cm[:, np.arange(3), np.arange(3)]
    den = cm.sum(axis=1) + cm.sum(axis=2)
    accuracy = tp.sum(axis=1) / cm.sum(axis=(1, 2))
    f1 = np.divide(2*tp, den, out=np.zeros_like(tp), where=den > 0).mean(axis=1)
    auc = np.empty(len(counts))
    scores = frame[PROBS].to_numpy()
    for start in range(0, len(counts), 200):
        weights = counts[start:start+200, cluster].astype(float)
        auc[start:start+200] = np.mean([weighted_auc(y == k, scores[:, k], weights) for k in range(3)], axis=0)
    return np.column_stack([accuracy, f1, auc])

def point(frame):
    y, pred = frame.true_label.to_numpy(int), frame.predicted_label.to_numpy(int)
    return np.array([accuracy_score(y, pred), f1_score(y, pred, average='macro', labels=[0,1,2], zero_division=0),
                     roc_auc_score(y, frame[PROBS].to_numpy(), average='macro', multi_class='ovr', labels=[0,1,2])])

def interval(x):
    return np.nanpercentile(x, [2.5, 97.5], method='linear')

def validate_synthetic():
    checks = []
    assert mcnemar([1,1], [1,1])['p_exact'] == 1
    assert mcnemar([1]*10, [0]*10)['p_exact'] == 2/1024
    for b in range(16):
        for c in range(16):
            r = mcnemar([1]*b+[0]*c, [0]*b+[1]*c)
            expected = min(1., 2 * sum(math.comb(b+c,k) for k in range(min(b,c)+1)) / 2**(b+c))
            assert abs(r['p_exact']-expected) < 1e-14
    checks.append('Exact McNemar agrees with independent binomial enumeration for all 256 discordance pairs (0..15).')
    np.testing.assert_allclose(holm([.01,.04,.03,.002]), [.03,.06,.06,.008])
    np.testing.assert_allclose(holm([1,1,1]), [1,1,1])
    np.testing.assert_allclose(holm([.02,.02,.001]), [.04,.04,.003])
    checks.append('Holm known examples, ties, monotonicity and clipping passed.')
    rng = np.random.default_rng(19)
    y = np.repeat([0,1,2], 4)
    probs = rng.integers(1,5, size=(12,3)).astype(float)
    probs /= probs.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(probs, columns=PROBS)
    frame['PID'] = np.repeat(np.arange(6).astype(str), 2)
    frame['true_label'], frame['predicted_label'] = y, probs.argmax(axis=1)
    pids = sorted(frame.PID.unique())
    counts = draws(6, 50, rng)
    actual = bootstrap_metrics(frame, pids, counts)
    for i, count in enumerate(counts):
        expanded = frame.iloc[np.repeat(np.arange(12), count[np.repeat(np.arange(6),2)])]
        assert abs(actual[i,0] - accuracy_score(expanded.true_label, expanded.predicted_label)) < 1e-14
        assert abs(actual[i,1] - f1_score(expanded.true_label, expanded.predicted_label, average='macro', labels=[0,1,2], zero_division=0)) < 1e-14
        if expanded.true_label.nunique() == 3:
            np.testing.assert_allclose(actual[i], point(expanded), atol=2e-15, rtol=0)
        else:
            assert np.isnan(actual[i,2])
    weights = rng.integers(1,5,size=(20,12))
    for k in range(3):
        expected = [roc_auc_score(y==k, probs[:,k], sample_weight=w) for w in weights]
        np.testing.assert_allclose(weighted_auc(y==k, probs[:,k], weights), expected, atol=2e-15, rtol=0)
    np.testing.assert_allclose(weighted_auc([0,1], [.5,.5], np.ones((1,2))), [.5])
    assert np.isnan(weighted_auc([1,1], [.1,.2], np.ones((1,2)))[0])
    checks.append('Weighted tied-score AUC matches sklearn; clustered bootstrap matches explicitly expanded samples, including absent-class handling.')
    np.testing.assert_allclose(interval(actual[:,0]-actual[:,0]), [0,0])
    np.testing.assert_allclose(interval(actual[:,0]), -interval(-actual[:,0])[::-1])
    checks.append('Paired identity effect is zero and reversed percentile intervals have correct sign.')
    return dict(status='PASS', checks=checks)

def source_paths(root, model):
    if model in BASELINES:
        folder = root/'results/baselines/evaluation'/model
        return folder, folder/'patient_level_predictions.csv', 'test_predictions.csv', 'test_evaluation_receipt.json'
    if model == 'A5':
        folder = root/'results/main'
        return folder, folder/'patient_level_predictions.csv', 'test_predictions.csv', 'test_evaluation_receipt.json'
    folder = root/'results/ablation/evaluation'/model
    return folder, folder/'patient_predictions.csv', 'predictions.csv', 'evaluation_receipt.json'

def assert_frame(frame, patient=False):
    key = 'PID' if patient else 'sample_id'
    assert not frame[key].duplicated().any(), f'Duplicate {key}'
    p = frame[PROBS].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6, rtol=0)
    np.testing.assert_array_equal(p.argmax(axis=1), frame.predicted_label)
    assert set(frame.true_label) == {0,1,2}
    np.testing.assert_array_equal(CLASSES[frame.true_label.to_numpy(int)], frame.true_class)
    np.testing.assert_array_equal(CLASSES[frame.predicted_label.to_numpy(int)], frame.predicted_class)

def verify_manifest(root, manifest):
    for name, expected in manifest.items():
        path = root/name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f'Frozen artifact failed SHA-256 verification: {name}')

def load_aligned(root):
    tests = pd.concat([read(root/f'splits/fold_{f}_test.csv') for f in range(1,6)], ignore_index=True).sort_values('sample_id').reset_index(drop=True)
    assert len(tests) == 3064 and tests.PID.nunique() == 233 and not tests.sample_id.duplicated().any()
    assert tests.groupby('PID').fold.nunique().max() == 1
    # Confirm the frozen patient-disjoint train/validation/test design, without generating splits.
    for f in range(1,6):
        groups = [set(read(root/f'splits/fold_{f}_{s}.csv').PID) for s in ['train','val','test']]
        assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    frames, report = {}, []
    for model in MODELS:
        folder, patient_path, foldname, receiptname = source_paths(root, model)
        slices = read(folder/'oof_predictions.csv').sort_values('sample_id').reset_index(drop=True)
        patients = read(patient_path).sort_values('PID').reset_index(drop=True)
        assert_frame(slices)
        assert_frame(patients, True)
        for col in ['sample_id','PID','fold','file_path']:
            np.testing.assert_array_equal(slices[col], tests[col], err_msg=f'{model} {col} alignment')
        np.testing.assert_array_equal(slices.true_label, tests.original_label.to_numpy(int)-1)
        assert slices.groupby('PID').true_label.nunique().max() == 1
        assert len(patients)==233
        maxfold = 0.
        for f in range(1,6):
            path = folder/f'fold_{f}'/foldname
            receipt = json.loads((folder/f'fold_{f}'/receiptname).read_text())
            assert receipt['status']=='complete' and receipt['inference_passes']==1
            assert sha(path)==receipt['predictions_sha256']
            raw = read(path).sort_values('sample_id').reset_index(drop=True)
            pooled = slices.loc[slices.fold==f].reset_index(drop=True)
            for col in ['sample_id','PID','true_label','predicted_label','fold','file_path']:
                np.testing.assert_array_equal(raw[col], pooled[col])
            diff = float(np.max(np.abs(raw[PROBS].to_numpy()-pooled[PROBS].to_numpy())))
            assert diff <= 3e-16, (model, f, diff)
            maxfold=max(maxfold,diff)
        grouped = slices.groupby('PID', sort=True)
        mean = grouped[PROBS].mean()
        np.testing.assert_array_equal(mean.index, patients.PID)
        np.testing.assert_allclose(mean.to_numpy(), patients[PROBS].to_numpy(), atol=1e-15, rtol=0)
        np.testing.assert_array_equal(mean.to_numpy().argmax(axis=1), patients.predicted_label)
        np.testing.assert_array_equal(grouped.size().to_numpy(), patients.n_slices)
        for col in ['true_label','fold']:
            np.testing.assert_array_equal(grouped[col].first().to_numpy(), patients[col])
        frames[model] = dict(slice=slices, patient=patients)
        if model!='A0':
            for col in ['PID','fold','true_label','n_slices']:
                np.testing.assert_array_equal(patients[col], frames['A0']['patient'][col])
        report.append(dict(model=model, slices=len(slices), patients=len(patients), max_fold_probability_serialization_delta=maxfold,
                           max_patient_mean_delta=float(np.max(np.abs(mean.to_numpy()-patients[PROBS].to_numpy()))), alignment='PASS', receipts_verified=5))
    # Verify that the Phase 13 A5 reference is the same frozen Phase 7/10 model.
    for level, fname, key in [('slice','oof_predictions.csv','sample_id'),('patient','patient_level_predictions.csv','PID')]:
        ref = read(root/'results/ablation/evaluation/A5_reference'/fname).sort_values(key).reset_index(drop=True)
        original = frames['A5'][level]
        for col in [key,'true_label','predicted_label','fold']:
            np.testing.assert_array_equal(ref[col], original[col])
        np.testing.assert_allclose(ref[PROBS], original[PROBS], atol=3e-16, rtol=0)
    return frames, pd.DataFrame(report)

def primary_tests(frames, men=False):
    rows = []
    for left,right in PAIRS:
        l,r = frames[left]['patient'], frames[right]['patient']
        mask = l.true_label==0 if men else np.ones(len(l),bool)
        row = dict(left=left,right=right,n=int(np.sum(mask)))
        row.update(mcnemar((l.predicted_label==l.true_label)[mask], (r.predicted_label==r.true_label)[mask]))
        row['effect_pp'] = 100*(row['left_only']-row['right_only'])/row['n']
        rows.append(row)
    df=pd.DataFrame(rows)
    df['p_holm']=holm(df.p_exact)
    df['significant_holm_0.05']=df.p_holm<=.05
    return df

def with_reverse(df):
    reverse = df.loc[(df.left=='A4') & (df.right=='A5')].copy()
    reverse['left'], reverse['right']='A5','A4'
    for a,b in [('left_only','right_only'), ('left_accuracy_pct','right_accuracy_pct')]:
        if a in reverse:
            reverse[a], reverse[b] = reverse[b].copy(), reverse[a].copy()
    if 'effect_pp' in reverse:
        reverse['effect_pp'] = -reverse.effect_pp
    if 'ci_low_pp' in reverse:
        reverse['ci_low_pp'], reverse['ci_high_pp']=-reverse.ci_high_pp.copy(),-reverse.ci_low_pp.copy()
    return pd.concat([df,reverse],ignore_index=True)

def markdown_table(df):
    cols=list(df.columns)
    rows=['| '+' | '.join(map(str,cols))+' |','|'+'|'.join(['---']*len(cols))+'|']
    rows += ['| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def figures(out, paired, tests, folds, men):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    folder=out/'figures'; folder.mkdir(exist_ok=True)
    for level in ['patient','slice']:
        fig,axes=plt.subplots(1,3,figsize=(14,6),sharey=True,layout='constrained')
        for ax,metric in zip(axes,METRICS):
            d=paired[(paired.level==level)&(paired.metric==metric)].iloc[:10]
            y=np.arange(len(d))
            for i,row in enumerate(d.itertuples()):
                color='#1565a8' if i<6 else '#bb641a'
                ax.plot([row.ci_low_pp,row.ci_high_pp],[i,i],color=color,lw=1.7)
                ax.plot(row.effect_pp,i,'o',color=color,ms=5)
            ax.axvline(0,color='gray',ls='--',lw=1)
            ax.set_yticks(y,[f'{DISPLAY.get(r.left,r.left)} - {DISPLAY.get(r.right,r.right)}' for r in d.itertuples()])
            ax.set_title(metric); ax.set_xlabel('Paired difference (percentage points)'); ax.grid(axis='x',alpha=.2)
        axes[0].invert_yaxis()
        fig.suptitle(f'{level.title()} metrics: paired 95% percentile intervals (10,000 patient draws)\nIntervals are marginal; significance uses Holm-adjusted patient McNemar tests',fontsize=12)
        for ext in ['png','pdf']: fig.savefig(folder/f'{level}_paired_effects.{ext}',dpi=350,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,level in zip(axes,['patient','slice']):
        d=folds[folds.level==level].pivot(index='comparison',columns='fold',values='effect_pp').reindex([f'{a} - {b}' for a,b in PAIRS])
        vmax=max(5,float(np.abs(d.to_numpy()).max()))
        im=ax.imshow(d,cmap='RdBu',vmin=-vmax,vmax=vmax,aspect='auto')
        ax.set_xticks(range(5),[f'Fold {f}' for f in range(1,6)])
        ax.set_yticks(range(10),[s.replace('mobilenet_v3_small','MobileNetV3-Small').replace('efficientnet_b0','EfficientNet-B0').replace('densenet121','DenseNet121').replace('resnet18','ResNet18').replace('resnet50','ResNet50') for s in d.index])
        for i in range(10):
            for j in range(5): ax.text(j,i,f'{d.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=9)
        ax.set_title(f'{level.title()} accuracy difference (pp)'); fig.colorbar(im,ax=ax,shrink=.6)
    fig.suptitle('Fold robustness: descriptive differences, not independent replications')
    for ext in ['png','pdf']: fig.savefig(folder/f'fold_robustness.{ext}',dpi=350,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True,layout='constrained')
    for ax,level in zip(axes,['patient','slice']):
        d=men[men.level==level].iloc[:10]
        for i,r in enumerate(d.itertuples()):
            ax.plot([r.ci_low_pp,r.ci_high_pp],[i,i],color='#1565a8');ax.plot(r.effect_pp,i,'o',color='#1565a8')
        ax.set_yticks(range(10),[f'{DISPLAY.get(r.left,r.left)} - {DISPLAY.get(r.right,r.right)}' for r in d.itertuples()])
        ax.axvline(0,color='gray',ls='--');ax.set_title(f'{level.title()} Meningioma recall');ax.set_xlabel('Paired difference (pp)')
    axes[0].invert_yaxis();fig.suptitle('Exploratory Meningioma subgroup: 82 patients / 708 slices; 95% marginal intervals')
    for ext in ['png','pdf']:fig.savefig(folder/f'meningioma_paired_recall.{ext}',dpi=350,bbox_inches='tight')
    plt.close(fig)

METHODS = '''# Phase 14: frozen-prediction statistical analysis

All results are conditional on the existing, fixed out-of-fold predictions: 233 patients, 3064 slices, five patient-disjoint held-out folds, 11 models. A5 is the original full MorphoNet-BT model. No inference, model loading, training, checkpoint selection, split generation or preprocessing is performed.

Primary endpoint: patient accuracy, based on argmax of each patient's mean slice probabilities. Exact two-sided McNemar tests compare paired correctness, using Binomial(n_discordant, 0.5), including p=1 for no discordances. Holm correction controls the family of 10 unique comparisons (A4 vs A5/five baselines and four additional adjacent ablations). A5-A4 reverses A4-A5 and is displayed without adding a duplicate hypothesis. Differences are left minus right, in percentage points; AUC is multiplied by 100 only for presentation.

Unstratified, paired patient bootstrap: 10,000 shared draws of 233 patient IDs with replacement, NumPy default_rng seed 42. All models and both levels use the identical draws. Slice statistics retain all slices from each drawn patient, including multiplicities; they are slice-weighted estimands with patient clustering. Patient statistics weight each patient equally. Accuracy, three-class macro-F1 and macro one-vs-rest AUC are recalculated in every replicate. Weighted tied-score rank AUC is validated against sklearn and explicit repeated-cluster samples. Percentile 2.5/97.5 intervals use linear interpolation. AUC replicates missing any class are marked invalid, excluded from its percentile interval, and counted. Intervals are marginal, not multiplicity-adjusted; no p-values are inferred from bootstrap sign counts.

The exploratory Meningioma analysis conditions on the 82 truly Meningioma patients (708 slices), reports recall and paired patient/cluster recall intervals, and uses a separate 10-test Holm family for patient McNemar. Its shared 82-patient draws continue the same seeded random stream. Per-fold patient and slice accuracy differences are descriptive only; folds are not treated as five independent experimental replications.

Limitations: A4 was highlighted after observing Phase 13 outcomes; these are exploratory post-selection analyses, not preregistered confirmatory evidence. Holm does not repair outcome-driven selection or cover unknown prior comparisons. These conditional prediction analyses omit training/checkpoint randomness; overlapping cross-validation training sets induce dependence not captured by ordinary patient resampling or McNemar. Patient independence and representative cohort sampling remain assumptions. No external cohort or prospective clinical validation is supplied. Slice and patient endpoints differ in weighting; subgroup analysis has only 82 patients. Non-significance does not establish equivalence. Bootstrap intervals for F1/AUC are supplementary and unadjusted. Sequential effects measure each added component in that specific context, not independent causal contributions or a factorial interaction analysis. No new training experiments are performed.

References: [SciPy exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html), [McNemar exact-test definition](https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html), [Holm multiple testing](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html), [paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
'''

def run(root, settings=None):
    root=Path(root).resolve(); settings=SETTINGS if settings is None else settings
    assert settings==SETTINGS, 'Frozen Phase 14 controls differ from authorized settings.'
    out=root/'results/statistical_analysis';out.mkdir(parents=True,exist_ok=True)
    code_hash=sha(Path(__file__))
    receipt_path=out/'completion_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        assert receipt['settings']==settings and receipt['analysis_source_sha256']==code_hash
        verify_manifest(root, receipt['input_hashes'])
        verify_manifest(root, receipt['output_hashes'])
        print('RESUME_ANALYSIS: completed results and input/output hashes verified; no recomputation.')
        return out
    synthetic=validate_synthetic();dump(out/'synthetic_validation.json',synthetic)
    print('Synthetic statistical validation: PASS',flush=True)
    frozen=json.loads((root/'results/ablation/evaluation/protected_artifacts.json').read_text())
    # Include Phase 13 itself, which did not exist in its own pre-evaluation snapshot.
    for base in ['results/ablation/evaluation','results/baselines/evaluation','results/main','splits','configs']:
        for path in (root/base).rglob('*'):
            if path.is_file():
                rel=path.relative_to(root).as_posix()
                if rel not in frozen:frozen[rel]=sha(path)
    print(f'Checking {len(frozen)} frozen artifact hashes before analysis...',flush=True)
    verify_manifest(root,frozen)
    dump(out/'frozen_input_sha256.json',frozen)
    frames,alignment=load_aligned(root);save(alignment,out/'alignment_audit.csv')
    print('All 11 models aligned: 233 patients, 3064 slices; all 55 prediction receipts verified.',flush=True)
    pids=frames['A0']['patient'].PID.tolist()
    rng=np.random.default_rng(settings['RANDOM_SEED'])
    counts=draws(len(pids),settings['BOOTSTRAP_ITERATIONS'],rng)
    menpids=frames['A0']['patient'].loc[lambda d:d.true_label==0,'PID'].tolist()
    mencounts=draws(len(menpids),settings['BOOTSTRAP_ITERATIONS'],rng)
    np.savez_compressed(out/'shared_patient_draws.npz',pids=np.array(pids),counts=counts,meningioma_pids=np.array(menpids),meningioma_counts=mencounts)
    replicates={};points={};absolute=[];invalid={};menreps={};menpoints={};menabsolute=[]
    for model in MODELS:
        for level in ['patient','slice']:
            frame=frames[model][level];key=f'{model}__{level}'
            reps=bootstrap_metrics(frame,pids,counts);estimate=point(frame)
            replicates[key],points[key]=reps,estimate
            for k,metric in enumerate(METRICS):
                lo,hi=interval(reps[:,k])*100
                invalid[f'{key}__{metric}']=int(np.isnan(reps[:,k]).sum())
                absolute.append(dict(model=model,level=level,metric=metric,estimate_pct=100*estimate[k],ci_low_pct=lo,ci_high_pct=hi,valid_replicates=int(np.isfinite(reps[:,k]).sum())))
            men=frame.loc[frame.true_label==0]
            grouped=men.assign(correct=men.predicted_label==0).groupby('PID').agg(correct=('correct','sum'),total=('correct','size')).reindex(menpids)
            mr=(mencounts@grouped.correct.to_numpy(float))/(mencounts@grouped.total.to_numpy(float))
            mp=float((men.predicted_label==0).mean())
            menreps[key],menpoints[key]=mr,mp
            lo,hi=100*interval(mr)
            menabsolute.append(dict(model=model,level=level,n=len(men),correct=int((men.predicted_label==0).sum()),recall_pct=100*mp,ci_low_pct=lo,ci_high_pct=hi))
        print(f'{model}: 10,000 paired patient and cluster bootstrap replicates complete.',flush=True)
    np.savez_compressed(out/'bootstrap_metric_replicates.npz',**replicates)
    np.savez_compressed(out/'meningioma_recall_replicates.npz',**menreps)
    tests=primary_tests(frames);mentests=primary_tests(frames,True)
    paired=[];menpaired=[];folds=[]
    for left,right in PAIRS:
        for level in ['patient','slice']:
            lk,rk=f'{left}__{level}',f'{right}__{level}'
            for k,metric in enumerate(METRICS):
                delta=100*(replicates[lk][:,k]-replicates[rk][:,k]);lo,hi=interval(delta)
                paired.append(dict(left=left,right=right,level=level,metric=metric,effect_pp=100*(points[lk][k]-points[rk][k]),ci_low_pp=lo,ci_high_pp=hi,valid_replicates=int(np.isfinite(delta).sum())))
            lo,hi=100*interval(menreps[lk]-menreps[rk])
            menpaired.append(dict(left=left,right=right,level=level,effect_pp=100*(menpoints[lk]-menpoints[rk]),ci_low_pp=lo,ci_high_pp=hi))
            l,r=frames[left][level],frames[right][level]
            for f in range(1,6):
                mask=l.fold==f
                la=float((l.loc[mask,'true_label']==l.loc[mask,'predicted_label']).mean())
                ra=float((r.loc[mask,'true_label']==r.loc[mask,'predicted_label']).mean())
                folds.append(dict(comparison=f'{left} - {right}',left=left,right=right,level=level,fold=f,n=int(mask.sum()),left_accuracy_pct=100*la,right_accuracy_pct=100*ra,effect_pp=100*(la-ra)))
    paired=with_reverse(pd.DataFrame(paired));menpaired=with_reverse(pd.DataFrame(menpaired));tests=with_reverse(tests);mentests=with_reverse(mentests)
    folds=with_reverse(pd.DataFrame(folds));folds['comparison']=folds.left+' - '+folds.right
    robustness=folds.groupby(['left','right','level'],sort=False).effect_pp.agg(positive_folds=lambda v:int((v>1e-10).sum()),tied_folds=lambda v:int((v.abs()<=1e-10).sum()),negative_folds=lambda v:int((v < -1e-10).sum()),mean_fold_difference_pp='mean',fold_difference_sd_pp='std',min_fold_difference_pp='min',max_fold_difference_pp='max').reset_index()
    tables={'model_metrics_with_ci':pd.DataFrame(absolute),'paired_effects_with_ci':paired,'exact_mcnemar_holm':tests,'meningioma_absolute_recall':pd.DataFrame(menabsolute),'meningioma_paired_recall':menpaired,'meningioma_exact_mcnemar_holm':mentests,'fold_accuracy_differences':folds,'fold_robustness_summary':robustness}
    for name,df in tables.items():save(df,out/f'{name}.csv')
    save(tests.loc[tests.left.eq('A4')],out/'a4_primary_comparisons.csv')
    sequential=[('A1','A0'),('A2','A1'),('A3','A2'),('A4','A3'),('A5','A4')]
    save(tests.loc[[ (a,b) in sequential for a,b in zip(tests.left,tests.right)]],out/'sequential_ablation_tests.csv')
    figures(out,paired,tests,folds,menpaired)
    report=METHODS+'\n\n## Numerical results\n'
    for name,df in tables.items():
        view=df.copy()
        for col in view.select_dtypes('float').columns:view[col]=view[col].map(lambda v:f'{v:.10g}' if col.startswith('p_') else f'{v:.6f}')
        report+='\n\n### '+name.replace('_',' ').title()+'\n\n'+markdown_table(view)
        (out/f'{name}.tex').write_text(df.to_latex(index=False,float_format=lambda x:f'{x:.6g}'),encoding='utf-8')
    significant=tests.iloc[:10].loc[lambda d:d['significant_holm_0.05']]
    report+='\n\n## Primary conclusion\n\n'+('No primary comparison survives Holm correction.' if significant.empty else 'Primary Holm-significant comparisons: '+', '.join(significant.left+' - '+significant.right))+ '\n\nPatient-level A4 accuracy differences must be interpreted alongside discordance counts and multiplicity-adjusted p-values. Unadjusted bootstrap intervals are not substitutes for Holm decisions.\n'
    (out/'statistical_report.md').write_text(report,encoding='utf-8')
    print('Verifying frozen artifacts again after all analyses and figures...',flush=True)
    verify_manifest(root,frozen)
    audit=dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),settings=settings,models=MODELS,patients=233,slices=3064,meningioma_patients=82,meningioma_slices=708,
               synthetic_validation='PASS',alignment='PASS',verified_prediction_receipts=55,primary_unique_hypotheses=10,holm_family='10 unique primary patient-accuracy comparisons',subgroup_holm_family='10 separate exploratory Meningioma patient-recall comparisons',
               bootstrap_invalid_replicates=invalid,shared_paired_patient_draws=True,all_slices_preserved_in_cluster_bootstrap=True,bootstrap_interval='marginal percentile 95%',training_runs=0,optimizer_steps=0,model_loads=0,inference_passes=0,
               model_weights_changed=False,architecture_changed=False,checkpoints_reselected=False,splits_regenerated=False,preprocessing_changed=False,predictions_modified=False,frozen_artifacts_verified=len(frozen),frozen_artifacts_unchanged=True,
               A5_reference_verified=True,phase15_started=False,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),analysis_source_sha256=code_hash)
    dump(out/'final_scientific_audit.json',audit)
    output_hashes={p.relative_to(root).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['completion_receipt.json','notebook_execution_audit.json']}
    dump(receipt_path,dict(status='complete',settings=settings,analysis_source_sha256=code_hash,input_hashes=frozen,output_hashes=output_hashes))
    print('Phase 14 complete. Frozen-artifact and scientific audits: PASS. Phase 15 not started.',flush=True)
    return out
