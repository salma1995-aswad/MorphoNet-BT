"""Metrics and patient-cluster uncertainty from fixed held-out predictions."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from .protocol import CLASSES, PROBS, require

METRICS = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Weighted F1', 'Macro ROC-AUC']


def metrics(frame):
    """Use fixed class order; undefined class metrics propagate as NaN.

    Macro AUC is the unweighted mean of three one-vs-rest AUCs.
    Specificity is TN/(TN+FP) under the same one-vs-rest convention.
    """
    truth = frame.true_label.to_numpy(dtype=int)
    predicted = frame.predicted_label.to_numpy(dtype=int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    rows = []
    for index, name in enumerate(CLASSES):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_count = int(matrix[:, index].sum())
        fp, fn = predicted_count-tp, support-tp
        tn = len(frame)-tp-fp-fn
        divide = lambda numerator, denominator: numerator/denominator if denominator else np.nan
        auc = roc_auc_score(truth == index, frame[PROBS[index]]) if 0 < support < len(frame) else np.nan
        rows.append(dict(Class=name, Precision=divide(tp, predicted_count), Sensitivity=divide(tp, support),
                         Specificity=divide(tn, tn+fp), F1=divide(2*tp, 2*tp+fp+fn),
                         **{'ROC-AUC': auc}, Support=support, PredictedSupport=predicted_count))
    per_class = pd.DataFrame(rows)
    summary = dict(zip(METRICS, [np.trace(matrix)/len(frame), np.mean(per_class.Precision.to_numpy()),
                                np.mean(per_class.Sensitivity.to_numpy()), np.mean(per_class.F1.to_numpy()),
                                np.sum(per_class.F1.to_numpy()*per_class.Support.to_numpy())/len(frame),
                                np.mean(per_class['ROC-AUC'].to_numpy())]))
    return summary, per_class, pd.DataFrame(matrix, index=pd.Index(CLASSES, name='True class'), columns=CLASSES)


def aggregate_patients(oof):
    """Average slice softmax probabilities, weighting every slice equally within PID."""
    require(oof.groupby('PID').true_label.nunique().eq(1).all(), 'Inconsistent patient labels')
    require(oof.groupby('PID').fold.nunique().eq(1).all(), 'Patient spans test folds')
    # Accumulate in float64 both before and after CSV serialization.
    groups = oof.astype({name: 'float64' for name in PROBS}).groupby('PID', sort=True)
    patients = groups[PROBS].mean()
    patients['fold'] = groups.fold.first()
    patients['true_label'] = groups.true_label.first()
    patients['n_slices'] = groups.size()
    patients['predicted_label'] = patients[PROBS].to_numpy().argmax(axis=1)
    for prefix in ['true', 'predicted']:
        patients[f'{prefix}_class'] = [CLASSES[int(i)] for i in patients[f'{prefix}_label']]
    patients = patients.reset_index()
    correct = oof.assign(correct=oof.true_label.eq(oof.predicted_label)).groupby('PID').correct.sum()
    consistency = patients.copy()
    consistency['n_correct'] = consistency.PID.map(correct)
    consistency['slice_accuracy'] = consistency.n_correct/consistency.n_slices
    require(len(patients) == 233 and patients.PID.is_unique, 'Patient count mismatch')
    return patients, consistency


def cluster_bootstrap(consistency, iterations=2000, seed=42):
    """Resample PIDs uniformly, retaining full cluster sizes and multiplicities.

    Summing cluster correct counts and sizes is exactly equivalent to concatenating
    all slices for each selected patient, including repeatedly selected patients.
    This interval conditions on the fixed trained models and OOF predictions.
    """
    counts = consistency.n_slices.to_numpy()
    correct = consistency.n_correct.to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(counts), size=(iterations, len(counts)))
    values = correct[indices].sum(axis=1)/counts[indices].sum(axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    interval = pd.DataFrame([dict(Metric='OOF slice accuracy', Estimate=correct.sum()/counts.sum(),
                                 Lower95=low, Upper95=high, ResamplingUnit='Patient', Iterations=iterations,
                                 Seed=seed, Method='Percentile')])
    return interval, pd.DataFrame({'Iteration': np.arange(1, iterations+1), 'Accuracy': values})


def save_analysis(context, oof, frames):
    """Save distinct fold, pooled-slice and pooled-patient result tables."""
    output = context['root'] / 'results/main'
    fold_rows = []
    for fold, frame in frames.items():
        summary, per_class, matrix = metrics(frame)
        per_class.to_csv(output / f'fold_{fold}/test_per_class_metrics.csv', index=False)
        matrix.to_csv(output / f'fold_{fold}/test_confusion_matrix.csv')
        fold_rows.append({'Fold': fold, 'Test Patients': frame.PID.nunique(), 'Test Slices': len(frame), **summary})
    folds = pd.DataFrame(fold_rows)
    summary, per_class, matrix = metrics(oof)
    final = pd.DataFrame([dict(Metric=name, FoldMean=folds[name].mean(skipna=False), FoldSD=folds[name].std(ddof=1, skipna=False),
                               AggregatedOOFValue=summary[name]) for name in METRICS])
    patients, consistency = aggregate_patients(oof)
    patient_summary, patient_classes, patient_matrix = metrics(patients)
    patient_metrics = pd.DataFrame([{'Metric': name, 'Value': value} for name, value in patient_summary.items()])
    interval, replicates = cluster_bootstrap(consistency)
    mistakes = oof.loc[oof.true_label.ne(oof.predicted_label)].copy()
    patient_mistakes = patients.loc[patients.true_label.ne(patients.predicted_label)].copy()
    pairs = mistakes.groupby(['true_class', 'predicted_class']).size().rename('Count').reset_index().sort_values('Count', ascending=False)
    patient_folds, patient_fold_classes = [], []
    for fold, frame in patients.groupby('fold'):
        values, classes, confusion = metrics(frame)
        patient_folds.append(dict(Fold=fold, Patients=len(frame), **values))
        patient_fold_classes.append(classes.assign(Fold=fold))
        confusion.to_csv(output / f'fold_{fold}/patient_confusion_matrix.csv')
    tables = {'five_fold_test_results': folds, 'final_performance_summary': final, 'oof_per_class_metrics': per_class,
              'patient_level_predictions': patients, 'patient_level_metrics': patient_metrics,
              'patient_level_per_class_metrics': patient_classes, 'patient_consistency': consistency,
              'misclassified_samples': mistakes, 'misclassified_patients': patient_mistakes,
              'misclassification_pairs': pairs, 'accuracy_cluster_bootstrap_ci': interval,
              'bootstrap_replicates': replicates, 'patient_fold_metrics': pd.DataFrame(patient_folds),
              'patient_fold_per_class_metrics': pd.concat(patient_fold_classes, ignore_index=True)}
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False, float_format='%.17g')
    matrix.to_csv(output / 'final_confusion_matrix.csv')
    patient_matrix.to_csv(output / 'patient_level_confusion_matrix.csv')
    return dict(tables=tables, slice_summary=summary, patient_summary=patient_summary,
                slice_matrix=matrix, patient_matrix=patient_matrix)
