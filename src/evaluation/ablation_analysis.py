"""Phase 7 metric definitions applied to baseline output directories."""
import pandas as pd
from .analysis import metrics, aggregate_patients, cluster_bootstrap, METRICS
def save_analysis(context, name, oof, frames):
    """Save distinct fold, pooled-slice and pooled-patient result tables."""
    output = context['root'] / f'results/ablation/evaluation/{name}'
    output.mkdir(parents=True, exist_ok=True)
    for fold in frames: (output / f'fold_{fold}').mkdir(exist_ok=True)
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
