"""Publication figures from fixed OOF predictions (PNG 400 dpi and vector PDF)."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from .protocol import CLASSES, PROBS


def save_figures(root, model_name, oof, results):
    output = root / f'results/baselines/evaluation/{model_name}/figures'
    output.mkdir(exist_ok=True)
    paths = []
    style = {'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
             'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white', 'axes.spines.top': False,
             'axes.spines.right': False}

    def save(fig, name):
        for suffix in ['png', 'pdf']:
            path = output / f'{name}.{suffix}'
            fig.savefig(path, dpi=400, bbox_inches='tight')
            paths.append(str(path))
        plt.close(fig)

    with plt.rc_context(style):
        for name, matrix, normalized, title in [
                ('oof_confusion_matrix', results['slice_matrix'], False, 'OOF slice-level confusion matrix'),
                ('oof_confusion_matrix_normalized', results['slice_matrix'], True, 'OOF slice-level confusion matrix (%)'),
                ('patient_confusion_matrix', results['patient_matrix'], False, 'Patient-level confusion matrix')]:
            values = matrix.to_numpy()
            if normalized:
                values = values/values.sum(axis=1, keepdims=True)*100
            fig, ax = plt.subplots(figsize=(5.1, 4.5), layout='constrained')
            ax.imshow(values, cmap='Blues', vmin=0, vmax=100 if normalized else values.max())
            for row in range(3):
                for col in range(3):
                    label = f'{values[row,col]:.1f}%' if normalized else str(values[row,col])
                    ax.text(col, row, label, ha='center', va='center', fontsize=12,
                            color='white' if values[row,col] > (100 if normalized else values.max())*.55 else 'black')
            ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
                   xlabel='Predicted class', ylabel='True class', title=title)
            save(fig, name)
        fig, ax = plt.subplots(figsize=(5.3, 4.5), layout='constrained')
        for index, (name, color) in enumerate(zip(CLASSES, ['#0072B2', '#D55E00', '#009E73'])):
            truth = oof.true_label.eq(index)
            fpr, tpr, _ = roc_curve(truth, oof[PROBS[index]])
            auc = roc_auc_score(truth, oof[PROBS[index]])
            ax.plot(fpr, tpr, color=color, lw=1.7, label=f'{name} (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], '--', color='0.6', lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1.015), xlabel='False positive rate', ylabel='True positive rate',
               title=f"OOF one-vs-rest ROC (macro AUC = {results['slice_summary']['Macro ROC-AUC']:.4f})")
        ax.legend(loc='lower right', frameon=False, fontsize=9)
        save(fig, 'oof_roc_curves')
        folds = results['tables']['five_fold_test_results']
        for metric, name in [('Macro F1', 'fold_macro_f1'), ('Accuracy', 'fold_accuracy')]:
            fig, ax = plt.subplots(figsize=(5.3, 3.6), layout='constrained')
            ax.plot(folds.Fold, folds[metric], 'o-', color='#0072B2', lw=1.4)
            ax.axhline(folds[metric].mean(), color='0.4', ls='--', lw=1, label='Five-fold mean')
            for fold, value in zip(folds.Fold, folds[metric]):
                ax.annotate(f'{value:.4f}', (fold, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
            ax.set(xticks=range(1, 6), xlim=(.6, 5.4), ylim=(0, 1.05), xlabel='Outer test fold', ylabel=metric,
                   title=f'Held-out slice-level {metric.lower()}')
            ax.legend(frameon=False, loc='lower left')
            save(fig, name)
    return paths
