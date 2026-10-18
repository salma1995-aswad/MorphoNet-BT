"""Whole-epoch slice metrics and persistent identity-preserving predictions."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

CLASS_NAMES = ('Meningioma', 'Glioma', 'Pituitary')


def classification_metrics(predictions):
    true = predictions.true_label.to_numpy(dtype=int)
    predicted = predictions.predicted_label.to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(true, predicted, labels=[0,1,2], zero_division=0)
    return dict(accuracy=float(accuracy_score(true, predicted)), macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
                weighted_f1=float(np.average(f1, weights=support)),
                per_class={name: dict(precision=float(precision[i]), recall=float(recall[i]),
                                     f1=float(f1[i]), support=int(support[i])) for i,name in enumerate(CLASS_NAMES)},
                confusion_matrix=confusion_matrix(true, predicted, labels=[0,1,2]).tolist())


def plot_training_curves(log_csv, destination):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = pd.read_csv(log_csv)
    if data.empty:
        raise ValueError('No real epoch history; refusing to fabricate curves')
    figure, axes = plt.subplots(2,3,figsize=(13,7))
    for axis, key in zip(axes.ravel(), ['total_loss','classification_loss','morphology_loss','accuracy','macro_f1','learning_rate']):
        if key == 'learning_rate':
            axis.plot(data.epoch, data.learning_rate, label='LR used')
        else:
            for part in ('train','val'):
                axis.plot(data.epoch, data[f'{part}_{key}'], label=part)
        axis.set_title(key); axis.set_xlabel('Epoch'); axis.legend()
    figure.tight_layout(); figure.savefig(destination, dpi=220); plt.close(figure)
