import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def choose_threshold_max_f1(y_true, y_prob):
    ps, rs, th = precision_recall_curve(y_true, y_prob)
    f1 = 2 * ps * rs / (ps + rs + 1e-12)
    i = int(np.nanargmax(f1))
    return float(th[min(i, len(th)-1)]) if len(th) else 0.5


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    has_two_classes = len(np.unique(y_true)) > 1
    auroc = float(roc_auc_score(y_true, y_prob)) if has_two_classes else float("nan")
    auprc = float(average_precision_score(y_true, y_prob)) if has_two_classes else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    accuracy = float(accuracy_score(y_true, y_pred))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": accuracy,
        "acc": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
        "cm": cm.tolist(),
    }
