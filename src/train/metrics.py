import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, precision_recall_curve


def choose_threshold_max_f1(y_true, y_prob):
    ps, rs, th = precision_recall_curve(y_true, y_prob)
    f1 = 2 * ps * rs / (ps + rs + 1e-12)
    i = int(np.nanargmax(f1))
    return float(th[min(i, len(th)-1)]) if len(th) else 0.5


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "auroc": auroc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }
