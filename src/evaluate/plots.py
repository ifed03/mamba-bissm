from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def save_plots(run_dir, y_true, y_prob, threshold):
    run_dir = Path(run_dir)
    y_pred = (np.array(y_prob) >= threshold).astype(int)

    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.savefig(run_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.savefig(run_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.savefig(run_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(np.array(y_prob), bins=30)
    plt.title("Score Histogram")
    plt.savefig(run_dir / "score_hist.png", dpi=150, bbox_inches="tight")
    plt.close()
