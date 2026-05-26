import numpy as np

from evaluate.evaluator import aggregate_record_predictions
from train.metrics import compute_metrics


def test_record_level_or_aggregation_uses_max_probability():
    segment_outputs = {
        "record_id": ["a", "a", "b", "b"],
        "segment_idx": [0, 1, 0, 1],
        "y_true": [1, 1, 0, 0],
        "y_prob": [0.2, 0.9, 0.7, 0.1],
    }

    record_outputs = aggregate_record_predictions(segment_outputs)
    metrics = compute_metrics(record_outputs["y_true"], record_outputs["y_prob"], threshold=0.5)

    assert record_outputs["record_id"] == ["a", "b"]
    assert np.allclose(record_outputs["y_prob"], np.array([0.9, 0.7]))
    assert metrics["acc"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["f1"] == 2 / 3
    assert metrics["sensitivity"] == 1.0
    assert "auprc" in metrics
    assert "specificity" in metrics
    assert "confusion_matrix" in metrics
