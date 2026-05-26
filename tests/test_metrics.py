import math

import numpy as np

from train.metrics import compute_metrics


def test_metrics_binary_normal_case():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    m = compute_metrics(y_true, y_score, threshold=0.5)

    assert np.isfinite(m["auroc"])
    assert np.isfinite(m["auprc"])
    assert m["f1"] == 2 / 3
    assert m["accuracy"] == 0.75
    assert m["acc"] == m["accuracy"]
    assert m["sensitivity"] == 0.5
    assert m["specificity"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [1, 1]]


def test_metrics_specificity_nan_when_no_negatives():
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.2, 0.7, 0.9])
    m = compute_metrics(y_true, y_score, threshold=0.5)
    assert math.isnan(m["specificity"])
    assert m["confusion_matrix"] == [[0, 0], [1, 2]]


def test_metrics_sensitivity_nan_when_no_positives():
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.2, 0.7, 0.9])
    m = compute_metrics(y_true, y_score, threshold=0.5)
    assert math.isnan(m["sensitivity"])
    assert m["confusion_matrix"] == [[1, 2], [0, 0]]


def test_metrics_one_class_sets_auroc_auprc_nan():
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3])
    m = compute_metrics(y_true, y_score, threshold=0.5)
    assert math.isnan(m["auroc"])
    assert math.isnan(m["auprc"])
