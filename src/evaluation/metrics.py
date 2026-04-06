"""Evaluation metrics: accuracy, sensitivity, specificity, F1.

Positive class = 1 (Disorders of Lipid Metabolism present in next visit).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> Dict[str, float]:
    """Compute accuracy, sensitivity, specificity, and F1.

    Parameters
    ----------
    y_true : array-like of int (0/1)
    y_pred : array-like of int (0/1)
    positive_label : int — which class is positive

    Returns
    -------
    dict with keys: accuracy, sensitivity, specificity, f1, tp, fp, tn, fn
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    labels = [0, 1] if positive_label == 1 else [1, 0]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred) * 100
    sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0  # recall
    specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0) * 100

    return {
        "accuracy": round(accuracy, 2),
        "sensitivity": round(sensitivity, 2),
        "specificity": round(specificity, 2),
        "f1": round(f1, 2),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_metrics_with_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics with 95% bootstrap confidence intervals.

    Returns dict of metric_name → {mean, ci_lower, ci_upper}.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    boot_metrics = {k: [] for k in ["accuracy", "sensitivity", "specificity", "f1"]}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        m = compute_metrics(y_true[idx], y_pred[idx])
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    result = {}
    base = compute_metrics(y_true, y_pred)
    for k in boot_metrics:
        vals = sorted(boot_metrics[k])
        result[k] = {
            "mean": base[k],
            "ci_lower": round(vals[int(0.025 * n_bootstrap)], 2),
            "ci_upper": round(vals[int(0.975 * n_bootstrap)], 2),
        }
    return result
