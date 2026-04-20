"""Evaluation metrics: accuracy, precision, recall, specificity, F1, balanced accuracy.

Positive class = 1 (Disorders of Lipid Metabolism present in next visit).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: int = 1,
    y_score: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : array-like of int (0/1)
    y_pred : array-like of int (0/1)
    positive_label : int — which class is positive
    y_score : optional array of float scores in [0, 1] (e.g. parsed probability of positive class)

    Returns
    -------
    dict with accuracy, precision, recall, sensitivity, specificity, f1, balanced_accuracy,
    confusion_matrix_2x2, tp, fp, tn, fn, and optionally auc, auprc (percent scale 0–100).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = [0, 1] if positive_label == 1 else [1, 0]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0) * 100
    sensitivity = recall  # same as recall for binary positive class
    specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy), 2),
        "precision": round(float(precision), 2),
        "recall": round(float(recall), 2),
        "sensitivity": round(float(sensitivity), 2),
        "specificity": round(float(specificity), 2),
        "f1": round(float(f1), 2),
        "balanced_accuracy": round(float(bal_acc), 2),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        try:
            auc = roc_auc_score(y_true, y_score) * 100
            metrics["auc"] = round(float(auc), 2)
        except Exception as e:
            logger.info("ROC-AUC not computed: %s", e)
            metrics["auc"] = None
        try:
            auprc = average_precision_score(y_true, y_score) * 100
            metrics["auprc"] = round(float(auprc), 2)
        except Exception as e:
            logger.info("AUPRC not computed: %s", e)
            metrics["auprc"] = None
    return metrics


def compute_rank_curve_payloads(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Return serialisable ROC and PR curve data, or None if curves cannot be built."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    try:
        fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
        prec, rec, pr_thresh = precision_recall_curve(y_true, y_score)
        return {
            "roc": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresh.tolist(),
            },
            "pr": {
                "precision": prec.tolist(),
                "recall": rec.tolist(),
                "thresholds": pr_thresh.tolist(),
            },
        }
    except Exception as e:
        logger.info("Curve export skipped: %s", e)
        return None


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

    boot_metrics = {k: [] for k in ["accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "balanced_accuracy"]}

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
