"""Validation-set threshold search for binary scores (recall / F1 tradeoffs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np

from src.evaluation.metrics import compute_metrics

MetricMode = Literal["f1", "recall"]


@dataclass
class ThresholdPick:
    threshold: float
    metric_name: str
    metric_value: float
    metrics_at_threshold: Dict[str, float]


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    grid: Optional[np.ndarray] = None,
    mode: MetricMode = "f1",
    min_precision: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Return list of dicts with threshold and metrics for each grid point."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    rows: List[Dict[str, float]] = []
    for t in grid:
        y_pred = (y_score >= float(t)).astype(int)
        m = compute_metrics(y_true, y_pred, y_score=y_score)
        if min_precision is not None:
            tp, fp = m["tp"], m["fp"]
            prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if prec < float(min_precision) * 100.0:
                continue
        f1 = m["f1"]
        rec = m["sensitivity"]
        rows.append(
            {
                "threshold": float(t),
                "f1": float(f1),
                "recall": float(rec),
                "precision": float(_precision(m)),
                "accuracy": float(m["accuracy"]),
            }
        )
    return rows


def _precision(m: Dict) -> float:
    tp, fp = m["tp"], m["fp"]
    if tp + fp == 0:
        return 0.0
    return 100.0 * tp / (tp + fp)


def pick_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    mode: MetricMode = "f1",
    min_precision: Optional[float] = None,
) -> ThresholdPick:
    rows = sweep_thresholds(y_true, y_score, mode=mode, min_precision=min_precision)
    if not rows:
        y_pred = (y_score >= 0.5).astype(int)
        full = compute_metrics(y_true, y_pred, y_score=y_score)
        return ThresholdPick(
            threshold=0.5,
            metric_name=mode,
            metric_value=0.0,
            metrics_at_threshold={k: float(v) for k, v in full.items() if isinstance(v, (int, float))},
        )
    key = "f1" if mode == "f1" else "recall"
    best = max(rows, key=lambda r: r[key])
    thr = best["threshold"]
    y_pred = (y_score >= thr).astype(int)
    full = compute_metrics(y_true, y_pred, y_score=y_score)
    return ThresholdPick(
        threshold=thr,
        metric_name=key,
        metric_value=float(best[key]),
        metrics_at_threshold={k: float(v) for k, v in full.items() if isinstance(v, (int, float)) or v is None},
    )
