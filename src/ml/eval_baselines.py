"""Evaluate ML baselines and save results."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    setting: str,
) -> Dict[str, Any]:
    """Evaluate a single model and return a metrics dict."""
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = model_name
    metrics["setting"] = setting
    logger.info("%s (%s): %s", model_name, setting, {k: f"{v:.2f}" for k, v in metrics.items() if isinstance(v, float)})
    return metrics


def evaluate_all(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    setting: str,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Evaluate all models and return/save results DataFrame."""
    rows = []
    for name, model in models.items():
        row = evaluate_model(model, X_test, y_test, name, setting)
        rows.append(row)

    df = pd.DataFrame(rows)
    if output_dir:
        save_dataframe(df, f"{output_dir}/ml_results_{setting}.csv")
    return df
