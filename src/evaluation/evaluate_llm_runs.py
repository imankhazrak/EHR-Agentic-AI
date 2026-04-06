"""Evaluate LLM prediction results against ground-truth labels."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, compute_metrics_with_bootstrap
from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_llm_results(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    mode: str,
    output_dir: str = "data/outputs",
    label_col: str = "label_lipid_disorder",
    id_col: str = "pair_id",
    bootstrap_ci: bool = False,
    bootstrap_n: int = 1000,
) -> Dict:
    """Merge predictions with true labels and compute metrics.

    Parameters
    ----------
    test_df : DataFrame with true labels
    pred_df : DataFrame from predictor (with parsed_prediction column)
    mode : str — prompt mode name for labelling outputs

    Returns
    -------
    dict with metrics and parse-failure info
    """
    merged = test_df[[id_col, label_col]].merge(pred_df, on=id_col, how="inner")

    # Map parsed prediction to binary
    pred_map = {"Yes": 1, "No": 0}
    merged["pred_binary"] = merged["parsed_prediction"].map(pred_map)

    # Count parse failures
    n_unparseable = merged["pred_binary"].isna().sum()
    n_total = len(merged)
    logger.info("Mode=%s: %d total, %d unparseable (%.1f%%)", mode, n_total, n_unparseable, n_unparseable / n_total * 100 if n_total else 0)

    # Drop unparseable for metric computation
    valid = merged.dropna(subset=["pred_binary"]).copy()
    valid["pred_binary"] = valid["pred_binary"].astype(int)

    y_true = valid[label_col].values
    y_pred = valid["pred_binary"].values

    metrics = compute_metrics(y_true, y_pred)
    metrics["mode"] = mode
    metrics["model"] = valid["model_name"].iloc[0] if len(valid) > 0 else ""
    metrics["n_total"] = n_total
    metrics["n_valid"] = len(valid)
    metrics["n_unparseable"] = int(n_unparseable)

    if bootstrap_ci and len(valid) > 10:
        ci = compute_metrics_with_bootstrap(y_true, y_pred, n_bootstrap=bootstrap_n)
        metrics["bootstrap_ci"] = ci

    # Save per-sample results
    merged["true_label"] = merged[label_col]
    merged["split"] = "test"
    save_dataframe(merged, f"{output_dir}/llm_{mode}_results.csv")
    save_json(metrics, f"{output_dir}/llm_{mode}_metrics.json")

    logger.info("Mode=%s metrics: ACC=%.2f  Sens=%.2f  Spec=%.2f  F1=%.2f",
                mode, metrics["accuracy"], metrics["sensitivity"], metrics["specificity"], metrics["f1"])
    return metrics
