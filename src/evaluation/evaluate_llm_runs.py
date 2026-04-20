"""Evaluate LLM prediction results against ground-truth labels."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, compute_metrics_with_bootstrap, compute_rank_curve_payloads
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

    Hard metrics use ``pred_binary`` / ``parsed_prediction`` (Yes/No). Rank metrics
    use ``parsed_probability`` where ``parse_valid_probability`` is true.
    """
    merged = test_df[[id_col, label_col]].merge(pred_df, on=id_col, how="inner")

    pred_map = {"Yes": 1, "No": 0}
    if "pred_binary" not in merged.columns:
        merged["pred_binary"] = merged["parsed_prediction"].map(pred_map)
    else:
        missing_pb = merged["pred_binary"].isna() & merged["parsed_prediction"].notna()
        if missing_pb.any():
            merged.loc[missing_pb, "pred_binary"] = merged.loc[missing_pb, "parsed_prediction"].map(pred_map)

    if "parsed_probability" not in merged.columns:
        merged["parsed_probability"] = np.nan
    if "parse_valid_probability" not in merged.columns:
        merged["parse_valid_probability"] = False
    if "probability_parse_status" not in merged.columns:
        merged["probability_parse_status"] = "missing"
    elif merged["probability_parse_status"].isna().any():
        merged["probability_parse_status"] = merged["probability_parse_status"].fillna("missing")

    n_total = len(merged)
    n_unparseable_pred = int(merged["pred_binary"].isna().sum())
    n_valid_pred = int((~merged["pred_binary"].isna()).sum())

    st = merged["probability_parse_status"]
    n_valid_prob = int(merged["parse_valid_probability"].fillna(False).astype(bool).sum())
    n_missing_prob = int((st == "missing").sum())
    n_invalid_prob = int((st == "invalid").sum())
    n_out_of_range_prob = int((st == "out_of_range").sum())

    logger.info(
        "Mode=%s: n_total=%d valid_pred=%d unparseable_pred=%d valid_prob=%d missing_prob=%d invalid_prob=%d oor_prob=%d",
        mode,
        n_total,
        n_valid_pred,
        n_unparseable_pred,
        n_valid_prob,
        n_missing_prob,
        n_invalid_prob,
        n_out_of_range_prob,
    )

    valid = merged.dropna(subset=["pred_binary"]).copy()
    valid["pred_binary"] = valid["pred_binary"].astype(int)

    y_true = valid[label_col].values
    y_pred = valid["pred_binary"].values

    metrics = compute_metrics(y_true, y_pred)
    metrics["mode"] = mode
    metrics["model"] = valid["model_name"].iloc[0] if len(valid) > 0 and "model_name" in valid.columns else ""
    metrics["n_total"] = n_total
    metrics["n_valid_prediction"] = n_valid_pred
    metrics["n_unparseable_prediction"] = n_unparseable_pred
    metrics["n_valid_probability"] = n_valid_prob
    metrics["n_missing_probability"] = n_missing_prob
    metrics["n_invalid_probability"] = n_invalid_prob
    metrics["n_out_of_range_probability"] = n_out_of_range_prob

    prob_mask = merged["parse_valid_probability"].fillna(False).astype(bool)
    prob_rows = merged.loc[prob_mask].dropna(subset=["pred_binary"])
    rank_note = None
    if len(prob_rows) > 0:
        y_t = prob_rows[label_col].values.astype(int)
        y_s = prob_rows["parsed_probability"].values.astype(float)
        y_p = prob_rows["pred_binary"].astype(int).values
        rank_metrics = compute_metrics(y_t, y_p, y_score=y_s)
        metrics["auc"] = rank_metrics.get("auc")
        metrics["auprc"] = rank_metrics.get("auprc")
        if metrics.get("auc") is None or metrics.get("auprc") is None:
            rank_note = "auc_or_auprc_unavailable"
            logger.info(
                "Mode=%s: rank metrics partial/None (prob_valid=%d, unique_labels=%s)",
                mode,
                len(prob_rows),
                np.unique(y_t).tolist(),
            )
        curves = compute_rank_curve_payloads(y_t, y_s)
        if curves:
            out_p = Path(output_dir)
            save_json(curves["roc"], str(out_p / f"llm_{mode}_roc_curve.json"))
            save_json(curves["pr"], str(out_p / f"llm_{mode}_pr_curve.json"))
    else:
        metrics["auc"] = None
        metrics["auprc"] = None
        rank_note = "no_valid_probabilities"
        logger.info("Mode=%s: no rows with valid parsed probability — AUC/AUPRC omitted", mode)

    if rank_note:
        metrics["rank_metrics_note"] = rank_note

    if bootstrap_ci and len(valid) > 10:
        ci = compute_metrics_with_bootstrap(y_true, y_pred, n_bootstrap=bootstrap_n)
        metrics["bootstrap_ci"] = ci

    merged["true_label"] = merged[label_col]
    merged["split"] = "test"
    save_dataframe(merged, f"{output_dir}/llm_{mode}_results.csv")
    save_json(metrics, f"{output_dir}/llm_{mode}_metrics.json")

    logger.info(
        "Mode=%s metrics: ACC=%.2f P=%.2f Sens=%.2f Spec=%.2f F1=%.2f BalAcc=%.2f AUC=%s AUPRC=%s",
        mode,
        metrics["accuracy"],
        metrics["precision"],
        metrics["sensitivity"],
        metrics["specificity"],
        metrics["f1"],
        metrics["balanced_accuracy"],
        metrics.get("auc"),
        metrics.get("auprc"),
    )
    return metrics
