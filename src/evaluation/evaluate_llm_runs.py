"""Evaluate LLM prediction results against ground-truth labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, compute_metrics_with_bootstrap, compute_rank_curve_payloads
from src.llm.output_parser import parse_multitask_output_with_meta
from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_raw_response_map(jsonl_path: str) -> Dict[int, str]:
    """Load sample_id -> raw text from predictor JSONL."""
    out: Dict[int, str] = {}
    p = Path(jsonl_path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("sample_id")
            resp = obj.get("text")
            if sid is None or resp is None:
                continue
            try:
                out[int(sid)] = str(resp)
            except Exception:
                continue
    return out


def _maybe_salvage_multitask_rows(merged: pd.DataFrame) -> pd.DataFrame:
    """Backfill multitask parse failures from raw_response without rerunning LLM calls."""
    if "raw_response" not in merged.columns:
        return merged
    if "parser_status" not in merged.columns:
        merged["parser_status"] = ""

    mt_required = [
        "lipid_prob",
        "lipid_pred",
        "diabetes_prob",
        "diabetes_pred",
        "hypertension_prob",
        "hypertension_pred",
        "obesity_prob",
        "obesity_pred",
        "cardio_prob",
        "cardio_pred",
        "kidney_prob",
        "kidney_pred",
        "stroke_prob",
        "stroke_pred",
    ]
    for c in mt_required:
        if c not in merged.columns:
            merged[c] = np.nan
    if "parse_failure_reason" not in merged.columns:
        merged["parse_failure_reason"] = ""
    if "salvage_used" not in merged.columns:
        merged["salvage_used"] = False
    if "n_tasks_salvaged" not in merged.columns:
        merged["n_tasks_salvaged"] = 0

    salvage_mask = merged["pred_binary"].isna() | merged["parser_status"].astype(str).str.contains("multitask_parse_failed", na=False)
    # If CSV raw_response is corrupted by malformed quoting, fallback to raw JSONL output.
    raw_map: Dict[int, str] = {}
    if ("raw_response_path" in merged.columns) and (id_col := "pair_id") in merged.columns:
        path_vals = merged["raw_response_path"].dropna().astype(str)
        if len(path_vals) > 0:
            raw_map = _load_raw_response_map(path_vals.iloc[0])

    if not salvage_mask.any():
        return merged

    for idx in merged.index[salvage_mask]:
        raw_text = str(merged.at[idx, "raw_response"])
        mt = parse_multitask_output_with_meta(raw_text)
        if (mt.get("n_tasks_salvaged", 0) == 0) and raw_map:
            sid = merged.at[idx, id_col]
            try:
                fallback_text = raw_map.get(int(sid), "")
            except Exception:
                fallback_text = ""
            if fallback_text:
                mt = parse_multitask_output_with_meta(fallback_text)
                if mt.get("n_tasks_salvaged", 0) > 0:
                    merged.at[idx, "raw_response"] = fallback_text
        for c in mt_required:
            if c in mt and pd.notna(mt[c]):
                merged.at[idx, c] = mt[c]
        merged.at[idx, "parse_failure_reason"] = mt.get("parse_failure_reason", "") or ""
        merged.at[idx, "salvage_used"] = bool(mt.get("salvage_used", False))
        merged.at[idx, "n_tasks_salvaged"] = int(mt.get("n_tasks_salvaged", 0))
        merged.at[idx, "parser_status"] = mt.get("parser_status", merged.at[idx, "parser_status"])

        if pd.notna(merged.at[idx, "lipid_pred"]):
            lp = int(merged.at[idx, "lipid_pred"])
            merged.at[idx, "pred_binary"] = lp
            merged.at[idx, "parsed_prediction"] = "Yes" if lp == 1 else "No"
        if pd.notna(merged.at[idx, "lipid_prob"]):
            merged.at[idx, "parsed_probability"] = float(merged.at[idx, "lipid_prob"])
            merged.at[idx, "parse_valid_probability"] = True
            merged.at[idx, "probability_parse_status"] = "ok"

    return merged


def _write_parse_failure_report(merged: pd.DataFrame, mode: str, output_dir: str, id_col: str) -> None:
    """Write CSV/markdown diagnostics for salvaged and failed multitask rows."""
    if "parser_status" not in merged.columns:
        return
    dmask = merged["parser_status"].astype(str).str.startswith("multitask_")
    dmask &= ~merged["parser_status"].astype(str).eq("multitask_json")
    if not dmask.any():
        return

    cols = [id_col, "parser_status", "parse_failure_reason", "n_tasks_salvaged", "pred_binary", "parse_valid_probability", "raw_response_path"]
    cols = [c for c in cols if c in merged.columns]
    diag = merged.loc[dmask, cols].copy()
    diag = diag.sort_values(["parser_status", "parse_failure_reason", id_col], kind="stable")

    out_p = Path(output_dir)
    csv_path = out_p / f"llm_{mode}_parse_failure_report.csv"
    md_path = out_p / f"llm_{mode}_parse_failure_report.md"
    save_dataframe(diag, csv_path)

    reason_counts = (
        diag["parse_failure_reason"].fillna("").replace("", "unspecified").value_counts().rename_axis("reason").reset_index(name="count")
    )
    status_counts = diag["parser_status"].value_counts().rename_axis("status").reset_index(name="count")
    lines = [
        f"# Parse Failure Report ({mode})",
        "",
        f"- Total diagnostics rows: **{len(diag)}**",
        "",
        "## By parser status",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    lines.extend(f"| {r.status} | {int(r.count)} |" for r in status_counts.itertuples(index=False))
    lines.extend(["", "## By failure reason", "", "| reason | count |", "|---|---:|"])
    lines.extend(f"| {r.reason} | {int(r.count)} |" for r in reason_counts.itertuples(index=False))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    merged = _maybe_salvage_multitask_rows(merged)

    n_total = len(merged)
    n_unparseable_pred = int(merged["pred_binary"].isna().sum())
    n_valid_pred = int((~merged["pred_binary"].isna()).sum())
    ps = merged.get("parser_status", pd.Series([""] * len(merged)))
    salvage_mask = ps.astype(str).str.startswith("multitask_salvaged")
    n_salvaged_pred = int((salvage_mask & merged["pred_binary"].notna()).sum())
    n_strict_parse = int((ps.astype(str) == "multitask_json").sum())
    n_reasoning_unclosed = int((merged.get("parse_failure_reason", pd.Series([""] * len(merged))).astype(str) == "reasoning_unclosed").sum())

    st = merged["probability_parse_status"]
    n_valid_prob = int(merged["parse_valid_probability"].fillna(False).astype(bool).sum())
    n_salvaged_prob = int((salvage_mask & merged["parse_valid_probability"].fillna(False).astype(bool)).sum())
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

    # Keep long-running jobs from hard-failing when every row is unparseable.
    # This can happen with truncated generations (e.g., local models under tight token caps).
    if len(valid) == 0:
        logger.warning(
            "Mode=%s: no valid hard predictions after parsing; writing diagnostics-only metrics.",
            mode,
        )
        metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "sensitivity": None,
            "specificity": None,
            "f1": None,
            "balanced_accuracy": None,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "confusion_matrix": {"labels": [0, 1], "matrix": [[0, 0], [0, 0]]},
            "auc": None,
            "auprc": None,
            "rank_metrics_note": "no_valid_predictions",
        }
    else:
        y_true = valid[label_col].values
        y_pred = valid["pred_binary"].values
        metrics = compute_metrics(y_true, y_pred)
    metrics["mode"] = mode
    metrics["model"] = valid["model_name"].iloc[0] if len(valid) > 0 and "model_name" in valid.columns else ""
    metrics["n_total"] = n_total
    metrics["n_valid_prediction"] = n_valid_pred
    metrics["n_unparseable_prediction"] = n_unparseable_pred
    metrics["n_strict_parse"] = n_strict_parse
    metrics["n_salvaged_prediction"] = n_salvaged_pred
    metrics["n_salvaged_probability"] = n_salvaged_prob
    metrics["n_reasoning_unclosed"] = n_reasoning_unclosed
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
        y_true = valid[label_col].values
        y_pred = valid["pred_binary"].values
        ci = compute_metrics_with_bootstrap(y_true, y_pred, n_bootstrap=bootstrap_n)
        metrics["bootstrap_ci"] = ci

    merged["true_label"] = merged[label_col]
    merged["split"] = "test"
    save_dataframe(merged, f"{output_dir}/llm_{mode}_results.csv")
    save_json(metrics, f"{output_dir}/llm_{mode}_metrics.json")
    _write_parse_failure_report(merged, mode=mode, output_dir=output_dir, id_col=id_col)

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
