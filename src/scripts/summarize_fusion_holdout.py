"""Build a single CSV comparison table from fusion metrics JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.utils.io import save_dataframe


def _row_from_metrics(path: Path, train_score_mode: str) -> Dict[str, Any]:
    j = json.loads(path.read_text(encoding="utf-8"))
    tm = j.get("threshold_tuned_metrics") or {}
    fam = str(j.get("learner", ""))
    if fam == "logistic_regression":
        family = "LogisticRegression"
    elif fam == "mlp":
        family = "MLP"
    elif fam == "hybrid_cnn_transformer":
        family = "Hybrid_CNN_Transformer"
    else:
        family = fam
    reg = str(j.get("feature_regime", ""))
    feat = reg.replace("_", " ").title().replace(" ", "_")
    return {
        "train_llm_score_mode": train_score_mode,
        "model_family": family,
        "feature_set": reg,
        "threshold_val_tuned": j.get("val_threshold"),
        "accuracy": tm.get("accuracy"),
        "recall_sensitivity": tm.get("sensitivity") or tm.get("recall"),
        "specificity": tm.get("specificity"),
        "precision": tm.get("precision"),
        "f1": tm.get("f1"),
        "auc": j.get("auc"),
        "auprc": j.get("auprc"),
        "tp": tm.get("tp"),
        "fp": tm.get("fp"),
        "tn": tm.get("tn"),
        "fn": tm.get("fn"),
        "source_metrics": str(path),
    }


def _baseline_rows(exp_root: Path, ml_csv: Path | None, llm_csv: Path | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if ml_csv and ml_csv.exists():
        mdf = pd.read_csv(ml_csv)
        for _, r in mdf.iterrows():
            rows.append(
                {
                    "train_llm_score_mode": "n/a_ml_baseline",
                    "model_family": str(r.get("model", "")),
                    "feature_set": "structured_ml_baseline",
                    "threshold_val_tuned": None,
                    "accuracy": r.get("accuracy"),
                    "recall_sensitivity": r.get("sensitivity") or r.get("recall"),
                    "specificity": r.get("specificity"),
                    "precision": r.get("precision"),
                    "f1": r.get("f1"),
                    "auc": r.get("auc"),
                    "auprc": r.get("auprc"),
                    "tp": r.get("tp"),
                    "fp": r.get("fp"),
                    "tn": r.get("tn"),
                    "fn": r.get("fn"),
                    "source_metrics": str(ml_csv),
                }
            )
    if llm_csv and llm_csv.exists():
        df = pd.read_csv(llm_csv)
        if {"true_label", "pred_binary", "prob_yes"}.issubset(df.columns):
            y = df["true_label"].to_numpy()
            s = df["prob_yes"].to_numpy()
            m_hard = compute_metrics(y, df["pred_binary"].to_numpy(), y_score=s)
            pred_p = (s >= 0.5).astype(int)
            m_prob = compute_metrics(y, pred_p, y_score=s)
            rows.append(
                {
                    "train_llm_score_mode": "test_only_llm_hard",
                    "model_family": "LLM_Gemma_finetuned",
                    "feature_set": "llm_hard_decode",
                    "threshold_val_tuned": None,
                    "accuracy": m_hard.get("accuracy"),
                    "recall_sensitivity": m_hard.get("sensitivity"),
                    "specificity": m_hard.get("specificity"),
                    "precision": m_hard.get("precision"),
                    "f1": m_hard.get("f1"),
                    "auc": m_hard.get("auc"),
                    "auprc": m_hard.get("auprc"),
                    "tp": m_hard.get("tp"),
                    "fp": m_hard.get("fp"),
                    "tn": m_hard.get("tn"),
                    "fn": m_hard.get("fn"),
                    "source_metrics": str(llm_csv),
                }
            )
            rows.append(
                {
                    "train_llm_score_mode": "test_only_llm_prob",
                    "model_family": "LLM_Gemma_finetuned",
                    "feature_set": "llm_prob_ge_0.5",
                    "threshold_val_tuned": 0.5,
                    "accuracy": m_prob.get("accuracy"),
                    "recall_sensitivity": m_prob.get("sensitivity"),
                    "specificity": m_prob.get("specificity"),
                    "precision": m_prob.get("precision"),
                    "f1": m_prob.get("f1"),
                    "auc": m_prob.get("auc"),
                    "auprc": m_prob.get("auprc"),
                    "tp": m_prob.get("tp"),
                    "fp": m_prob.get("fp"),
                    "tn": m_prob.get("tn"),
                    "fn": m_prob.get("fn"),
                    "source_metrics": str(llm_csv),
                }
            )
    return rows


def main(
    experiment_dir: Path,
    ml_csv: Path | None,
    llm_csv: Path | None,
) -> Path:
    rows: List[Dict[str, Any]] = []
    for mode, sub in (("infold_train_llm", "fusion_infold"), ("oof_train_llm", "fusion_oof")):
        root = experiment_dir / sub
        if not root.exists():
            continue
        for p in sorted(root.glob("metrics_*.json")):
            rows.append(_row_from_metrics(p, mode))
    rows.extend(_baseline_rows(experiment_dir, ml_csv, llm_csv))
    out = experiment_dir / "fusion_final_comparison_table.csv"
    save_dataframe(pd.DataFrame(rows), out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-dir", type=Path, required=True)
    ap.add_argument("--ml-csv", type=Path, default=None)
    ap.add_argument("--llm-csv", type=Path, default=None)
    args = ap.parse_args()
    p = main(args.experiment_dir, args.ml_csv, args.llm_csv)
    print(p)
