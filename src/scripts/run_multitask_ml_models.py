"""Run multitask ML models on encoded splits and export metrics + predictions.

Usage:
    python -m src.scripts.run_multitask_ml_models \
        --input-dir data/processed/mimiciii_multitask_ml \
        --output-dir data/outputs/mimiciii_multitask_ml_models
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.metrics import compute_metrics
from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="data/outputs/run_multitask_ml_models.log")

EVAL_LABELS = [
    "label_lipid_disorder",
    "label_diabetes_current",
    "label_hypertension_current",
    "label_obesity_current",
    "label_cardio_next",
    "label_kidney_next",
    "label_stroke_next",
]
ALL_OUTPUT_LABELS = [
    "label_lipid_disorder",
    "label_lipid_next",
    "label_diabetes_current",
    "label_hypertension_current",
    "label_obesity_current",
    "label_cardio_next",
    "label_kidney_next",
    "label_stroke_next",
]


def _ensure_lipid_next(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "label_lipid_next" not in out.columns and "label_lipid_disorder" in out.columns:
        out["label_lipid_next"] = out["label_lipid_disorder"].astype(int)
    return out


def _load_split(input_dir: Path, split_name: str) -> tuple[Any, pd.DataFrame]:
    x = sparse.load_npz(input_dir / f"{split_name}_encoded.npz")
    meta = pd.read_csv(input_dir / f"{split_name}_meta.csv")
    meta = _ensure_lipid_next(meta)
    return x, meta


def _build_model(model_name: str, seed: int) -> MultiOutputClassifier:
    if model_name == "logistic_regression":
        base = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    elif model_name == "decision_tree":
        base = DecisionTreeClassifier(random_state=seed)
    elif model_name == "random_forest":
        base = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return MultiOutputClassifier(base)


def _predict_with_proba(model: MultiOutputClassifier, x) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(x)
    prob_list = model.predict_proba(x)
    y_prob = np.zeros_like(y_pred, dtype=float)
    for j, cls_prob in enumerate(prob_list):
        # MultiOutputClassifier returns per-task probability arrays.
        if cls_prob.shape[1] == 1:
            y_prob[:, j] = 0.0
        else:
            y_prob[:, j] = cls_prob[:, 1]
    return y_pred, y_prob


def _evaluate_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for j, label_name in enumerate(labels):
        m = compute_metrics(y_true[:, j], y_pred[:, j], y_score=y_prob[:, j])
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "label": label_name,
                "accuracy": m["accuracy"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "auc": m.get("auc"),
                "auprc": m.get("auprc"),
                "tp": m["tp"],
                "fp": m["fp"],
                "tn": m["tn"],
                "fn": m["fn"],
            }
        )
    df = pd.DataFrame(rows)
    macro = {
        "model": model_name,
        "split": split_name,
        "label": "macro_avg",
        "accuracy": round(float(df["accuracy"].mean()), 2),
        "sensitivity": round(float(df["sensitivity"].mean()), 2),
        "specificity": round(float(df["specificity"].mean()), 2),
        "f1": round(float(df["f1"].mean()), 2),
        "auc": round(float(df["auc"].dropna().mean()), 2) if df["auc"].notna().any() else None,
        "auprc": round(float(df["auprc"].dropna().mean()), 2) if df["auprc"].notna().any() else None,
        "tp": int(df["tp"].sum()),
        "fp": int(df["fp"].sum()),
        "tn": int(df["tn"].sum()),
        "fn": int(df["fn"].sum()),
    }
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def _build_prediction_frame(
    meta_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    eval_labels: list[str],
) -> pd.DataFrame:
    out = _ensure_lipid_next(meta_df).copy()
    pred_map: dict[str, tuple[np.ndarray, np.ndarray]] = {
        label: (y_pred[:, idx], y_prob[:, idx]) for idx, label in enumerate(eval_labels)
    }
    # Duplicate lipid_next outputs from lipid_disorder task to provide 8 task columns.
    pred_map["label_lipid_next"] = pred_map["label_lipid_disorder"]
    for label in ALL_OUTPUT_LABELS:
        pred_vals, prob_vals = pred_map[label]
        out[f"pred_{label}"] = pred_vals.astype(int)
        out[f"prob_{label}"] = prob_vals.astype(float)
    return out


def main(input_dir: Path, output_dir: Path, seed: int = 42) -> None:
    x_train, train_meta = _load_split(input_dir, "train_inner")
    x_val, val_meta = _load_split(input_dir, "val_inner")
    x_test, test_meta = _load_split(input_dir, "test")

    y_train = train_meta[EVAL_LABELS].astype(int).values
    y_val = val_meta[EVAL_LABELS].astype(int).values
    y_test = test_meta[EVAL_LABELS].astype(int).values

    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = ["logistic_regression", "decision_tree", "random_forest"]
    all_metrics: list[pd.DataFrame] = []

    for model_name in model_names:
        logger.info("Training multitask model: %s", model_name)
        model = _build_model(model_name, seed=seed)
        model.fit(x_train, y_train)

        yhat_train, prob_train = _predict_with_proba(model, x_train)
        yhat_val, prob_val = _predict_with_proba(model, x_val)
        yhat_test, prob_test = _predict_with_proba(model, x_test)

        train_metrics = _evaluate_split(y_train, yhat_train, prob_train, EVAL_LABELS, model_name, "train_inner")
        val_metrics = _evaluate_split(y_val, yhat_val, prob_val, EVAL_LABELS, model_name, "val_inner")
        test_metrics = _evaluate_split(y_test, yhat_test, prob_test, EVAL_LABELS, model_name, "test")
        model_metrics = pd.concat([train_metrics, val_metrics, test_metrics], ignore_index=True)
        save_dataframe(model_metrics, output_dir / f"multitask_metrics_{model_name}.csv")
        all_metrics.append(model_metrics)

        save_dataframe(
            _build_prediction_frame(train_meta, yhat_train, prob_train, EVAL_LABELS),
            output_dir / f"train_inner_with_predictions_{model_name}.csv",
        )
        save_dataframe(
            _build_prediction_frame(val_meta, yhat_val, prob_val, EVAL_LABELS),
            output_dir / f"val_inner_with_predictions_{model_name}.csv",
        )
        save_dataframe(
            _build_prediction_frame(test_meta, yhat_test, prob_test, EVAL_LABELS),
            output_dir / f"test_with_predictions_{model_name}.csv",
        )

    combined = pd.concat(all_metrics, ignore_index=True)
    save_dataframe(combined, output_dir / "multitask_metrics_all_models.csv")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "models": model_names,
        "eval_labels": EVAL_LABELS,
        "prediction_output_labels": ALL_OUTPUT_LABELS,
        "prediction_columns_added_per_row": 2 * len(ALL_OUTPUT_LABELS),
        "splits": {"train_inner": int(len(train_meta)), "val_inner": int(len(val_meta)), "test": int(len(test_meta))},
    }
    save_json(summary, output_dir / "multitask_run_summary.json")
    logger.info("Multitask ML run complete: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multitask ML models and export metrics/predictions")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/mimiciii_multitask_ml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/mimiciii_multitask_ml_models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(input_dir=args.input_dir.resolve(), output_dir=args.output_dir.resolve(), seed=args.seed)
