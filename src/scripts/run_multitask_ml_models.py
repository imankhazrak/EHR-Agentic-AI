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
from sklearn.metrics import average_precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.metrics import compute_metrics
from src.utils.io import load_json, save_dataframe, save_json
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


def _build_model(model_name: str, seed: int):
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
            class_weight="balanced",
        )
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(
            random_state=seed,
            class_weight="balanced",
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def _predict_one_with_proba(model, x) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(x).astype(int)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] > 1:
            y_prob = prob[:, 1]
        else:
            y_prob = np.zeros(len(y_pred), dtype=float)
    else:
        y_prob = np.zeros(len(y_pred), dtype=float)
    return y_pred, y_prob


def _upsample_positive_class_train_only(x, y: np.ndarray, seed: int):
    """Upsample minority class to parity using only training rows."""
    y = y.astype(int)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return x, y
    if len(pos_idx) == len(neg_idx):
        return x, y
    rng = np.random.default_rng(seed)
    if len(pos_idx) < len(neg_idx):
        sampled = rng.choice(pos_idx, size=(len(neg_idx) - len(pos_idx)), replace=True)
    else:
        sampled = rng.choice(neg_idx, size=(len(pos_idx) - len(neg_idx)), replace=True)
    all_idx = np.concatenate([np.arange(len(y)), sampled])
    rng.shuffle(all_idx)
    return x[all_idx], y[all_idx]


def _select_threshold_with_dual_metric(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Primary metric: AUPRC; tie-breaker: F1 across threshold grid."""
    y_true = y_true.astype(int)
    auprc = float(average_precision_score(y_true, y_prob))
    candidates = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_tuple = (-1.0, -1.0)
    for thr in candidates:
        y_hat = (y_prob >= thr).astype(int)
        f1 = float(f1_score(y_true, y_hat, zero_division=0))
        score = (auprc, f1)
        if score > best_tuple:
            best_tuple = score
            best_thr = float(thr)
    return {
        "threshold": round(best_thr, 4),
        "val_auprc": round(auprc, 4),
        "val_f1_at_threshold": round(best_tuple[1], 4),
    }


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


def _load_task_masks(input_dir: Path) -> dict[str, list[int]]:
    mask_path = input_dir / "task_feature_masks.json"
    mask_obj = load_json(mask_path)
    out: dict[str, list[int]] = {}
    for label in EVAL_LABELS:
        if label not in mask_obj:
            raise KeyError(f"Missing mask for {label} in {mask_path}")
        keep = mask_obj[label].get("keep_indices", [])
        if not keep:
            raise ValueError(f"Empty keep_indices for {label}")
        out[label] = [int(i) for i in keep]
    return out


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

    task_keep_indices = _load_task_masks(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = ["logistic_regression", "decision_tree", "random_forest"]
    all_metrics: list[pd.DataFrame] = []
    threshold_table: dict[str, dict[str, dict[str, float]]] = {}

    for model_name in model_names:
        logger.info("Training multitask model: %s", model_name)
        threshold_table[model_name] = {}
        yhat_train = np.zeros_like(y_train, dtype=int)
        yhat_val = np.zeros_like(y_val, dtype=int)
        yhat_test = np.zeros_like(y_test, dtype=int)
        prob_train = np.zeros_like(y_train, dtype=float)
        prob_val = np.zeros_like(y_val, dtype=float)
        prob_test = np.zeros_like(y_test, dtype=float)

        for j, label in enumerate(EVAL_LABELS):
            keep = task_keep_indices[label]
            x_train_task = x_train[:, keep]
            x_val_task = x_val[:, keep]
            x_test_task = x_test[:, keep]
            x_fit, y_fit = _upsample_positive_class_train_only(
                x_train_task,
                y_train[:, j],
                seed=seed + j,
            )
            model = _build_model(model_name, seed=seed)
            model.fit(x_fit, y_fit)
            _, s_train = _predict_one_with_proba(model, x_train_task)
            _, s_val = _predict_one_with_proba(model, x_val_task)
            _, s_test = _predict_one_with_proba(model, x_test_task)
            thr_info = _select_threshold_with_dual_metric(y_val[:, j], s_val)
            threshold = thr_info["threshold"]
            threshold_table[model_name][label] = {
                **thr_info,
                "train_rows_before_upsample": int(len(y_train[:, j])),
                "train_rows_after_upsample": int(len(y_fit)),
                "upsampling_applied": bool(len(y_fit) > len(y_train[:, j])),
            }
            p_train = (s_train >= threshold).astype(int)
            p_val = (s_val >= threshold).astype(int)
            p_test = (s_test >= threshold).astype(int)
            yhat_train[:, j], prob_train[:, j] = p_train, s_train
            yhat_val[:, j], prob_val[:, j] = p_val, s_val
            yhat_test[:, j], prob_test[:, j] = p_test, s_test

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
        "leakage_safe": True,
        "imbalance_strategy": "hybrid_class_weight_balanced_plus_train_upsampling_plus_threshold_tuning",
        "threshold_selection": {
            "primary_metric": "macro_auprc",
            "tie_breaker": "macro_f1",
            "search_grid": [round(float(x), 2) for x in np.linspace(0.05, 0.95, 19)],
        },
        "thresholds_by_model_and_label": threshold_table,
        "task_feature_masks_file": "task_feature_masks.json",
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
