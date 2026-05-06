"""Prepare encoded multitask ML inputs with train/val split and test holdout.

Usage:
    python -m src.scripts.prepare_multitask_ml_inputs \
        --input-dir data/processed/mimiciii_multitask \
        --output-dir data/processed/mimiciii_multitask_ml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

from src.ml.feature_builder import (
    bag_of_codes_to_dataframe,
    fit_bag_of_codes_vectorizer,
    transform_bag_of_codes,
)
from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="data/outputs/prepare_multitask_ml_inputs.log")

DEFAULT_LABEL_COLS = [
    "label_lipid_disorder",
    "label_diabetes_current",
    "label_hypertension_current",
    "label_obesity_current",
    "label_cardio_next",
    "label_kidney_next",
    "label_stroke_next",
]
DEFAULT_ID_COLS = ["pair_id", "SUBJECT_ID", "hadm_id_current", "hadm_id_next", "split"]


def _build_label_distribution(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    for label_col in DEFAULT_LABEL_COLS:
        if label_col not in df.columns:
            continue
        y = pd.to_numeric(df[label_col], errors="coerce")
        n_rows = len(df)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        n_missing = int(y.isna().sum())
        pos_rate = float(n_pos / (n_pos + n_neg)) if (n_pos + n_neg) > 0 else float("nan")
        rows.append(
            {
                "split": split_name,
                "label": label_col,
                "n_rows": n_rows,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_missing": n_missing,
                "positive_rate": round(pos_rate, 6) if pos_rate == pos_rate else pos_rate,
            }
        )
    return pd.DataFrame(rows)


def _build_meta_frame(source_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = pd.DataFrame(index=source_df.index.copy())
    for col in DEFAULT_ID_COLS:
        if col in source_df.columns:
            out[col] = source_df[col].values
    for col in DEFAULT_LABEL_COLS:
        if col in source_df.columns:
            out[col] = source_df[col].astype(int).values
    out["split"] = split_name
    return out


def main(
    input_dir: Path,
    output_dir: Path,
    val_ratio_within_train: float = 0.15,
    split_seed: int = 42,
) -> None:
    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "label_lipid_disorder" not in train_df.columns:
        raise KeyError("label_lipid_disorder column missing from train.csv")

    train_inner, val_inner = train_test_split(
        train_df,
        test_size=val_ratio_within_train,
        random_state=split_seed,
        stratify=train_df["label_lipid_disorder"],
    )
    train_inner = train_inner.reset_index(drop=True).copy()
    val_inner = val_inner.reset_index(drop=True).copy()
    test_df = test_df.reset_index(drop=True).copy()

    vectorizer = fit_bag_of_codes_vectorizer(train_inner)
    x_train = transform_bag_of_codes(train_inner, vectorizer)
    x_val = transform_bag_of_codes(val_inner, vectorizer)
    x_test = transform_bag_of_codes(test_df, vectorizer)

    train_meta = _build_meta_frame(
        train_inner,
        split_name="train_inner",
    )
    val_meta = _build_meta_frame(
        val_inner,
        split_name="val_inner",
    )
    test_meta = _build_meta_frame(
        test_df,
        split_name="test",
    )

    assert len(train_meta) == len(train_inner), "Train row count mismatch after encoding"
    assert len(val_meta) == len(val_inner), "Validation row count mismatch after encoding"
    assert len(test_meta) == len(test_df), "Test row count mismatch after encoding"
    assert x_train.shape[1] == x_val.shape[1] == x_test.shape[1], "Feature schema mismatch across splits"

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_dir / "train_inner_encoded.npz", x_train)
    sparse.save_npz(output_dir / "val_inner_encoded.npz", x_val)
    sparse.save_npz(output_dir / "test_encoded.npz", x_test)
    save_dataframe(train_meta, output_dir / "train_inner_meta.csv")
    save_dataframe(val_meta, output_dir / "val_inner_meta.csv")
    save_dataframe(test_meta, output_dir / "test_meta.csv")

    # Optional small sample CSV for quick inspection.
    preview_rows = min(200, len(train_inner))
    train_preview = _build_meta_frame(train_inner.iloc[:preview_rows], split_name="train_inner")
    train_preview_encoded = bag_of_codes_to_dataframe(x_train[:preview_rows], vectorizer)
    save_dataframe(pd.concat([train_preview.reset_index(drop=True), train_preview_encoded.reset_index(drop=True)], axis=1), output_dir / "train_inner_encoded_preview.csv")

    vocab = vectorizer.get_feature_names_out().tolist()
    save_json(vocab, output_dir / "feature_vocab.json")

    dist = pd.concat(
        [
            _build_label_distribution(train_inner, "train_inner"),
            _build_label_distribution(val_inner, "val_inner"),
            _build_label_distribution(test_df, "test"),
        ],
        ignore_index=True,
    )
    save_dataframe(dist, output_dir / "label_distribution.csv")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "split_seed": split_seed,
        "val_ratio_within_train": val_ratio_within_train,
        "n_rows_train_source": int(len(train_df)),
        "n_rows_test_source": int(len(test_df)),
        "n_rows_train_inner": int(len(train_inner)),
        "n_rows_val_inner": int(len(val_inner)),
        "n_rows_test": int(len(test_df)),
        "n_features": int(len(vocab)),
        "feature_matrix_files": {
            "train_inner": "train_inner_encoded.npz",
            "val_inner": "val_inner_encoded.npz",
            "test": "test_encoded.npz",
        },
        "meta_files": {
            "train_inner": "train_inner_meta.csv",
            "val_inner": "val_inner_meta.csv",
            "test": "test_meta.csv",
        },
        "labels_included": DEFAULT_LABEL_COLS,
        "dropped_duplicate_label": "label_lipid_next",
        "fit_data_for_vectorizer": "train_inner_only",
    }
    save_json(summary, output_dir / "encoding_summary.json")
    logger.info("Saved encoded multitask datasets to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare encoded multitask train/val/test datasets")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/mimiciii_multitask"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mimiciii_multitask_ml"))
    parser.add_argument("--val-ratio-within-train", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    main(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        val_ratio_within_train=args.val_ratio_within_train,
        split_seed=args.split_seed,
    )
