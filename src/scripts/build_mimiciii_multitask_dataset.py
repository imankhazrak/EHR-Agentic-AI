"""Build ``data/processed/mimiciii_multitask/{train,test}.csv`` from existing processed splits.

Reads real ICD-9 aggregates from ``data/processed/mimicii/train|test.csv`` (default),
appends multitask label columns via :func:`add_multitask_labels`, and writes to a
**separate** output directory without modifying ``mimiciii/`` or ``mimiciii_full_dataset/``.

Usage:
  python -m src.scripts.build_mimiciii_multitask_dataset \\
      --source-processed data/processed/mimiciii \\
      --output-processed data/processed/mimiciii_multitask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.build_target_labels import add_multitask_labels
from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="data/outputs/build_mimiciii_multitask.log")

STAT_LABELS = [
    "label_lipid_disorder",
    "label_lipid_next",
    "label_diabetes_current",
    "label_hypertension_current",
    "label_obesity_current",
    "label_cardio_next",
    "label_kidney_next",
    "label_stroke_next",
]


def _label_stats_rows(df: pd.DataFrame, split: str) -> list[dict]:
    rows: list[dict] = []
    n_rows = len(df)
    for col in STAT_LABELS:
        if col not in df.columns:
            continue
        s = df[col]
        n_miss = int(s.isna().sum())
        v = pd.to_numeric(s, errors="coerce")
        n_valid = int(v.notna().sum())
        pos = int((v == 1).sum())
        neg = int((v == 0).sum())
        pr = float(pos / n_valid) if n_valid else float("nan")
        rows.append(
            {
                "split": split,
                "label": col,
                "n_rows": n_rows,
                "n_positive": pos,
                "n_negative": neg,
                "positive_rate": round(pr, 6) if pr == pr else pr,
                "n_missing": n_miss,
            }
        )
    return rows


def main(
    source_processed: Path,
    output_processed: Path,
) -> None:
    train_src = source_processed / "train.csv"
    test_src = source_processed / "test.csv"
    if not train_src.is_file():
        raise FileNotFoundError(f"Missing {train_src}")
    if not test_src.is_file():
        raise FileNotFoundError(f"Missing {test_src}")

    train_in = pd.read_csv(train_src)
    test_in = pd.read_csv(test_src)
    n_tr, n_te = len(train_in), len(test_in)

    train_out = add_multitask_labels(train_in)
    test_out = add_multitask_labels(test_in)

    out_dir = output_processed
    out_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_dir / "train.csv"
    out_test = out_dir / "test.csv"
    if out_train.resolve() == train_src.resolve() or out_test.resolve() == test_src.resolve():
        raise ValueError("output directory must differ from source to avoid overwrite")

    # ---- validation ----
    assert len(train_out) == n_tr, "train row count mismatch"
    assert len(test_out) == n_te, "test row count mismatch"
    assert (train_out["label_lipid_next"] == train_out["label_lipid_disorder"]).all()
    assert (test_out["label_lipid_next"] == test_out["label_lipid_disorder"]).all()
    new_cols = [
        "label_lipid_next",
        "label_diabetes_current",
        "label_hypertension_current",
        "label_obesity_current",
        "label_cardio_next",
        "label_kidney_next",
        "label_stroke_next",
    ]
    for name, dfo in (("train", train_out), ("test", test_out)):
        for c in new_cols:
            assert c in dfo.columns, c
            assert dfo[c].isna().sum() == 0, f"{name}.{c} has nulls"
            assert bool(dfo[c].isin([0, 1]).all()), f"{name}.{c} not binary 0/1"

    save_dataframe(train_out, out_train)
    save_dataframe(test_out, out_test)

    stats = _label_stats_rows(train_out, "train") + _label_stats_rows(test_out, "test")
    stats_path = out_dir / "label_stats.csv"
    save_dataframe(pd.DataFrame(stats), stats_path)

    logger.info(
        "Wrote %d train + %d test rows to %s (label_stats.csv included)",
        n_tr,
        n_te,
        out_dir,
    )
    print("Validation OK: row counts, label_lipid_next == label_lipid_disorder, binary 0/1, no nulls in new labels.")
    print("Saved:", out_train)
    print("Saved:", out_test)
    print("Saved:", stats_path)
    for r in stats:
        print(
            f"  [{r['split']}] {r['label']}: n={r['n_rows']} pos={r['n_positive']} neg={r['n_negative']}"
            f" rate={r['positive_rate']} miss={r['n_missing']}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-processed",
        type=Path,
        default=Path("data/processed/mimiciii"),
        help="Directory containing train.csv and test.csv to extend",
    )
    ap.add_argument(
        "--output-processed",
        type=Path,
        default=Path("data/processed/mimiciii_multitask"),
        help="Output directory (separate from source)",
    )
    args = ap.parse_args()
    main(args.source_processed.resolve(), args.output_processed.resolve())
