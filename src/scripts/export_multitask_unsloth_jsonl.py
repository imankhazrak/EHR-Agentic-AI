"""Export MIMIC-III multitask train/test CSVs to JSONL for Unsloth-style SFT.

Uses a short fixed instruction (not ``prompts_v3`` files). Each line is Alpaca-like:
``instruction``, ``input`` (narrative only), ``output`` (strict multitask JSON
matching ``parse_multitask_output`` in ``src.llm.output_parser``).

Usage::

    python -m src.scripts.export_multitask_unsloth_jsonl

Writes ``dataset_for_unsloth/train.jsonl`` and ``dataset_for_unsloth/test.jsonl``.

Downstream: ``scripts/train_unsloth_router.py`` loads these files and re-validates each
row's ``output`` with ``parse_multitask_output`` from ``src.llm.output_parser`` (same rules).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.llm.output_parser import MULTITASK_JSON_TASK_KEYS

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

# JSON task key -> CSV column (gold)
_TASK_TO_LABEL_COL: Tuple[Tuple[str, str], ...] = tuple(
    zip(
        MULTITASK_JSON_TASK_KEYS,
        (
            "label_lipid_next",
            "label_diabetes_current",
            "label_hypertension_current",
            "label_obesity_current",
            "label_cardio_next",
            "label_kidney_next",
            "label_stroke_next",
        ),
    )
)

_INSTRUCTION = (
    "You are given one current hospital visit narrative for a patient. "
    "Output exactly one JSON object with keys: lipid_next, diabetes_current, "
    "hypertension_current, obesity_current, cardio_next, kidney_next, stroke_next. "
    "Each key maps to an object with fields prediction (Yes or No) and probability "
    "(a number from 0 to 1, use two decimal places). Also include key reasoning "
    "with a short string. Do not add text outside the JSON object."
)


def _gold_multitask_json(row: pd.Series) -> str:
    obj: Dict[str, Any] = {}
    for task_key, col in _TASK_TO_LABEL_COL:
        v = int(row[col])
        pred = "Yes" if v == 1 else "No"
        prob = 1.00 if v == 1 else 0.00
        obj[task_key] = {"prediction": pred, "probability": prob}
    obj["reasoning"] = "Supervised gold labels from ICD-derived visit-pair coding."
    return json.dumps(obj, ensure_ascii=False)


def _row_to_record(row: pd.Series) -> Dict[str, Any]:
    return {
        "pair_id": int(row["pair_id"]),
        "instruction": _INSTRUCTION,
        "input": str(row["narrative_current"] or ""),
        "output": _gold_multitask_json(row),
    }


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Export multitask CSV splits to Unsloth JSONL.")
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory with train.csv and test.csv (default: <root>/data/processed/mimiciii_multitask)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/dataset_for_unsloth)",
    )
    args = ap.parse_args()
    root = args.root
    proc = args.processed_dir or (root / "data/processed/mimiciii_multitask")
    out_dir = args.out_dir or (root / "dataset_for_unsloth")

    train_csv = proc / "train.csv"
    test_csv = proc / "test.csv"
    if not train_csv.is_file():
        raise SystemExit(f"Missing train CSV: {train_csv}")
    if not test_csv.is_file():
        raise SystemExit(f"Missing test CSV: {test_csv}")

    required = ["pair_id", "narrative_current"] + [c for _, c in _TASK_TO_LABEL_COL]
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    for name, df in (("train", train_df), ("test", test_df)):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"{name}.csv missing columns: {missing}")

    train_recs = [_row_to_record(row) for _, row in train_df.iterrows()]
    test_recs = [_row_to_record(row) for _, row in test_df.iterrows()]

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    n_train = _write_jsonl(train_recs, train_path)
    n_test = _write_jsonl(test_recs, test_path)

    manifest = {
        "instruction_note": "Short fixed string; not loaded from prompts_v3.",
        "source_train_csv": str(train_csv.resolve()),
        "source_test_csv": str(test_csv.resolve()),
        "n_train": n_train,
        "n_test": n_test,
        "outputs": {
            "train_jsonl": str(train_path.resolve()),
            "test_jsonl": str(test_path.resolve()),
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
        mf.write("\n")

    print(f"Wrote {n_train} rows -> {train_path}")
    print(f"Wrote {n_test} rows -> {test_path}")
    print(f"Wrote manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
