"""Export MIMIC-III multitask train/test CSVs to JSONL for Unsloth-style SFT.

Uses a fixed instruction (not ``prompts_v3`` files). Each line is Alpaca-like:
``instruction``, ``input`` (narrative only), ``output`` (strict multitask JSON
matching ``parse_multitask_output`` in ``src.llm.output_parser``).

Usage::

    python -m src.scripts.export_multitask_unsloth_jsonl

    python -m src.scripts.export_multitask_unsloth_jsonl \\
      --instruction-version strict_v2 \\
      --compact-output-json \\
      --reasoning-style minimal \\
      --out-dir dataset_for_unsloth_multitask_strict_v2

Writes ``train.jsonl`` and ``test.jsonl`` under the output directory.

Downstream: ``scripts/train_unsloth_router.py`` loads these files and re-validates each
row's ``output`` with ``parse_multitask_output`` from ``src.llm.output_parser`` (same rules).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.llm.output_parser import MULTITASK_JSON_TASK_KEYS, parse_multitask_output

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

_INSTRUCTION_V1 = (
    "You are given one current hospital visit narrative for a patient. "
    "Output exactly one JSON object with keys: lipid_next, diabetes_current, "
    "hypertension_current, obesity_current, cardio_next, kidney_next, stroke_next. "
    "Each key maps to an object with fields prediction (Yes or No) and probability "
    "(a number from 0 to 1, use two decimal places). Also include key reasoning "
    "with a short string. Do not add text outside the JSON object."
)

_INSTRUCTION_STRICT_V2 = (
    "You are a seven-task clinical classifier. "
    "You are given one current hospital visit narrative for a patient. "
    "Output exactly one JSON object. "
    'The response must start with {"lipid_next":. '
    "The only allowed top-level keys are: "
    "lipid_next, diabetes_current, hypertension_current, obesity_current, "
    "cardio_next, kidney_next, stroke_next, reasoning. "
    "Do not output top-level keys named prediction or probability. "
    "Do not output a single global prediction. "
    "Do not output diagnoses, medications, procedures, response, summary, or labels. "
    'For every task, output exactly {"prediction":"Yes" or "No","probability":number}. '
    "The final answer must contain no Markdown and no text outside the JSON."
)

STRICT_V2_OUTPUT_PREFIX = '{"lipid_next":'

_FORBIDDEN_TOP_LEVEL_KEYS = frozenset(
    {
        "prediction",
        "probability",
        "diagnoses",
        "medications",
        "procedures",
        "response",
        "summary",
        "labels",
    }
)

_REASONING_MINIMAL = "codes"
_REASONING_DEFAULT = "Supervised gold labels from ICD-derived visit-pair coding."


def _instruction_for_version(version: str) -> str:
    if version == "v1":
        return _INSTRUCTION_V1
    if version == "strict_v2":
        return _INSTRUCTION_STRICT_V2
    raise ValueError(f"Unknown instruction version: {version!r}")


def _gold_multitask_json(
    row: pd.Series,
    *,
    compact: bool,
    reasoning: str,
) -> str:
    obj: Dict[str, Any] = {}
    for task_key, col in _TASK_TO_LABEL_COL:
        v = int(row[col])
        pred = "Yes" if v == 1 else "No"
        prob = 1.0 if v == 1 else 0.0
        obj[task_key] = {"prediction": pred, "probability": prob}
    obj["reasoning"] = reasoning
    if compact:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(obj, ensure_ascii=False)


def _parse_top_level_keys(raw: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _validate_output_record(
    output: str,
    row_index: int,
    *,
    strict_v2: bool,
) -> None:
    """Raise ValueError if output fails strict parser or strict_v2 extra rules."""
    if parse_multitask_output(output) is None:
        raise ValueError(
            f"Row {row_index}: output failed parse_multitask_output. "
            f"First 200 chars: {output[:200]!r}"
        )

    if not strict_v2:
        return

    stripped = output.strip()
    if not stripped.startswith(STRICT_V2_OUTPUT_PREFIX):
        raise ValueError(
            f"Row {row_index}: output must start with {STRICT_V2_OUTPUT_PREFIX!r}; "
            f"got start {stripped[:40]!r}"
        )

    data = _parse_top_level_keys(stripped)
    if data is None:
        raise ValueError(f"Row {row_index}: output is not valid JSON object.")

    for bad in _FORBIDDEN_TOP_LEVEL_KEYS:
        if bad in data:
            raise ValueError(
                f"Row {row_index}: forbidden top-level key {bad!r} present in output."
            )

    for tk in MULTITASK_JSON_TASK_KEYS:
        if tk not in data:
            raise ValueError(f"Row {row_index}: missing task key {tk!r}.")


def _row_to_record(
    row: pd.Series,
    *,
    instruction: str,
    compact: bool,
    reasoning: str,
) -> Dict[str, Any]:
    return {
        "pair_id": int(row["pair_id"]),
        "instruction": instruction,
        "input": str(row["narrative_current"] or ""),
        "output": _gold_multitask_json(row, compact=compact, reasoning=reasoning),
    }


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def _validate_all_records(
    records: List[Dict[str, Any]],
    *,
    split_name: str,
    strict_v2: bool,
) -> Dict[str, int]:
    n = len(records)
    parse_ok = 0
    for i, rec in enumerate(records):
        out = str(rec.get("output") or "")
        _validate_output_record(out, i, strict_v2=strict_v2)
        parse_ok += 1
    print(
        f"Validated {split_name}: {n} rows — parse_multitask_output OK; "
        f"strict_v2 rules={'yes' if strict_v2 else 'no'}"
    )
    return {"n": n, "parse_ok": parse_ok}


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
        help="Output directory (default depends on --instruction-version)",
    )
    ap.add_argument(
        "--instruction-version",
        choices=("v1", "strict_v2"),
        default="v1",
        help="Instruction template (strict_v2 adds explicit multitask JSON contract).",
    )
    ap.add_argument(
        "--compact-output-json",
        action="store_true",
        help="Emit compact JSON (no spaces after separators).",
    )
    ap.add_argument(
        "--reasoning-style",
        choices=("default", "minimal"),
        default="default",
        help='Reasoning string in gold JSON ("minimal" -> "codes").',
    )
    args = ap.parse_args()
    root = args.root
    proc = args.processed_dir or (root / "data/processed/mimiciii_multitask")
    strict_v2 = args.instruction_version == "strict_v2"
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif strict_v2:
        out_dir = root / "dataset_for_unsloth_multitask_strict_v2"
    else:
        out_dir = root / "dataset_for_unsloth"

    if strict_v2 and not args.compact_output_json:
        print(
            "Note: strict_v2 export typically uses --compact-output-json "
            '(gold starts with {"lipid_next":).',
            file=sys.stderr,
        )

    instruction = _instruction_for_version(args.instruction_version)
    reasoning = _REASONING_MINIMAL if args.reasoning_style == "minimal" else _REASONING_DEFAULT
    compact = bool(args.compact_output_json)

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

    train_recs = [
        _row_to_record(row, instruction=instruction, compact=compact, reasoning=reasoning)
        for _, row in train_df.iterrows()
    ]
    test_recs = [
        _row_to_record(row, instruction=instruction, compact=compact, reasoning=reasoning)
        for _, row in test_df.iterrows()
    ]

    _validate_all_records(train_recs, split_name="train", strict_v2=strict_v2)
    _validate_all_records(test_recs, split_name="test", strict_v2=strict_v2)

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    n_train = _write_jsonl(train_recs, train_path)
    n_test = _write_jsonl(test_recs, test_path)

    sample_out = train_recs[0]["output"] if train_recs else ""
    manifest = {
        "instruction_version": args.instruction_version,
        "compact_output_json": compact,
        "reasoning_style": args.reasoning_style,
        "strict_v2_output_prefix": STRICT_V2_OUTPUT_PREFIX if strict_v2 else None,
        "instruction_note": "Fixed string; not loaded from prompts_v3.",
        "source_train_csv": str(train_csv.resolve()),
        "source_test_csv": str(test_csv.resolve()),
        "n_train": n_train,
        "n_test": n_test,
        "sample_output_prefix": sample_out[:48],
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
    if strict_v2:
        print(f"All outputs start with: {STRICT_V2_OUTPUT_PREFIX!r}")


if __name__ == "__main__":
    main()
