"""Merge coagent parsed_probability columns from LLM runs into finetuning test.csv.

Reads:
  - data/mimic III/data_ready_for_finetuning_next_timing_and_lipid/test.csv
  - data/outputs/mimiciii_llm_gpt4o_mini_promptv2/llm_coagent_results.csv
  - data/outputs/mimiciii_llm_promptv2_gemma/llm_coagent_results.csv

Joins on pair_id (left: test.csv row order preserved). Adds:
  - coagent_parsed_probability_gpt4o_mini_promptv2
  - coagent_parsed_probability_gemma_promptv2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEST = _REPO_ROOT / "data/mimic III/data_ready_for_finetuning_next_timing_and_lipid/test.csv"
_GPT_COAGENT = _REPO_ROOT / "data/outputs/mimiciii_llm_gpt4o_mini_promptv2/llm_coagent_results.csv"
_GEMMA_COAGENT = _REPO_ROOT / "data/outputs/mimiciii_llm_promptv2_gemma/llm_coagent_results.csv"

_COL_GPT = "coagent_parsed_probability_gpt4o_mini_promptv2"
_COL_GEMMA = "coagent_parsed_probability_gemma_promptv2"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--test-csv",
        type=Path,
        default=_DEFAULT_TEST,
        help="Finetuning test.csv to update (default: %(default)s)",
    )
    ap.add_argument(
        "--gpt-coagent",
        type=Path,
        default=_GPT_COAGENT,
        help="GPT-4o-mini coagent results CSV",
    )
    ap.add_argument(
        "--gemma-coagent",
        type=Path,
        default=_GEMMA_COAGENT,
        help="Gemma coagent results CSV",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate merge only; do not write",
    )
    args = ap.parse_args()

    base = pd.read_csv(args.test_csv)
    for col in (_COL_GPT, _COL_GEMMA):
        if col in base.columns:
            base = base.drop(columns=[col])

    gpt = pd.read_csv(args.gpt_coagent)[["pair_id", "parsed_probability"]].rename(
        columns={"parsed_probability": _COL_GPT}
    )
    gem = pd.read_csv(args.gemma_coagent)[["pair_id", "parsed_probability"]].rename(
        columns={"parsed_probability": _COL_GEMMA}
    )

    out = base.merge(gpt, on="pair_id", how="left", validate="one_to_one")
    out = out.merge(gem, on="pair_id", how="left", validate="one_to_one")

    n = len(out)
    assert n == len(base), "row count must match base after merge"
    assert out[_COL_GPT].notna().all(), f"{_COL_GPT} has nulls"
    assert out[_COL_GEMMA].notna().all(), f"{_COL_GEMMA} has nulls"

    if args.dry_run:
        print(f"OK dry-run: {n} rows, columns {len(out.columns)}")
        return

    out.to_csv(args.test_csv, index=False)
    print(f"Wrote {args.test_csv} ({n} rows)")


if __name__ == "__main__":
    main()
