#!/usr/bin/env python3
"""Merge LLM result CSVs with test narratives into one markdown audit file.

Shows, per sample and mode: ground truth, parsed prediction, match/mismatch,
parser status, full model reply, and the patient narrative. Full rendered
prompts (system + user as sent to the API) live under
``prompts_used/<mode>/prompt_<pair_id>.txt`` after a run with an updated
predictor (or rebuild from templates + narrative for zero-shot modes).

Usage (from repo root, with venv activated)::

    python scripts/export_llm_audit.py
    python scripts/export_llm_audit.py --out /tmp/audit.md --modes zero_shot coagent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODES = ("zero_shot", "zero_shot_plus", "few_shot", "coagent")


def _fence(s: str) -> str:
    return "```\n" + (s or "").replace("```", "``\u200b`") + "\n```\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Export LLM prompt/answer/eval audit markdown.")
    p.add_argument(
        "--outputs",
        type=Path,
        default=Path("data/outputs/mimiciii"),
        help="Directory containing llm_*_results.csv",
    )
    p.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/mimiciii/test.csv"),
        help="Test split with narrative_current and label_lipid_disorder",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/outputs/mimiciii/llm_audit_report.md"),
        help="Output markdown path",
    )
    p.add_argument(
        "--modes",
        nargs="*",
        default=list(MODES),
        choices=MODES,
        help="Which modes to include (default: all)",
    )
    args = p.parse_args()

    test_df = pd.read_csv(args.test)
    need = {"pair_id", "narrative_current", "label_lipid_disorder"}
    missing = need - set(test_df.columns)
    if missing:
        raise SystemExit(f"test CSV missing columns: {missing}")

    lines: list[str] = []
    lines.append("# LLM audit: narrative, model answer, evaluation\n\n")
    lines.append(
        "Evaluation: `parsed_prediction` is mapped Yes→1, No→0; compared to "
        "`label_lipid_disorder` (1 = next visit has lipid disorder). "
        "`parsed_probability` (when `parse_valid_probability`) is used for ROC-AUC / AUPRC. "
        "`unparseable` predictions are excluded from hard-metric counts. "
        "See `src/evaluation/evaluate_llm_runs.py` and `src/llm/output_parser.py`.\n\n"
    )

    for mode in args.modes:
        res_path = args.outputs / f"llm_{mode}_results.csv"
        if not res_path.exists():
            lines.append(f"## Mode `{mode}`\n\n*(no file {res_path})*\n\n")
            continue

        df = pd.read_csv(res_path)
        merged = df.merge(
            test_df[["pair_id", "narrative_current"]],
            on="pair_id",
            how="left",
        )
        merged = merged.sort_values("pair_id")

        lines.append(f"## Mode `{mode}`\n\n")
        lines.append(f"- Results: `{res_path}`\n")
        lines.append(
            f"- Raw responses (JSONL): `{args.outputs}/raw_llm_responses/{mode}/responses.jsonl`\n"
        )
        lines.append(
            f"- Rendered prompts (if present): `{args.outputs}/prompts_used/{mode}/prompt_<pair_id>.txt`\n\n"
        )

        for _, row in merged.iterrows():
            pid = row["pair_id"]
            y_true = int(row["true_label"]) if pd.notna(row.get("true_label")) else row.get("label_lipid_disorder")
            pred_raw = row.get("parsed_prediction", "")
            pred_map = {"Yes": 1, "No": 0}
            pred_bin = pred_map.get(str(pred_raw), None)
            if pred_bin is None:
                verdict = "unparseable (excluded from ACC)"
            else:
                verdict = "correct" if pred_bin == int(y_true) else "wrong"

            lines.append(f"### pair_id={pid}\n\n")
            lines.append(f"- **Ground truth** (`label_lipid_disorder`): {y_true}\n")
            lines.append(f"- **Parsed prediction**: {pred_raw}\n")
            lines.append(f"- **Verdict**: {verdict}\n")
            lines.append(f"- **Parser**: `{row.get('parser_status', '')}`\n")
            if "parsed_probability" in row.index and pd.notna(row.get("parsed_probability")):
                lines.append(f"- **Parsed probability**: {row.get('parsed_probability')} (valid={row.get('parse_valid_probability', '')})\n")
            lines.append(f"- **Model**: `{row.get('model_name', '')}`\n\n")
            lines.append("**Input narrative** (`narrative_current`):\n\n")
            lines.append(_fence(str(row.get("narrative_current", ""))))
            lines.append("\n**Model reply** (`raw_response`):\n\n")
            lines.append(_fence(str(row.get("raw_response", ""))))
            lines.append("\n---\n\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
