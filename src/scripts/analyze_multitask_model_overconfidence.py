"""Multitask overconfidence and probability histograms (GPT vs Gemma, MIMIC-III).

Merges gold labels from the multitask test split with GPT ``llm_*_results.csv`` and
optional Gemma JSONL or CSV, then for each clinical task writes:

  - ``gpt_prob_hist__<task>.png``
  - ``gemma_prob_hist__<task>.png`` (when Gemma probabilities exist)
  - ``overconfidence_report.md`` (one file, section per task)

For single-task (flat comparison CSV) analysis, see ``analyze_model_overconfidence``.

Usage::

    python -m src.scripts.analyze_multitask_model_overconfidence
    python -m src.scripts.analyze_multitask_model_overconfidence \\
        --gpt-csv data/outputs/.../llm_few_shot_results.csv \\
        --gemma-jsonl data/outputs/.../raw_llm_responses/few_shot/responses.jsonl
    python -m src.scripts.analyze_multitask_model_overconfidence --no-gemma
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.llm.output_parser import multitask_flat_column_names
from src.scripts.analyze_model_overconfidence import (
    bin_analysis,
    extreme_stats,
    make_hist,
    table_to_md,
)
from src.scripts.build_multitask_expert_scenarios import TASK_SPECS, _load_gemma_from_jsonl

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TEST_CSV = _DEFAULT_ROOT / "data/processed/mimiciii_multitask/test.csv"
DEFAULT_GPT_CSV = (
    _DEFAULT_ROOT
    / "data/outputs/mimiciii_llm_gpt4o_mini_promptv3_multitask_test/llm_coagent_results.csv"
)
DEFAULT_GEMMA_JSONL = (
    _DEFAULT_ROOT
    / "data/outputs/mimiciii_llm_gemma4_promptv3_multitask_test/raw_llm_responses/zero_shot/responses.jsonl"
)

TASK_TITLES: Dict[str, str] = {
    "lipid": "Lipid Disorder (Next Visit)",
    "diabetes": "Diabetes (Current Visit)",
    "hypertension": "Hypertension (Current Visit)",
    "obesity": "Obesity (Current Visit)",
    "cardio": "Cardiovascular Condition (Next Visit)",
    "kidney": "Kidney Condition (Next Visit)",
    "stroke": "Stroke (Next Visit)",
}


def _slug_from_gpt_csv(gpt_csv: Path) -> str:
    parent = gpt_csv.parent.name
    stem = gpt_csv.stem
    m = re.match(r"llm_(.+)_results$", stem)
    mode = m.group(1) if m else stem
    return f"{parent}__{mode}"


def _default_output_dir(gpt_csv: Path) -> Path:
    return _DEFAULT_ROOT / "data/outputs/case_studies/analysis_multitask" / _slug_from_gpt_csv(gpt_csv)


def _load_gemma_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pair_id" not in df.columns:
        raise ValueError(f"Gemma CSV missing pair_id: {path}")
    rename: Dict[str, str] = {}
    for c in multitask_flat_column_names():
        if c in df.columns:
            rename[c] = f"gemma_{c}"
    if not rename:
        raise ValueError(f"No multitask probability/prediction columns found in Gemma CSV: {path}")
    out = df[["pair_id"] + [c for c in multitask_flat_column_names() if c in df.columns]].copy()
    return out.rename(columns=rename)


def _merge_frames(
    test_csv: Path,
    gpt_csv: Path,
    gemma_jsonl: Optional[Path],
    gemma_csv: Optional[Path],
) -> pd.DataFrame:
    label_cols = ["pair_id"] + [t[3] for t in TASK_SPECS]
    test_full = pd.read_csv(test_csv)
    missing = [c for c in label_cols if c not in test_full.columns]
    if missing:
        raise ValueError(f"Test CSV missing columns: {missing}")
    test_df = test_full[label_cols].copy()

    gpt = pd.read_csv(gpt_csv)
    if "pair_id" not in gpt.columns:
        raise ValueError(f"GPT CSV missing pair_id: {gpt_csv}")
    for c in [t[3] for t in TASK_SPECS]:
        if c in gpt.columns:
            gpt = gpt.drop(columns=[c])

    merged = test_df.merge(gpt, on="pair_id", how="inner")
    if len(merged) == 0:
        raise RuntimeError("No rows after test+GPT merge; check pair_id alignment.")

    if gemma_jsonl is not None:
        gem = _load_gemma_from_jsonl(gemma_jsonl)
        merged = merged.merge(gem, on="pair_id", how="inner")
        if len(merged) == 0:
            raise RuntimeError("No rows after adding Gemma (JSONL); check pair_id alignment.")
    elif gemma_csv is not None:
        gem = _load_gemma_from_csv(gemma_csv)
        merged = merged.merge(gem, on="pair_id", how="inner")
        if len(merged) == 0:
            raise RuntimeError("No rows after adding Gemma (CSV); check pair_id alignment.")

    return merged


def _empty_bins() -> pd.DataFrame:
    labels = ["[0.0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", "(0.6, 0.8]", "(0.8, 1.0]"]
    return pd.DataFrame(
        {
            "Probability Bin": labels,
            "Count": [0] * len(labels),
            "Avg True Label": [float("nan")] * len(labels),
        }
    )


def _bin_rows(df: pd.DataFrame) -> List[List[str]]:
    rows: List[List[str]] = []
    for _, r in df.iterrows():
        avg = "NA" if pd.isna(r["Avg True Label"]) else f"{float(r['Avg True Label']):.4f}"
        rows.append([str(r["Probability Bin"]), str(int(r["Count"])), avg])
    return rows


def _extreme_row(model: str, ext: Dict[str, float]) -> List[str]:
    return [
        model,
        str(ext["count_0"]),
        str(ext["count_1"]),
        str(ext["total"]),
        f"{ext['pct_extreme']:.2f}%",
    ]


def _build_task_section(
    task_key: str,
    title: str,
    n_gpt: int,
    n_gemma: int,
    has_gemma: bool,
    gemma_ext: Dict[str, float],
    gpt_ext: Dict[str, float],
    gemma_bins: pd.DataFrame,
    gpt_bins: pd.DataFrame,
) -> List[str]:
    lines = [
        f"## {title} (`{task_key}`)",
        "",
        f"- Rows with non-null **GPT** probability for this task: **{n_gpt}**",
    ]
    if has_gemma:
        lines.append(f"- Rows with non-null **Gemma** probability for this task: **{n_gemma}**")
    else:
        lines.append("- **Gemma:** not included in this run (no Gemma source or no usable probabilities).")
    lines.extend(["", "### Extreme probabilities", ""])
    ext_rows = []
    if has_gemma and n_gemma > 0:
        ext_rows.append(_extreme_row("Gemma", gemma_ext))
    if n_gpt > 0:
        ext_rows.append(_extreme_row("GPT-4o-mini", gpt_ext))
    if not ext_rows:
        lines.append("_No non-null probabilities for this task._")
    else:
        lines.append(table_to_md(["Model", "Count(0)", "Count(1)", "Total", "% Extreme"], ext_rows))
    lines.extend(["", "### Probability bin analysis", ""])
    if has_gemma and n_gemma > 0:
        lines.extend(["#### Gemma", "", table_to_md(["Probability Bin", "Count", "Avg True Label"], _bin_rows(gemma_bins)), ""])
    if n_gpt > 0:
        lines.extend(["#### GPT-4o-mini", "", table_to_md(["Probability Bin", "Count", "Avg True Label"], _bin_rows(gpt_bins)), ""])
    if (not has_gemma or n_gemma == 0) and n_gpt == 0:
        lines.append("_No bin analysis (no valid probabilities)._")
    lines.append("")
    return lines


def run_analysis(
    merged: pd.DataFrame,
    out_dir: Path,
    has_gemma: bool,
    provenance: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines: List[str] = [
        "# Multitask overconfidence report",
        "",
        "Per-task probability histograms and descriptive bin statistics (GPT-4o-mini vs Gemma when available).",
        "This report is descriptive only; it does not apply calibration or retraining.",
        "",
    ]
    if provenance:
        report_lines.extend(["## Data sources", "", provenance.strip(), ""])
    if not has_gemma:
        report_lines.append("**Note:** Gemma was omitted or not merged; GPT-only metrics are shown.")
        report_lines.append("")

    for task_key, pred_c, prob_c, label_c in TASK_SPECS:
        title = TASK_TITLES.get(task_key, task_key)
        gemma_c = f"gemma_{prob_c}"
        y = merged[label_c].astype(float)
        gpt_p = pd.to_numeric(merged[prob_c], errors="coerce")
        gem_p = pd.to_numeric(merged[gemma_c], errors="coerce") if has_gemma and gemma_c in merged.columns else pd.Series(dtype=float)

        sub_gpt = pd.DataFrame({"true_label": y, "gpt_prob": gpt_p}).dropna(subset=["gpt_prob"])
        sub_gpt["true_label"] = sub_gpt["true_label"].astype(int)
        n_gpt = len(sub_gpt)

        if n_gpt > 0:
            make_hist(
                sub_gpt["gpt_prob"],
                f"GPT-4o-mini — {title}",
                out_dir / f"gpt_prob_hist__{task_key}.png",
            )
            gpt_ext = extreme_stats(sub_gpt["gpt_prob"])
            gpt_bins = bin_analysis(sub_gpt, "gpt_prob")
        else:
            gpt_ext = {"count_0": 0, "count_1": 0, "total": 0, "pct_extreme": 0.0}
            gpt_bins = _empty_bins()

        n_gemma = 0
        gemma_ext = {"count_0": 0, "count_1": 0, "total": 0, "pct_extreme": 0.0}
        gemma_bins = _empty_bins()
        if has_gemma and gemma_c in merged.columns:
            sub_gem = pd.DataFrame({"true_label": y, "gemma_prob": gem_p}).dropna(subset=["gemma_prob"])
            sub_gem["true_label"] = sub_gem["true_label"].astype(int)
            n_gemma = len(sub_gem)
            if n_gemma > 0:
                make_hist(
                    sub_gem["gemma_prob"],
                    f"Gemma — {title}",
                    out_dir / f"gemma_prob_hist__{task_key}.png",
                )
                gemma_ext = extreme_stats(sub_gem["gemma_prob"])
                gemma_bins = bin_analysis(sub_gem, "gemma_prob")

        report_lines.extend(
            _build_task_section(
                task_key,
                title,
                n_gpt,
                n_gemma,
                has_gemma and gemma_c in merged.columns,
                gemma_ext,
                gpt_ext,
                gemma_bins,
                gpt_bins,
            )
        )

    report_lines.extend(
        [
            "## Notes",
            "",
            "- Counts can differ between models when one side has parse failures or missing task probabilities.",
            "- Extreme counts treat exactly 0.0 and 1.0 as boundary spikes.",
            "",
        ]
    )
    (out_dir / "overconfidence_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multitask probability histograms and overconfidence-style report.")
    ap.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    ap.add_argument("--gpt-csv", type=Path, default=DEFAULT_GPT_CSV)
    ap.add_argument("--gemma-jsonl", type=Path, default=None, help="Gemma raw responses JSONL (sample_id + text).")
    ap.add_argument("--gemma-csv", type=Path, default=None, help="Gemma llm_*_results.csv-style wide CSV.")
    ap.add_argument("--no-gemma", action="store_true", help="Only GPT vs gold (no Gemma merge).")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.gemma_jsonl and args.gemma_csv:
        raise SystemExit("Pass at most one of --gemma-jsonl and --gemma-csv.")
    if args.no_gemma and (args.gemma_jsonl or args.gemma_csv):
        raise SystemExit("--no-gemma cannot be combined with --gemma-jsonl or --gemma-csv.")

    gemma_jsonl: Optional[Path] = None
    gemma_csv: Optional[Path] = None
    has_gemma = not args.no_gemma
    if has_gemma:
        if args.gemma_csv:
            gemma_csv = args.gemma_csv
            if not gemma_csv.is_file():
                raise SystemExit(f"Gemma CSV not found: {gemma_csv}")
        elif args.gemma_jsonl:
            gemma_jsonl = args.gemma_jsonl
            if not gemma_jsonl.is_file():
                raise SystemExit(f"Gemma JSONL not found: {gemma_jsonl}")
        else:
            gemma_jsonl = DEFAULT_GEMMA_JSONL if DEFAULT_GEMMA_JSONL.is_file() else None
            if gemma_jsonl is None:
                has_gemma = False

    merged = _merge_frames(args.test_csv, args.gpt_csv, gemma_jsonl, gemma_csv)
    out_dir = args.output_dir or _default_output_dir(args.gpt_csv)

    # Per-task N summary
    print(f"Merged rows: {len(merged)}")
    for task_key, _, prob_c, label_c in TASK_SPECS:
        gpt_n = merged[prob_c].notna().sum()
        gem_c = f"gemma_{prob_c}"
        gem_n = int(merged[gem_c].notna().sum()) if has_gemma and gem_c in merged.columns else 0
        print(f"  {task_key}: GPT n={gpt_n}, Gemma n={gem_n}")

    merged_has_gemma = any(c.startswith("gemma_") and c.endswith("_prob") for c in merged.columns)
    prov_bits = [f"- **GPT CSV:** `{args.gpt_csv}`", f"- **Test CSV:** `{args.test_csv}`"]
    if gemma_jsonl is not None:
        prov_bits.append(f"- **Gemma JSONL:** `{gemma_jsonl}`")
    elif gemma_csv is not None:
        prov_bits.append(f"- **Gemma CSV:** `{gemma_csv}`")
    else:
        prov_bits.append("- **Gemma:** omitted")
    provenance = "\n".join(prov_bits)
    run_analysis(merged, out_dir, has_gemma=merged_has_gemma, provenance=provenance)
    print(f"Wrote analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()
