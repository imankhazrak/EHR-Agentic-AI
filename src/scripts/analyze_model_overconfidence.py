"""Analyze model confidence behavior from existing comparison outputs.

Creates:
  - gemma_prob_hist.png
  - gpt_prob_hist.png
  - overconfidence_report.md

No model outputs are modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path(
    "/users/PCS0229/imankhazrak/EHR-Agentic-AI/data/outputs/case_studies/model_comparison_gemma_coagent_vs_gpt4omini_fewshot.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/users/PCS0229/imankhazrak/EHR-Agentic-AI/data/outputs/case_studies/analysis"
)
REQUIRED_COLUMNS = ["case_id", "true_label", "gpt_prob", "gemma_prob"]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_hist(series: pd.Series, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    counts, _, patches = plt.hist(series, bins=50, edgecolor="black")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.title(title)
    for count, patch in zip(counts, patches):
        if count <= 0:
            continue
        x = patch.get_x() + patch.get_width() / 2.0
        y = patch.get_height()
        plt.text(x, y, f"{int(count)}", ha="center", va="bottom", fontsize=7, rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def extreme_stats(series: pd.Series) -> Dict[str, float]:
    total = int(series.shape[0])
    count0 = int((series == 0).sum())
    count1 = int((series == 1).sum())
    extreme = count0 + count1
    pct_extreme = (extreme / total * 100.0) if total else 0.0
    return {
        "count_0": count0,
        "count_1": count1,
        "total": total,
        "pct_extreme": pct_extreme,
    }


def bin_analysis(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["[0.0, 0.2]", "(0.2, 0.4]", "(0.4, 0.6]", "(0.6, 0.8]", "(0.8, 1.0]"]
    binned = pd.cut(df[prob_col], bins=bins, labels=labels, include_lowest=True, right=True)
    grouped = (
        df.assign(prob_bin=binned)
        .groupby("prob_bin", observed=False)["true_label"]
        .agg(["count", "mean"])
        .reindex(labels)
        .reset_index()
    )
    grouped.columns = ["Probability Bin", "Count", "Avg True Label"]
    grouped["Count"] = grouped["Count"].fillna(0).astype(int)
    grouped["Avg True Label"] = grouped["Avg True Label"].round(4)
    return grouped


def table_to_md(headers: List[str], rows: List[List[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header_line, sep_line] + body)


def build_report(
    gemma_ext: Dict[str, float],
    gpt_ext: Dict[str, float],
    gemma_bins: pd.DataFrame,
    gpt_bins: pd.DataFrame,
) -> str:
    extreme_rows = [
        [
            "Gemma",
            str(gemma_ext["count_0"]),
            str(gemma_ext["count_1"]),
            str(gemma_ext["total"]),
            f"{gemma_ext['pct_extreme']:.2f}%",
        ],
        [
            "GPT-4o-mini",
            str(gpt_ext["count_0"]),
            str(gpt_ext["count_1"]),
            str(gpt_ext["total"]),
            f"{gpt_ext['pct_extreme']:.2f}%",
        ],
    ]

    def bin_rows(df: pd.DataFrame) -> List[List[str]]:
        rows: List[List[str]] = []
        for _, r in df.iterrows():
            avg = "NA" if pd.isna(r["Avg True Label"]) else f"{float(r['Avg True Label']):.4f}"
            rows.append([str(r["Probability Bin"]), str(int(r["Count"])), avg])
        return rows

    lines = [
        "# Overconfidence Report",
        "",
        "## Section 1 — Distribution Overview",
        "Gemma and GPT-4o-mini probability histograms summarize the spread of predicted probabilities over the full dataset.",
        "Both distributions can be inspected for concentration near boundaries and overall shape across the probability range.",
        "",
        "## Section 2 — Extreme Probabilities",
        table_to_md(["Model", "Count(0)", "Count(1)", "Total", "% Extreme"], extreme_rows),
        "",
        "## Section 3 — Probability Bin Analysis",
        "",
        "### Gemma",
        table_to_md(["Probability Bin", "Count", "Avg True Label"], bin_rows(gemma_bins)),
        "",
        "### GPT-4o-mini",
        table_to_md(["Probability Bin", "Count", "Avg True Label"], bin_rows(gpt_bins)),
        "",
        "## Section 4 — Notes (IMPORTANT)",
        "- A portion of predictions fall at extreme probability values (0 or 1).",
        "- The empirical positive rate varies across probability bins.",
        "- This report is descriptive only and does not apply calibration, threshold changes, or model retraining.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze probability distribution and overconfidence behavior.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    validate_columns(df)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    make_hist(df["gemma_prob"], "Gemma Probability Distribution", out_dir / "gemma_prob_hist.png")
    make_hist(df["gpt_prob"], "GPT-4o-mini Probability Distribution", out_dir / "gpt_prob_hist.png")

    gemma_ext = extreme_stats(df["gemma_prob"])
    gpt_ext = extreme_stats(df["gpt_prob"])
    gemma_bins = bin_analysis(df, "gemma_prob")
    gpt_bins = bin_analysis(df, "gpt_prob")

    report = build_report(gemma_ext, gpt_ext, gemma_bins, gpt_bins)
    (out_dir / "overconfidence_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()
