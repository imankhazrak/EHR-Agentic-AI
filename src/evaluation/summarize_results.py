"""Aggregate all experiment results into a single summary table matching paper style."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd

from src.utils.io import save_dataframe, load_json, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def collect_results(output_dir: str) -> pd.DataFrame:
    """Scan the output directory for metric JSONs and ML result CSVs, then build
    a summary table with columns: Model, Approach, ACC, Sensitivity, Specificity, F1.
    """
    odir = Path(output_dir)
    rows: List[Dict] = []

    # ML baselines
    for setting in ("fully_supervised", "few_shot"):
        fp = odir / f"ml_results_{setting}.csv"
        if fp.exists():
            ml_df = pd.read_csv(fp)
            for _, r in ml_df.iterrows():
                rows.append({
                    "Model": r["model"],
                    "Approach": r["setting"].replace("_", " ").title(),
                    "ACC": r["accuracy"],
                    "Sensitivity": r["sensitivity"],
                    "Specificity": r["specificity"],
                    "F1": r["f1"],
                })

    # LLM results
    for mode in ("zero_shot", "zero_shot_plus", "few_shot", "coagent"):
        fp = odir / f"llm_{mode}_metrics.json"
        if fp.exists():
            m = load_json(fp)
            model_name = m.get("model", "LLM")
            approach_map = {
                "zero_shot": "Zero-Shot",
                "zero_shot_plus": "Zero-Shot+",
                "few_shot": "Few-Shot (N=6)",
                "coagent": "EHR-CoAgent",
            }
            rows.append({
                "Model": model_name,
                "Approach": approach_map.get(mode, mode),
                "ACC": m["accuracy"],
                "Sensitivity": m["sensitivity"],
                "Specificity": m["specificity"],
                "F1": m["f1"],
            })

    summary = pd.DataFrame(rows)
    if len(summary) > 0:
        save_dataframe(summary, odir / "summary_table.csv")
        logger.info("Summary table:\n%s", summary.to_string(index=False))
    else:
        logger.warning("No results found to summarise")

    return summary
