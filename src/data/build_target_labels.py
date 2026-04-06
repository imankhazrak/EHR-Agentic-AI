"""Step 4 — Target label: Disorders of Lipid Metabolism in next visit.

The paper identifies this phenotype via CCS (Clinical Classifications
Software) single-level diagnosis category **53** from HCUP.

CCS category 53 maps to the following ICD-9-CM codes (source:
https://www.hcup-us.ahrq.gov/toolssoftware/ccs/AppendixASingleDX.txt):

    272.0  Pure hypercholesterolemia
    272.1  Pure hyperglyceridemia
    272.2  Mixed hyperlipidemia
    272.3  Hyperchylomicronemia
    272.4  Other and unspecified hyperlipidemia
    272.5  Lipoprotein deficiencies
    272.6  Lipodystrophy
    272.7  Lipidoses
    272.8  Other disorders of lipoid metabolism
    272.9  Unspecified disorder of lipoid metabolism

In MIMIC-III the ICD-9 codes are stored WITHOUT the dot, e.g. "2720".

ASSUMPTION: We match on prefix "272" which covers all 272.x codes.
This is isolated here so you can refine the exact code list if needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set

import pandas as pd

from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ICD-9 codes for CCS category 53 — Disorders of lipoid metabolism
# Codes stored without decimal in MIMIC-III
# ---------------------------------------------------------------------------
CCS53_ICD9_CODES: List[str] = [
    "2720", "2721", "2722", "2723", "2724",
    "2725", "2726", "2727", "2728", "2729",
]

# We also accept prefix match for safety (some codes have extra digits)
CCS53_PREFIX = "272"


def is_lipid_disorder(code: str) -> bool:
    """Return True if the ICD-9 code maps to CCS category 53."""
    code = str(code).strip()
    # Exact match
    if code in CCS53_ICD9_CODES:
        return True
    # Prefix match (e.g., "27200")
    if code.startswith(CCS53_PREFIX) and len(code) >= 4:
        try:
            sub = int(code[3])  # 4th character should be 0-9
            return sub <= 9
        except (ValueError, IndexError):
            return False
    return False


def codes_contain_lipid_disorder(codes_str: str) -> bool:
    """Check a semicolon-separated code string for lipid disorder codes."""
    if not codes_str or pd.isna(codes_str):
        return False
    for code in codes_str.split(";"):
        if is_lipid_disorder(code.strip()):
            return True
    return False


def add_target_labels(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add binary label column ``label_lipid_disorder`` to visit-pair table.

    1 = Disorders of Lipid Metabolism present in next visit
    0 = absent
    """
    pairs = pairs.copy()
    pairs["label_lipid_disorder"] = pairs["diagnoses_codes_next"].apply(
        lambda x: int(codes_contain_lipid_disorder(x))
    )
    return pairs


def label_sanity_report(df: pd.DataFrame, output_dir: str) -> dict:
    """Print and save a sanity-check report on the target label distribution."""
    n_pos = int(df["label_lipid_disorder"].sum())
    n_neg = int(len(df) - n_pos)
    prevalence = n_pos / len(df) if len(df) > 0 else 0.0

    report = {
        "total_records": len(df),
        "positive": n_pos,
        "negative": n_neg,
        "prevalence": round(prevalence, 4),
    }

    # Example positive/negative visits
    pos_examples = df[df["label_lipid_disorder"] == 1].head(3)[["pair_id", "hadm_id_current", "hadm_id_next", "diagnoses_codes_next"]].to_dict("records")
    neg_examples = df[df["label_lipid_disorder"] == 0].head(3)[["pair_id", "hadm_id_current", "hadm_id_next", "diagnoses_codes_next"]].to_dict("records")
    report["example_positive"] = pos_examples
    report["example_negative"] = neg_examples

    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    save_json(report, odir / "target_label_report.json")
    save_dataframe(
        pd.DataFrame([
            {"label_lipid_disorder": 0, "count": n_neg},
            {"label_lipid_disorder": 1, "count": n_pos},
        ]),
        odir / "label_distribution.csv",
    )

    logger.info(
        "Target label — total=%d  pos=%d  neg=%d  prevalence=%.2f%%",
        len(df), n_pos, n_neg, prevalence * 100,
    )
    return report
