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
from typing import List, Sequence

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


# ---------------------------------------------------------------------------
# Multitask labels (prompt_v3) — all derived from MIMIC-III ICD-9-CM in
# ``diagnoses_codes_current`` / ``diagnoses_codes_next`` (aggregated from
# DIAGNOSES_ICD; see ``build_patient_visits`` / ``extract_mimic_tables``).
# ---------------------------------------------------------------------------


def normalize_icd9_code(code: str) -> str:
    """Strip spaces, remove dots, uppercase. Empty string if missing/invalid.

    MIMIC-III stores codes without periods (e.g. ``2780``). This also accepts
    dotted forms (``278.0`` → ``2780``) for prefix matching.
    """
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return ""
    s = str(code).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    s = s.replace(".", "").upper()
    return s


def codes_contain_prefix(codes: str, prefixes: Sequence[str]) -> bool:
    """True if any semicolon-separated ICD-9 token starts with one of *prefixes*.

    Tokens and prefix strings are normalized with :func:`normalize_icd9_code`.
    Match rule: ``normalized_token.startswith(prefix)`` for at least one prefix.
    Empty *codes* → False.
    """
    if codes is None or (isinstance(codes, float) and pd.isna(codes)):
        return False
    s = str(codes).strip()
    if not s or s.lower() in ("nan", "none"):
        return False
    pnorm = [normalize_icd9_code(str(p)) for p in prefixes]
    pnorm = [p for p in pnorm if p]
    if not pnorm:
        return False
    for raw in s.split(";"):
        n = normalize_icd9_code(raw)
        if not n:
            continue
        for p in pnorm:
            if n.startswith(p):
                return True
    return False


def add_multitask_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Append prompt_v3 multitask label columns; keeps all existing columns.

    All new labels are binary 0/1, derived from aggregated ICD-9 strings already
    on the frame (from real MIMIC-III ``DIAGNOSES_ICD`` / visit aggregation).

    Requires: ``label_lipid_disorder``, ``diagnoses_codes_current``,
    ``diagnoses_codes_next``.

    - ``label_lipid_next``: copy of ``label_lipid_disorder`` (same as existing
      next-visit lipid target).
    - ``*_current``: from ``diagnoses_codes_current``.
    - ``*_next`` (other tasks): from ``diagnoses_codes_next``.
    """
    out = df.copy()
    need = ("label_lipid_disorder", "diagnoses_codes_current", "diagnoses_codes_next")
    for c in need:
        if c not in out.columns:
            raise KeyError(f"add_multitask_labels: missing required column {c!r}")
    out["label_lipid_next"] = out["label_lipid_disorder"].astype(int)
    out["label_diabetes_current"] = out["diagnoses_codes_current"].apply(
        lambda x: int(codes_contain_prefix(x, ["250"]))
    )
    out["label_hypertension_current"] = out["diagnoses_codes_current"].apply(
        lambda x: int(codes_contain_prefix(x, ["401", "402", "403", "404", "405"]))
    )
    out["label_obesity_current"] = out["diagnoses_codes_current"].apply(
        lambda x: int(codes_contain_prefix(x, ["2780"]))
    )
    out["label_cardio_next"] = out["diagnoses_codes_next"].apply(
        lambda x: int(
            codes_contain_prefix(
                x, ["410", "411", "412", "413", "414", "428"]
            )
        )
    )
    out["label_kidney_next"] = out["diagnoses_codes_next"].apply(
        lambda x: int(codes_contain_prefix(x, ["584", "585", "586"]))
    )
    out["label_stroke_next"] = out["diagnoses_codes_next"].apply(
        lambda x: int(
            codes_contain_prefix(
                x, ["430", "431", "432", "433", "434", "435", "436"]
            )
        )
    )
    return out


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
