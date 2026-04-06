"""Step 2 — Build patient-level chronologically ordered hospital visits.

Output: one row per admission with aggregated codes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _aggregate_codes(df: pd.DataFrame, code_col: str, hadm_col: str = "HADM_ID") -> pd.DataFrame:
    """Group codes per admission into a semicolon-separated list."""
    return (
        df.dropna(subset=[code_col])
        .groupby(hadm_col)[code_col]
        .apply(lambda x: ";".join(x.unique()))
        .reset_index()
        .rename(columns={code_col: f"{code_col}_list"})
    )


def build_visits(interim_dir: str) -> pd.DataFrame:
    """Build a visit-level table with diagnoses, procedures, medications.

    Returns DataFrame with columns:
        SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME,
        diagnoses_codes, procedures_codes, medications
    """
    idir = Path(interim_dir)

    adm = pd.read_csv(idir / "admissions.csv", parse_dates=["ADMITTIME", "DISCHTIME"])
    dx = pd.read_csv(idir / "diagnoses.csv", dtype={"ICD9_CODE": str})
    px = pd.read_csv(idir / "procedures.csv", dtype={"ICD9_CODE": str})
    rx = pd.read_csv(idir / "prescriptions.csv", dtype={"DRUG": str})

    # Aggregate codes per admission
    dx_agg = _aggregate_codes(dx, "ICD9_CODE").rename(columns={"ICD9_CODE_list": "diagnoses_codes"})
    px_agg = _aggregate_codes(px, "ICD9_CODE").rename(columns={"ICD9_CODE_list": "procedures_codes"})
    rx_agg = (
        rx.dropna(subset=["DRUG"])
        .groupby("HADM_ID")["DRUG"]
        .apply(lambda x: ";".join(x.unique()))
        .reset_index()
        .rename(columns={"DRUG": "medications"})
    )

    # Merge onto admissions
    visits = adm[["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"]].copy()
    visits = visits.merge(dx_agg, on="HADM_ID", how="left")
    visits = visits.merge(px_agg, on="HADM_ID", how="left")
    visits = visits.merge(rx_agg, on="HADM_ID", how="left")

    # Fill missing aggregates
    for col in ["diagnoses_codes", "procedures_codes", "medications"]:
        visits[col] = visits[col].fillna("")

    # Sort chronologically
    visits = visits.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)

    # Add within-patient visit index
    visits["visit_order"] = visits.groupby("SUBJECT_ID").cumcount()

    save_dataframe(visits, idir / "patient_visits.csv")
    logger.info("Built %d visits for %d patients", len(visits), visits["SUBJECT_ID"].nunique())
    return visits
