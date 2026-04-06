"""Step 3 — Create adjacent visit pairs for next-visit prediction.

For each patient with ≥ min_visits hospital stays, create
(visit_t, visit_{t+1}) pairs.  The current visit provides input features;
the next visit provides the prediction label.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_pairs(interim_dir: str, min_visits: int = 2) -> pd.DataFrame:
    """Build adjacent visit pairs.

    Returns DataFrame with columns:
        pair_id, SUBJECT_ID,
        hadm_id_current, admittime_current, diagnoses_codes_current,
        procedures_codes_current, medications_current,
        hadm_id_next, admittime_next, diagnoses_codes_next,
        procedures_codes_next, medications_next
    """
    idir = Path(interim_dir)
    visits = pd.read_csv(idir / "patient_visits.csv", parse_dates=["ADMITTIME", "DISCHTIME"])
    visits = visits.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)

    # Filter patients with >= min_visits
    counts = visits.groupby("SUBJECT_ID")["HADM_ID"].count()
    eligible = counts[counts >= min_visits].index
    visits = visits[visits["SUBJECT_ID"].isin(eligible)].copy()
    logger.info("Patients with >= %d visits: %d", min_visits, len(eligible))

    # Self-join: current visit with its immediate successor
    visits["_next_order"] = visits["visit_order"] + 1
    pairs = visits.merge(
        visits,
        left_on=["SUBJECT_ID", "_next_order"],
        right_on=["SUBJECT_ID", "visit_order"],
        suffixes=("_current", "_next"),
    )

    # Clean up columns
    pairs = pairs.rename(columns={
        "HADM_ID_current": "hadm_id_current",
        "HADM_ID_next": "hadm_id_next",
        "ADMITTIME_current": "admittime_current",
        "ADMITTIME_next": "admittime_next",
        "diagnoses_codes_current": "diagnoses_codes_current",
        "diagnoses_codes_next": "diagnoses_codes_next",
        "procedures_codes_current": "procedures_codes_current",
        "procedures_codes_next": "procedures_codes_next",
        "medications_current": "medications_current",
        "medications_next": "medications_next",
    })

    keep_cols = [
        "SUBJECT_ID",
        "hadm_id_current", "admittime_current",
        "diagnoses_codes_current", "procedures_codes_current", "medications_current",
        "hadm_id_next", "admittime_next",
        "diagnoses_codes_next", "procedures_codes_next", "medications_next",
    ]
    pairs = pairs[[c for c in keep_cols if c in pairs.columns]].copy()
    pairs = pairs.sort_values(["SUBJECT_ID", "admittime_current"]).reset_index(drop=True)
    pairs.insert(0, "pair_id", range(len(pairs)))

    save_dataframe(pairs, idir / "visit_pairs.csv")
    logger.info("Created %d visit pairs", len(pairs))
    return pairs
