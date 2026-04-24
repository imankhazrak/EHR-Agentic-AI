"""Build a single per-patient dataset (one row per SUBJECT_ID) from MIMIC-III.

Usage:
    python -m src.scripts.build_mimiciii_per_patient_dataset \
      --raw-dir "data/mimic III/physionet.org/files/mimiciii/1.4" \
      --output "data/processed/mimiciii_per_patient.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import read_mimic_table, save_dataframe
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _count_unique(df: pd.DataFrame, id_col: str, value_col: str, out_col: str) -> pd.DataFrame:
    """Count unique non-null values per patient."""
    if value_col not in df.columns:
        return pd.DataFrame(columns=[id_col, out_col])
    tmp = (
        df.dropna(subset=[id_col, value_col])[[id_col, value_col]]
        .drop_duplicates()
        .groupby(id_col, as_index=False)
        .size()
        .rename(columns={"size": out_col})
    )
    return tmp


def _count_rows(df: pd.DataFrame, id_col: str, out_col: str) -> pd.DataFrame:
    """Count rows per patient."""
    if id_col not in df.columns:
        return pd.DataFrame(columns=[id_col, out_col])
    return (
        df.dropna(subset=[id_col])[[id_col]]
        .groupby(id_col, as_index=False)
        .size()
        .rename(columns={"size": out_col})
    )


def build_per_patient(raw_dir: str, output_path: str, include_large_events: bool = False) -> pd.DataFrame:
    """Build one-row-per-patient aggregate dataset."""
    id_col = "SUBJECT_ID"

    patients = read_mimic_table(
        raw_dir,
        "PATIENTS",
        usecols=["SUBJECT_ID", "GENDER", "DOB", "DOD", "EXPIRE_FLAG"],
    ).copy()
    patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce")
    patients["DOD"] = pd.to_datetime(patients["DOD"], errors="coerce")

    admissions = read_mimic_table(
        raw_dir,
        "ADMISSIONS",
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "HOSPITAL_EXPIRE_FLAG"],
    ).copy()
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
    admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"], errors="coerce")

    base = patients[[id_col, "GENDER", "DOB", "DOD", "EXPIRE_FLAG"]].drop_duplicates(subset=[id_col]).copy()

    adm_agg = (
        admissions.groupby(id_col, as_index=False)
        .agg(
            n_admissions=("HADM_ID", "nunique"),
            first_admit_time=("ADMITTIME", "min"),
            last_discharge_time=("DISCHTIME", "max"),
            any_inhospital_death=("HOSPITAL_EXPIRE_FLAG", "max"),
        )
    )
    base = base.merge(adm_agg, on=id_col, how="left")

    icu = read_mimic_table(raw_dir, "ICUSTAYS", usecols=["SUBJECT_ID", "ICUSTAY_ID"])
    base = base.merge(_count_unique(icu, id_col, "ICUSTAY_ID", "n_icu_stays"), on=id_col, how="left")

    dx = read_mimic_table(raw_dir, "DIAGNOSES_ICD", usecols=["SUBJECT_ID", "ICD9_CODE"])
    base = base.merge(_count_rows(dx, id_col, "n_diagnosis_rows"), on=id_col, how="left")
    base = base.merge(_count_unique(dx, id_col, "ICD9_CODE", "n_unique_diagnosis_codes"), on=id_col, how="left")

    px = read_mimic_table(raw_dir, "PROCEDURES_ICD", usecols=["SUBJECT_ID", "ICD9_CODE"])
    base = base.merge(_count_rows(px, id_col, "n_procedure_rows"), on=id_col, how="left")
    base = base.merge(_count_unique(px, id_col, "ICD9_CODE", "n_unique_procedure_codes"), on=id_col, how="left")

    rx = read_mimic_table(raw_dir, "PRESCRIPTIONS", usecols=["SUBJECT_ID", "DRUG"])
    base = base.merge(_count_rows(rx, id_col, "n_prescription_rows"), on=id_col, how="left")
    base = base.merge(_count_unique(rx, id_col, "DRUG", "n_unique_drugs"), on=id_col, how="left")

    if include_large_events:
        logger.info("include_large_events=True, reading large event tables for row counts")
        labevents = read_mimic_table(raw_dir, "LABEVENTS", usecols=["SUBJECT_ID"])
        noteevents = read_mimic_table(raw_dir, "NOTEEVENTS", usecols=["SUBJECT_ID"])
        chartevents = read_mimic_table(raw_dir, "CHARTEVENTS", usecols=["SUBJECT_ID"])
        base = base.merge(_count_rows(labevents, id_col, "n_labevents_rows"), on=id_col, how="left")
        base = base.merge(_count_rows(noteevents, id_col, "n_noteevents_rows"), on=id_col, how="left")
        base = base.merge(_count_rows(chartevents, id_col, "n_chartevents_rows"), on=id_col, how="left")

    count_cols = [c for c in base.columns if c.startswith("n_") or c.startswith("any_")]
    for col in count_cols:
        base[col] = base[col].fillna(0).astype(int)

    base["has_multiple_admissions"] = (base["n_admissions"] > 1).astype(int)
    # MIMIC date shifting can produce edge values that overflow vectorized datetime subtraction.
    def _safe_survival_days(row: pd.Series) -> float | None:
        dob = row["DOB"]
        dod = row["DOD"]
        if pd.isna(dob) or pd.isna(dod):
            return None
        try:
            return float((dod - dob).days)
        except (OverflowError, ValueError):
            return None

    base["survival_days_from_dob_to_dod"] = base.apply(_safe_survival_days, axis=1)

    out = base.sort_values(id_col).reset_index(drop=True)
    save_dataframe(out, Path(output_path))
    logger.info("Built per-patient dataset with %d rows and %d columns", len(out), len(out.columns))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one-row-per-patient MIMIC-III dataset")
    parser.add_argument("--raw-dir", required=True, help="Directory containing MIMIC-III CSV/CSV.GZ files")
    parser.add_argument(
        "--output",
        default="data/processed/mimiciii_per_patient.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--include-large-events",
        action="store_true",
        help="Also include row counts from large LABEVENTS/NOTEEVENTS/CHARTEVENTS tables",
    )
    args = parser.parse_args()

    build_per_patient(
        raw_dir=args.raw_dir,
        output_path=args.output,
        include_large_events=args.include_large_events,
    )


if __name__ == "__main__":
    main()
