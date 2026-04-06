"""Step 1 — Extract and lightly clean the MIMIC-III tables we need.

Tables used:
  ADMISSIONS, PATIENTS, DIAGNOSES_ICD, PROCEDURES_ICD, PRESCRIPTIONS,
  D_ICD_DIAGNOSES, D_ICD_PROCEDURES
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.io import read_mimic_table, save_dataframe
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def extract_all(raw_dir: str, interim_dir: str) -> Dict[str, pd.DataFrame]:
    """Read raw MIMIC-III CSVs and write lightly-cleaned parquets / CSVs to *interim_dir*.

    Returns a dict of table name → DataFrame for downstream use.
    """
    out = Path(interim_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables: Dict[str, pd.DataFrame] = {}

    # ---- Admissions ----
    adm = read_mimic_table(
        raw_dir, "ADMISSIONS",
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "HOSPITAL_EXPIRE_FLAG"],
    )
    adm["ADMITTIME"] = pd.to_datetime(adm["ADMITTIME"])
    adm["DISCHTIME"] = pd.to_datetime(adm["DISCHTIME"])
    tables["admissions"] = adm
    save_dataframe(adm, out / "admissions.csv")

    # ---- Diagnoses ----
    dx = read_mimic_table(
        raw_dir, "DIAGNOSES_ICD",
        usecols=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
    )
    dx["ICD9_CODE"] = dx["ICD9_CODE"].astype(str).str.strip()
    tables["diagnoses"] = dx
    save_dataframe(dx, out / "diagnoses.csv")

    # ---- Procedures ----
    px = read_mimic_table(
        raw_dir, "PROCEDURES_ICD",
        usecols=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
    )
    px["ICD9_CODE"] = px["ICD9_CODE"].astype(str).str.strip()
    tables["procedures"] = px
    save_dataframe(px, out / "procedures.csv")

    # ---- Prescriptions ----
    rx = read_mimic_table(
        raw_dir, "PRESCRIPTIONS",
        usecols=["SUBJECT_ID", "HADM_ID", "DRUG", "DRUG_TYPE", "FORMULARY_DRUG_CD", "GSN", "NDC"],
    )
    rx["DRUG"] = rx["DRUG"].astype(str).str.strip()
    tables["prescriptions"] = rx
    save_dataframe(rx, out / "prescriptions.csv")

    # ---- Diagnosis dictionary ----
    d_dx = read_mimic_table(
        raw_dir, "D_ICD_DIAGNOSES",
        usecols=["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"],
    )
    d_dx["ICD9_CODE"] = d_dx["ICD9_CODE"].astype(str).str.strip()
    tables["d_icd_diagnoses"] = d_dx
    save_dataframe(d_dx, out / "d_icd_diagnoses.csv")

    # ---- Procedure dictionary ----
    d_px = read_mimic_table(
        raw_dir, "D_ICD_PROCEDURES",
        usecols=["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"],
    )
    d_px["ICD9_CODE"] = d_px["ICD9_CODE"].astype(str).str.strip()
    tables["d_icd_procedures"] = d_px
    save_dataframe(d_px, out / "d_icd_procedures.csv")

    logger.info("Extraction complete — %d tables saved to %s", len(tables), out)
    return tables
