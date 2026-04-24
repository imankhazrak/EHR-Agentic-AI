"""Build a per-patient MIMIC-III dataset with narrative summaries.

Usage:
    python -m src.scripts.build_mimiciii_per_patient_narrative_dataset \
      --raw-dir "data/mimic III/physionet.org/files/mimiciii/1.4" \
      --output "data/mimic III/per_patient_narrative_merged/mimiciii_per_patient_narrative.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.scripts.build_mimiciii_per_patient_dataset import build_per_patient
from src.utils.io import read_mimic_table, save_dataframe
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _group_unique_values(df: pd.DataFrame, id_col: str, value_col: str, limit: int) -> pd.DataFrame:
    """Collect up to `limit` unique values per patient as a list."""
    tmp = df.dropna(subset=[id_col, value_col])[[id_col, value_col]].copy()
    tmp[value_col] = tmp[value_col].astype(str).str.strip()
    tmp = tmp[tmp[value_col] != ""]
    if tmp.empty:
        return pd.DataFrame(columns=[id_col, value_col])
    out = (
        tmp.drop_duplicates()
        .groupby(id_col)[value_col]
        .apply(lambda s: list(s.head(limit)))
        .reset_index()
    )
    return out


def _format_narrative(
    row: pd.Series,
    max_dx: int,
    max_px: int,
    max_drug: int,
) -> str:
    """Build a compact readable narrative string for one patient."""
    dx_items = row.get("dx_titles", [])
    px_items = row.get("px_titles", [])
    drug_items = row.get("drug_names", [])

    if not isinstance(dx_items, list):
        dx_items = []
    if not isinstance(px_items, list):
        px_items = []
    if not isinstance(drug_items, list):
        drug_items = []

    dx_text = ", ".join(dx_items[:max_dx]) if dx_items else "None recorded"
    px_text = ", ".join(px_items[:max_px]) if px_items else "None recorded"
    drug_text = ", ".join(drug_items[:max_drug]) if drug_items else "None recorded"

    return (
        f"Patient {int(row['SUBJECT_ID'])}: "
        f"{row.get('n_admissions', 0)} admissions; "
        f"diagnoses: {dx_text}; "
        f"procedures: {px_text}; "
        f"medications: {drug_text}."
    )


def build_per_patient_narrative(
    raw_dir: str,
    output_path: str,
    max_dx_terms: int = 12,
    max_px_terms: int = 12,
    max_drug_terms: int = 20,
) -> pd.DataFrame:
    """Build per-patient aggregate features plus narrative summary text."""
    base = build_per_patient(raw_dir=raw_dir, output_path=output_path.replace(".csv", "_tmp.csv"))
    id_col = "SUBJECT_ID"

    dx = read_mimic_table(raw_dir, "DIAGNOSES_ICD", usecols=["SUBJECT_ID", "ICD9_CODE"])
    d_dx = read_mimic_table(raw_dir, "D_ICD_DIAGNOSES", usecols=["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"])
    d_dx["ICD9_CODE"] = d_dx["ICD9_CODE"].astype(str).str.strip()
    d_dx["dx_title"] = d_dx["LONG_TITLE"].fillna(d_dx["SHORT_TITLE"]).fillna("").astype(str).str.strip()
    dx["ICD9_CODE"] = dx["ICD9_CODE"].astype(str).str.strip()
    dx = dx.merge(d_dx[["ICD9_CODE", "dx_title"]], on="ICD9_CODE", how="left")
    dx["dx_title"] = dx["dx_title"].replace("", pd.NA).fillna(dx["ICD9_CODE"])

    px = read_mimic_table(raw_dir, "PROCEDURES_ICD", usecols=["SUBJECT_ID", "ICD9_CODE"])
    d_px = read_mimic_table(raw_dir, "D_ICD_PROCEDURES", usecols=["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"])
    d_px["ICD9_CODE"] = d_px["ICD9_CODE"].astype(str).str.strip()
    d_px["px_title"] = d_px["LONG_TITLE"].fillna(d_px["SHORT_TITLE"]).fillna("").astype(str).str.strip()
    px["ICD9_CODE"] = px["ICD9_CODE"].astype(str).str.strip()
    px = px.merge(d_px[["ICD9_CODE", "px_title"]], on="ICD9_CODE", how="left")
    px["px_title"] = px["px_title"].replace("", pd.NA).fillna(px["ICD9_CODE"])

    rx = read_mimic_table(raw_dir, "PRESCRIPTIONS", usecols=["SUBJECT_ID", "DRUG"])
    rx["DRUG"] = rx["DRUG"].astype(str).str.strip()

    dx_group = _group_unique_values(dx.rename(columns={"dx_title": "dx_titles"}), id_col, "dx_titles", max_dx_terms)
    px_group = _group_unique_values(px.rename(columns={"px_title": "px_titles"}), id_col, "px_titles", max_px_terms)
    rx_group = _group_unique_values(rx.rename(columns={"DRUG": "drug_names"}), id_col, "drug_names", max_drug_terms)

    out = base.merge(dx_group, on=id_col, how="left")
    out = out.merge(px_group, on=id_col, how="left")
    out = out.merge(rx_group, on=id_col, how="left")

    out["patient_narrative"] = out.apply(
        _format_narrative,
        axis=1,
        max_dx=max_dx_terms,
        max_px=max_px_terms,
        max_drug=max_drug_terms,
    )
    out = out.drop(columns=["dx_titles", "px_titles", "drug_names"])

    tmp_path = Path(output_path.replace(".csv", "_tmp.csv"))
    if tmp_path.exists():
        tmp_path.unlink()

    save_dataframe(out, Path(output_path))
    logger.info("Built per-patient narrative dataset with %d rows and %d columns", len(out), len(out.columns))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one-row-per-patient MIMIC-III dataset with narrative text")
    parser.add_argument("--raw-dir", required=True, help="Directory containing MIMIC-III CSV/CSV.GZ files")
    parser.add_argument(
        "--output",
        default="data/mimic III/per_patient_narrative_merged/mimiciii_per_patient_narrative.csv",
        help="Output CSV path",
    )
    parser.add_argument("--max-dx-terms", type=int, default=12, help="Max diagnosis terms included in narrative")
    parser.add_argument("--max-px-terms", type=int, default=12, help="Max procedure terms included in narrative")
    parser.add_argument("--max-drug-terms", type=int, default=20, help="Max medication terms included in narrative")
    args = parser.parse_args()

    build_per_patient_narrative(
        raw_dir=args.raw_dir,
        output_path=args.output,
        max_dx_terms=args.max_dx_terms,
        max_px_terms=args.max_px_terms,
        max_drug_terms=args.max_drug_terms,
    )


if __name__ == "__main__":
    main()
