"""Build train/test JSONL for fine-tuning under data/mimic III/date_ready_for_finetuing.

This script reuses the repository preprocessing flow, merges PATIENTS columns
into train/test by SUBJECT_ID, exports JSONL files, and writes merge/dimension
verification into merge_verification.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.scripts.run_preprocessing import main as run_preprocessing_main
from src.utils.config_utils import load_config
from src.utils.io import save_json


RAW_TABLES = (
    "ADMISSIONS",
    "DIAGNOSES_ICD",
    "PROCEDURES_ICD",
    "PRESCRIPTIONS",
    "D_ICD_DIAGNOSES",
    "D_ICD_PROCEDURES",
    "PATIENTS",
)


def _table_dimensions(csv_gz_path: Path, chunksize: int = 250_000) -> dict[str, int]:
    """Return row/column counts for a gzipped CSV using low memory."""
    header = pd.read_csv(csv_gz_path, nrows=0)
    n_cols = len(header.columns)
    n_rows = 0
    for chunk in pd.read_csv(csv_gz_path, chunksize=chunksize, low_memory=False):
        n_rows += len(chunk)
    return {"rows": int(n_rows), "cols": int(n_cols)}


def _count_jsonl_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _merge_patients(split_df: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
    patients = patients_df.copy()
    dedup = patients.drop_duplicates(subset=["SUBJECT_ID"], keep="first")
    before = len(split_df)
    merged = split_df.merge(dedup, on="SUBJECT_ID", how="left", suffixes=("", "_patient"))
    if len(merged) != before:
        raise RuntimeError("Patient merge changed split row count; expected left merge row parity.")
    return merged


def main(config_path: str = "configs/default.yaml", overrides: list[str] | None = None) -> None:
    cfg = load_config(config_path, overrides)
    run_preprocessing_main(config_path, overrides)

    root = Path(__file__).resolve().parents[2]
    raw_dir = root / cfg["paths"]["mimic_raw"]
    interim_dir = root / cfg["paths"]["interim"]
    processed_dir = root / cfg["paths"]["processed"]
    out_dir = root / "data/mimic III/date_ready_for_finetuing"

    # Source dimensions
    source_dimensions: dict[str, Any] = {}
    for tname in RAW_TABLES:
        fp = raw_dir / f"{tname}.csv.gz"
        if not fp.exists():
            raise FileNotFoundError(f"Missing raw table: {fp}")
        source_dimensions[tname] = _table_dimensions(fp)

    # Interim merge dimensions (admissions before/after code joins)
    admissions = pd.read_csv(interim_dir / "admissions.csv")
    diagnoses = pd.read_csv(interim_dir / "diagnoses.csv")
    procedures = pd.read_csv(interim_dir / "procedures.csv")
    prescriptions = pd.read_csv(interim_dir / "prescriptions.csv")
    patients = pd.read_csv(interim_dir / "patients.csv")
    patient_visits = pd.read_csv(interim_dir / "patient_visits.csv")
    visit_pairs = pd.read_csv(interim_dir / "visit_pairs.csv")

    dx_agg = diagnoses.dropna(subset=["ICD9_CODE"]).groupby("HADM_ID")["ICD9_CODE"].nunique().reset_index()
    px_agg = procedures.dropna(subset=["ICD9_CODE"]).groupby("HADM_ID")["ICD9_CODE"].nunique().reset_index()
    rx_agg = prescriptions.dropna(subset=["DRUG"]).groupby("HADM_ID")["DRUG"].nunique().reset_index()
    joined = admissions[["HADM_ID"]].copy()
    joined = joined.merge(dx_agg, on="HADM_ID", how="left")
    joined = joined.merge(px_agg, on="HADM_ID", how="left")
    joined = joined.merge(rx_agg, on="HADM_ID", how="left")

    # Split + JSONL export
    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    full_dataset = pd.read_csv(processed_dir / "full_dataset.csv")

    train_merged = _merge_patients(train, patients)
    test_merged = _merge_patients(test, patients)

    train_jsonl = out_dir / "train.jsonl"
    test_jsonl = out_dir / "test.jsonl"
    _write_jsonl(train_merged, train_jsonl)
    _write_jsonl(test_merged, test_jsonl)

    train_lines = _count_jsonl_lines(train_jsonl)
    test_lines = _count_jsonl_lines(test_jsonl)
    required_keys = ("pair_id", "SUBJECT_ID", "narrative_current", "label_lipid_disorder")

    verification = {
        "paths": {
            "raw_dir": str(raw_dir),
            "interim_dir": str(interim_dir),
            "processed_dir": str(processed_dir),
            "output_dir": str(out_dir),
        },
        "source_table_dimensions": source_dimensions,
        "interim_dimensions": {
            "admissions_rows_before_join": int(len(admissions)),
            "admissions_rows_after_join": int(len(joined)),
            "patient_visits_rows": int(len(patient_visits)),
            "patient_visits_unique_subject_id": int(patient_visits["SUBJECT_ID"].nunique()),
            "visit_pairs_rows": int(len(visit_pairs)),
            "visit_pairs_unique_pair_id": int(visit_pairs["pair_id"].nunique()),
        },
        "split_dimensions": {
            "train_rows": int(len(train_merged)),
            "test_rows": int(len(test_merged)),
            "full_dataset_rows": int(len(full_dataset)),
            "train_plus_test_equals_full_dataset": bool(len(train_merged) + len(test_merged) == len(full_dataset)),
            "train_label_prevalence": float(train_merged["label_lipid_disorder"].mean()),
            "test_label_prevalence": float(test_merged["label_lipid_disorder"].mean()),
        },
        "patient_merge_checks": {
            "patients_rows": int(len(patients)),
            "patients_unique_subject_id": int(patients["SUBJECT_ID"].nunique()),
            "train_rows_after_merge": int(len(train_merged)),
            "test_rows_after_merge": int(len(test_merged)),
            "train_missing_patient_rows": int(train_merged["GENDER"].isna().sum()) if "GENDER" in train_merged.columns else None,
            "test_missing_patient_rows": int(test_merged["GENDER"].isna().sum()) if "GENDER" in test_merged.columns else None,
        },
        "jsonl_parity_checks": {
            "train_jsonl_lines": int(train_lines),
            "test_jsonl_lines": int(test_lines),
            "train_jsonl_matches_rows": bool(train_lines == len(train_merged)),
            "test_jsonl_matches_rows": bool(test_lines == len(test_merged)),
            "required_keys": list(required_keys),
            "train_first_row_has_required_keys": bool(
                all(k in train_merged.columns for k in required_keys) and len(train_merged) > 0
            ),
            "test_first_row_has_required_keys": bool(
                all(k in test_merged.columns for k in required_keys) and len(test_merged) > 0
            ),
        },
    }

    save_json(verification, out_dir / "merge_verification.json")
    print(f"Wrote {train_jsonl}")
    print(f"Wrote {test_jsonl}")
    print(f"Wrote {out_dir / 'merge_verification.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build date_ready_for_finetuing JSONL files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.overrides)
