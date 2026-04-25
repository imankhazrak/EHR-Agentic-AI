"""Script: Run all preprocessing steps.

Usage:
    python -m src.scripts.run_preprocessing --config configs/default.yaml

After ``train.csv`` / ``test.csv`` exist under ``paths.processed``, you can build
the separate multitask copy (extra ICD-9–derived labels, new output folder only) with:

    python -m src.scripts.build_mimiciii_multitask_dataset \\
        --source-processed <paths.processed> \\
        --output-processed data/processed/mimiciii_multitask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config_utils import load_config
from src.utils.random_utils import set_seed
from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger
from src.data.extract_mimic_tables import extract_all
from src.data.build_patient_visits import build_visits
from src.data.build_visit_pairs import build_pairs
from src.data.build_target_labels import add_target_labels, label_sanity_report
from src.data.code_mappings import CodeMapper
from src.data.narrative_builder import add_narratives

logger = get_logger(__name__, log_file="data/outputs/preprocessing.log")


def main(config_path: str = "configs/default.yaml", overrides: list | None = None):
    cfg = load_config(config_path, overrides)
    set_seed(cfg.get("seed", 42))

    raw_dir = cfg["paths"]["mimic_raw"]
    interim_dir = cfg["paths"]["interim"]
    processed_dir = cfg["paths"]["processed"]
    output_dir = cfg["paths"]["outputs"]

    # ---- Step 1: Extract MIMIC tables ----
    logger.info("=== Step 1: Extract MIMIC-III tables ===")
    extract_all(raw_dir, interim_dir)

    # ---- Step 2: Build patient visits ----
    logger.info("=== Step 2: Build patient visits ===")
    build_visits(interim_dir)

    # ---- Step 3: Build visit pairs ----
    logger.info("=== Step 3: Build visit pairs ===")
    min_visits = cfg["data"]["min_visits"]
    pairs = build_pairs(interim_dir, min_visits=min_visits)

    # ---- Step 4: Add target labels ----
    logger.info("=== Step 4: Add target labels ===")
    pairs = add_target_labels(pairs)
    report = label_sanity_report(pairs, output_dir)
    logger.info("Label report: %s", report)

    max_pairs = cfg["data"].get("max_pairs")
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs.sample(n=max_pairs, random_state=cfg.get("seed", 42)).reset_index(drop=True)
        pairs["pair_id"] = range(len(pairs))
        logger.info("Subsampled to data.max_pairs=%d rows for this run", max_pairs)

    # ---- Step 5: Code-to-name mapping + narratives ----
    logger.info("=== Step 5: Build narratives ===")
    mapper = CodeMapper(interim_dir)
    ncfg = cfg.get("narrative", {})
    pairs = add_narratives(
        pairs, mapper,
        style=ncfg.get("style", "bullet"),
        max_diagnoses=ncfg.get("max_diagnoses", 50),
        max_medications=ncfg.get("max_medications", 50),
        max_procedures=ncfg.get("max_procedures", 50),
        empty_placeholder=ncfg.get("empty_placeholder", "None recorded"),
    )
    missing = mapper.report_missing()
    save_json(missing, Path(output_dir) / "code_mapping_missing.json")

    # ---- Step 6: Train/Test split ----
    logger.info("=== Step 6: Create train/test split ===")
    test_size = cfg["data"]["test_size"]
    stratify = cfg["data"].get("stratify_test", True)

    if stratify:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            pairs, test_size=test_size, random_state=cfg["seed"],
            stratify=pairs["label_lipid_disorder"],
        )
    else:
        test_df = pairs.sample(n=test_size, random_state=cfg["seed"])
        train_df = pairs.drop(test_df.index)

    train_df = train_df.reset_index(drop=True).copy()
    test_df = test_df.reset_index(drop=True).copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    # Save
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    save_dataframe(train_df, Path(processed_dir) / "train.csv")
    save_dataframe(test_df, Path(processed_dir) / "test.csv")
    save_dataframe(pd.concat([train_df, test_df]), Path(processed_dir) / "full_dataset.csv")

    summary = {
        "total_pairs": len(pairs),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_prevalence": round(float(train_df["label_lipid_disorder"].mean()), 4),
        "test_prevalence": round(float(test_df["label_lipid_disorder"].mean()), 4),
    }
    save_json(summary, Path(output_dir) / "processed_dataset_summary.json")
    logger.info("Preprocessing complete: %s", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MIMIC-III preprocessing")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.overrides)
