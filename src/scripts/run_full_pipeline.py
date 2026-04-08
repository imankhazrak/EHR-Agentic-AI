"""Script: Run the entire pipeline end-to-end.

Usage:
    python -m src.scripts.run_full_pipeline --config configs/default.yaml
"""

from __future__ import annotations

import argparse

from src.utils.config_utils import load_config
from src.utils.logging_utils import get_logger
from src.evaluation.summarize_results import collect_results

logger = get_logger(__name__, log_file="data/outputs/full_pipeline.log")


def main(config_path: str = "configs/default.yaml", overrides: list | None = None):
    cfg = load_config(config_path, overrides)
    output_dir = cfg["paths"]["outputs"]

    # ---- 1. Preprocessing ----
    logger.info("========== STEP 1: Preprocessing ==========")
    from src.scripts.run_preprocessing import main as run_preproc
    run_preproc(config_path, overrides)

    # ---- 2. ML Baselines ----
    logger.info("========== STEP 2: ML Baselines ==========")
    from src.scripts.run_ml_baselines import main as run_ml
    run_ml(config_path, overrides)

    # ---- 3. Zero-Shot ----
    logger.info("========== STEP 3: Zero-Shot ==========")
    from src.scripts.run_zero_shot import main as run_zs
    run_zs(config_path, overrides)

    # ---- 4. Zero-Shot+ ----
    logger.info("========== STEP 4: Zero-Shot+ ==========")
    from src.scripts.run_zero_shot_plus import main as run_zsp
    run_zsp(config_path, overrides)

    # ---- 5. Few-Shot ----
    logger.info("========== STEP 5: Few-Shot ==========")
    from src.scripts.run_few_shot import main as run_fs
    run_fs(config_path, overrides)

    # ---- 6. EHR-CoAgent ----
    logger.info("========== STEP 6: EHR-CoAgent ==========")
    from src.scripts.run_coagent import main as run_co
    run_co(config_path, overrides)

    # ---- 7. Summary ----
    logger.info("========== STEP 7: Summary ==========")
    summary = collect_results(output_dir)
    logger.info("Full pipeline complete.\n%s", summary.to_string(index=False) if len(summary) > 0 else "No results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.overrides)
