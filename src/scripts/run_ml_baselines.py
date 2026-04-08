"""Script: Train and evaluate ML baselines.

Usage:
    python -m src.scripts.run_ml_baselines --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config_utils import load_config
from src.utils.random_utils import set_seed
from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger
from src.ml.feature_builder import build_features
from src.ml.train_baselines import train_all_baselines
from src.ml.eval_baselines import evaluate_all
from src.data.exemplar_selector import select_exemplars

logger = get_logger(__name__, log_file="data/outputs/ml_baselines.log")


def main(config_path: str = "configs/default.yaml", overrides: list | None = None):
    cfg = load_config(config_path, overrides)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    processed_dir = cfg["paths"]["processed"]
    output_dir = cfg["paths"]["outputs"]
    ml_cfg = cfg.get("ml", {})
    model_names = ml_cfg.get("models", ["decision_tree", "logistic_regression", "random_forest"])
    feature_type = ml_cfg.get("feature_type", "bag_of_codes")

    train_df = pd.read_csv(Path(processed_dir) / "train.csv")
    test_df = pd.read_csv(Path(processed_dir) / "test.csv")

    y_train = train_df["label_lipid_disorder"].values
    y_test = test_df["label_lipid_disorder"].values

    # Build features
    X_train, X_test = build_features(train_df, test_df, feature_type=feature_type)

    all_results = []

    # ---- Fully supervised ----
    logger.info("=== Fully Supervised ML Baselines ===")
    models_full = train_all_baselines(model_names, X_train, y_train, seed=seed)
    results_full = evaluate_all(models_full, X_test, y_test, "fully_supervised", output_dir)
    all_results.append(results_full)

    # ---- Few-shot ----
    logger.info("=== Few-Shot ML Baselines ===")
    few_shot_n = ml_cfg.get("few_shot_n", 6)
    n_pos = few_shot_n // 2
    n_neg = few_shot_n - n_pos

    fs_exemplars = select_exemplars(
        train_df, n_positive=n_pos, n_negative=n_neg,
        strategy="random_balanced", seed=seed,
    )
    save_dataframe(fs_exemplars[["pair_id", "label_lipid_disorder"]], Path(output_dir) / "ml_few_shot_exemplar_ids.csv")

    # Build features for just the few-shot training subset
    X_fs, X_test_fs = build_features(fs_exemplars, test_df, feature_type=feature_type)
    y_fs = fs_exemplars["label_lipid_disorder"].values

    models_fs = train_all_baselines(model_names, X_fs, y_fs, seed=seed)
    results_fs = evaluate_all(models_fs, X_test_fs, y_test, "few_shot", output_dir)
    all_results.append(results_fs)

    # Combine results
    combined = pd.concat(all_results, ignore_index=True)
    save_dataframe(combined, Path(output_dir) / "ml_results.csv")
    logger.info("ML baselines complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML baselines")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.overrides)
