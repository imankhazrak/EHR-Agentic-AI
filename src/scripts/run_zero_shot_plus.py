"""Script: Run zero-shot+ LLM predictions.

Usage:
    python -m src.scripts.run_zero_shot_plus --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config_utils import load_config
from src.utils.random_utils import set_seed
from src.utils.logging_utils import get_logger
from src.llm.api_clients import LLMClient
from src.llm.predictor import run_predictions
from src.evaluation.evaluate_llm_runs import evaluate_llm_results

logger = get_logger(__name__, log_file="data/outputs/zero_shot_plus.log")


def main(config_path: str = "configs/default.yaml", overrides: list | None = None):
    cfg = load_config(config_path, overrides)
    set_seed(cfg.get("seed", 42))

    processed_dir = cfg["paths"]["processed"]
    output_dir = cfg["paths"]["outputs"]

    test_df = pd.read_csv(Path(processed_dir) / "test.csv")
    lim = cfg.get("llm", {}).get("max_test_samples")
    if lim is not None and lim > 0:
        test_df = test_df.head(int(lim)).copy()

    client = LLMClient(cfg["llm"])

    logger.info("=== Running Zero-Shot+ ===")
    pred_df = run_predictions(
        client=client,
        df=test_df,
        mode="zero_shot_plus",
        output_dir=output_dir,
    )

    metrics = evaluate_llm_results(
        test_df=test_df,
        pred_df=pred_df,
        mode="zero_shot_plus",
        output_dir=output_dir,
        bootstrap_ci=cfg.get("evaluation", {}).get("bootstrap_ci", False),
    )
    logger.info("Zero-shot+ complete: %s", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.overrides)
