"""Script: Run EHR-CoAgent pipeline (predictor + critic + re-run).

Usage:
    python -m src.scripts.run_coagent --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config_utils import load_config
from src.utils.random_utils import set_seed
from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger
from src.llm.api_clients import LLMClient
from src.llm.coagent import run_coagent_pipeline
from src.data.exemplar_selector import select_exemplars, format_exemplar_block
from src.evaluation.evaluate_llm_runs import evaluate_llm_results

logger = get_logger(__name__, log_file="data/outputs/coagent.log")


def main(config_path: str = "configs/default.yaml", overrides: list | None = None):
    cfg = load_config(config_path, overrides)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    processed_dir = cfg["paths"]["processed"]
    output_dir = cfg["paths"]["outputs"]
    fs_cfg = cfg.get("few_shot", {})
    co_cfg = cfg.get("coagent", {})

    train_df = pd.read_csv(Path(processed_dir) / "train.csv")
    test_df = pd.read_csv(Path(processed_dir) / "test.csv")
    lim = cfg.get("llm", {}).get("max_test_samples")
    if lim is not None and lim > 0:
        test_df = test_df.head(int(lim)).copy()

    # Select exemplars (same as few-shot for comparability)
    exemplars = select_exemplars(
        train_df,
        n_positive=fs_cfg.get("n_positive", 3),
        n_negative=fs_cfg.get("n_negative", 3),
        strategy=fs_cfg.get("strategy", "random_balanced"),
        seed=fs_cfg.get("seed", seed),
    )
    demo_text = format_exemplar_block(exemplars)

    client = LLMClient(cfg["llm"])

    logger.info("=== Running EHR-CoAgent Pipeline ===")
    # Merge coagent-specific config with top-level seed
    co_cfg_full = {**co_cfg, "seed": seed}

    pred_df = run_coagent_pipeline(
        client=client,
        train_df=train_df,
        test_df=test_df,
        demonstration_cases=demo_text,
        cfg=co_cfg_full,
        output_dir=output_dir,
    )

    metrics = evaluate_llm_results(
        test_df=test_df,
        pred_df=pred_df,
        mode="coagent",
        output_dir=output_dir,
        bootstrap_ci=cfg.get("evaluation", {}).get("bootstrap_ci", False),
    )
    logger.info("EHR-CoAgent complete: %s", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.overrides)
