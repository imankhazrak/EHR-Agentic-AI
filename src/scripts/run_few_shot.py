"""Script: Run few-shot LLM predictions.

Usage:
    python -m src.scripts.run_few_shot --config configs/default.yaml
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
from src.llm.predictor import run_predictions
from src.data.exemplar_selector import select_exemplars, format_exemplar_block
from src.evaluation.evaluate_llm_runs import evaluate_llm_results

logger = get_logger(__name__, log_file="data/outputs/few_shot.log")


def main(config_path: str = "configs/default.yaml", overrides: list | None = None):
    cfg = load_config(config_path, overrides)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    processed_dir = cfg["paths"]["processed"]
    output_dir = cfg["paths"]["outputs"]
    fs_cfg = cfg.get("few_shot", {})

    train_df = pd.read_csv(Path(processed_dir) / "train.csv")
    test_df = pd.read_csv(Path(processed_dir) / "test.csv")
    lim = cfg.get("llm", {}).get("max_test_samples")
    if lim is not None and lim > 0:
        test_df = test_df.head(int(lim)).copy()

    # Select exemplars
    exemplars = select_exemplars(
        train_df,
        n_positive=fs_cfg.get("n_positive", 3),
        n_negative=fs_cfg.get("n_negative", 3),
        strategy=fs_cfg.get("strategy", "random_balanced"),
        seed=fs_cfg.get("seed", seed),
    )
    save_dataframe(exemplars[["pair_id", "label_lipid_disorder", "narrative_current"]],
                   Path(output_dir) / "llm_few_shot_exemplars.csv")

    demo_text = format_exemplar_block(exemplars)

    client = LLMClient(cfg["llm"])
    prompt_td = cfg["llm"].get("prompt_template_dir", "prompts_v2")

    logger.info("=== Running Few-Shot ===")
    pred_df = run_predictions(
        client=client,
        df=test_df,
        mode="few_shot",
        demonstration_cases=demo_text,
        output_dir=output_dir,
        prompt_template_dir=prompt_td,
        multitask=bool(cfg.get("llm", {}).get("multitask", False)),
    )

    metrics = evaluate_llm_results(
        test_df=test_df,
        pred_df=pred_df,
        mode="few_shot",
        output_dir=output_dir,
        bootstrap_ci=cfg.get("evaluation", {}).get("bootstrap_ci", False),
    )
    logger.info("Few-shot complete: %s", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.overrides)
