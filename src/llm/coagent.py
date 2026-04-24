"""EHR-CoAgent pipeline — full two-agent workflow.

Steps:
1. Run few-shot predictor on calibration/dev subset
2. Identify wrong predictions
3. Run critic on wrong predictions in batches
4. Consolidate critic feedback
5. Re-run predictor with feedback on the test set
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from src.llm.api_clients import LLMClient
from src.llm.predictor import run_predictions
from src.llm.critic import run_critic, consolidate_feedback
from src.utils.io import save_dataframe, save_text
from src.utils.logging_utils import get_logger
from src.utils.random_utils import set_seed

logger = get_logger(__name__)


def run_coagent_pipeline(
    client: LLMClient,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    demonstration_cases: str,
    cfg: Dict,
    output_dir: str = "data/outputs",
    prompt_template_dir: Optional[Union[str, Path]] = None,
    multitask: bool = False,
) -> pd.DataFrame:
    """Run the full EHR-CoAgent pipeline.

    Parameters
    ----------
    client : LLMClient
    train_df : DataFrame — training data (superset of calibration)
    test_df : DataFrame — held-out test set
    demonstration_cases : str — pre-formatted few-shot exemplars
    cfg : dict — coagent section of config
    output_dir : str

    Returns
    -------
    DataFrame — test-set predictions with critic-enhanced prompts
    """
    odir = Path(output_dir)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    cal_size = cfg.get("calibration_size", 200)
    n_wrong = cfg.get("n_wrong_samples", 30)
    batch_size = cfg.get("critic_batch_size", 10)
    n_rounds = cfg.get("n_critic_rounds", 3)
    consolidation = cfg.get("consolidation_method", "llm")

    # ---- Step 1: Run predictor on calibration subset ----
    logger.info("Step 1: Running few-shot predictor on calibration set (n=%d)", cal_size)
    cal_df = train_df.sample(n=min(cal_size, len(train_df)), random_state=seed).copy()

    cal_results = run_predictions(
        client=client,
        df=cal_df,
        mode="coagent_calibration",
        demonstration_cases=demonstration_cases,
        output_dir=output_dir,
        prompt_template_dir=prompt_template_dir,
        multitask=multitask,
    )

    # Merge true labels
    cal_merged = cal_df[["pair_id", "label_lipid_disorder", "narrative_current"]].merge(
        cal_results[["pair_id", "parsed_prediction", "reasoning"]],
        on="pair_id",
    )

    # ---- Step 2: Identify wrong predictions ----
    cal_merged["true_str"] = cal_merged["label_lipid_disorder"].map({1: "Yes", 0: "No"})
    wrong_mask = cal_merged["parsed_prediction"] != cal_merged["true_str"]
    wrong_df = cal_merged[wrong_mask].copy()
    logger.info("Step 2: Found %d wrong predictions out of %d calibration samples", len(wrong_df), len(cal_merged))

    if len(wrong_df) == 0:
        logger.warning("No wrong predictions found — skipping critic, using coagent mode without feedback")
        return run_predictions(
            client=client,
            df=test_df,
            mode="coagent",
            demonstration_cases=demonstration_cases,
            output_dir=output_dir,
            prompt_template_dir=prompt_template_dir,
            multitask=multitask,
        )

    # Sample wrong predictions
    wrong_sample = wrong_df.sample(n=min(n_wrong, len(wrong_df)), random_state=seed)
    save_dataframe(wrong_sample, odir / "critic_feedback" / "wrong_predictions_sampled.csv")

    # ---- Step 3: Run critic ----
    logger.info("Step 3: Running critic agent (%d rounds, batch_size=%d)", n_rounds, batch_size)
    feedbacks = run_critic(
        client=client,
        wrong_preds_df=wrong_sample,
        batch_size=batch_size,
        n_rounds=n_rounds,
        output_dir=str(odir / "critic_feedback"),
        prompt_template_dir=prompt_template_dir,
    )

    # ---- Step 4: Consolidate feedback ----
    logger.info("Step 4: Consolidating feedback (method=%s)", consolidation)
    consolidated = consolidate_feedback(
        client=client,
        feedbacks=feedbacks,
        method=consolidation,
        output_dir=str(odir / "critic_feedback"),
        prompt_template_dir=prompt_template_dir,
    )

    # ---- Step 5: Re-run predictor on test set with feedback ----
    logger.info("Step 5: Re-running predictor on test set with critic feedback")
    test_results = run_predictions(
        client=client,
        df=test_df,
        mode="coagent",
        demonstration_cases=demonstration_cases,
        critic_feedback=consolidated,
        output_dir=output_dir,
        prompt_template_dir=prompt_template_dir,
        multitask=multitask,
    )

    # Save the final augmented prompt for reference
    save_text(
        f"=== Demonstration Cases ===\n{demonstration_cases}\n\n=== Critic Feedback ===\n{consolidated}",
        odir / "prompts_used" / "coagent_augmented_context.txt",
    )

    return test_results
