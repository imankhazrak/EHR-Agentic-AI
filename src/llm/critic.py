"""Critic agent — analyse wrong predictions and generate instructional feedback.

The critic receives batches of (narrative, true_label, wrong_prediction) and
produces generalised criteria for improving the predictor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.llm.api_clients import LLMClient
from src.llm.prompt_builder import build_messages
from src.utils.io import save_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def format_critic_batch(
    batch_df: pd.DataFrame,
    narrative_col: str = "narrative_current",
    label_col: str = "label_lipid_disorder",
    pred_col: str = "parsed_prediction",
    reasoning_col: str = "reasoning",
) -> str:
    """Format a batch of wrong predictions for the critic prompt.

    Each sample becomes:
        ---
        Sample N:
        [Patient Medical History]
        <narrative>
        [Actual Outcome] Yes/No
        [Wrong Prediction] Yes/No
        [Predictor Reasoning] (if available)
        ---
    """
    lines = []
    for i, (_, row) in enumerate(batch_df.iterrows(), 1):
        true_label = "Yes" if row[label_col] == 1 else "No"
        pred = row.get(pred_col, "")
        reasoning = row.get(reasoning_col, "")

        lines.append(f"--- Sample {i} ---")
        lines.append("[Patient Medical History]")
        lines.append(str(row[narrative_col]))
        lines.append(f"[Actual Outcome] {true_label}")
        lines.append(f"[Wrong Prediction] {pred}")
        if reasoning:
            lines.append(f"[Predictor Reasoning] {reasoning[:500]}")
        lines.append("")

    return "\n".join(lines)


def run_critic(
    client: LLMClient,
    wrong_preds_df: pd.DataFrame,
    batch_size: int = 10,
    n_rounds: int = 3,
    output_dir: str = "data/outputs/critic_feedback",
    **kwargs,
) -> List[str]:
    """Run the critic agent over batches of wrong predictions.

    Parameters
    ----------
    client : LLMClient
    wrong_preds_df : DataFrame
        Must contain narrative, label, parsed_prediction, reasoning columns.
    batch_size : int
        Number of samples per critic batch.
    n_rounds : int
        Number of critic batches to run.

    Returns
    -------
    list of str — one feedback text per batch.
    """
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    # Sample batches
    total_available = len(wrong_preds_df)
    feedbacks: List[str] = []

    for round_idx in range(n_rounds):
        start = (round_idx * batch_size) % total_available
        end = start + batch_size
        if end > total_available:
            batch = pd.concat([wrong_preds_df.iloc[start:], wrong_preds_df.iloc[:end - total_available]])
        else:
            batch = wrong_preds_df.iloc[start:end]

        if len(batch) == 0:
            logger.warning("No samples for critic round %d", round_idx)
            continue

        batch_text = format_critic_batch(batch)
        save_text(batch_text, odir / f"critic_input_round_{round_idx}.txt")

        messages = build_messages(
            template_file="critic_batch_reflection.txt",
            extra_vars={"batch_data": batch_text},
        )

        try:
            resp = client.complete(messages, use_cache=False)
            feedback = resp.text.strip()
        except Exception as e:
            logger.error("Critic call failed for round %d: %s", round_idx, e)
            feedback = ""

        save_text(feedback, odir / f"critic_output_round_{round_idx}.txt")
        feedbacks.append(feedback)
        logger.info("Critic round %d — generated %d chars of feedback", round_idx, len(feedback))

    return feedbacks


def consolidate_feedback(
    client: LLMClient,
    feedbacks: List[str],
    method: str = "llm",
    output_dir: str = "data/outputs/critic_feedback",
) -> str:
    """Consolidate multiple critic outputs into a single instruction set.

    Parameters
    ----------
    method : str
        ``"llm"`` — use LLM to merge, ``"heuristic"`` — simple concatenation.

    Returns
    -------
    str — consolidated feedback text.
    """
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    all_text = "\n\n---\n\n".join(f"Batch {i+1}:\n{fb}" for i, fb in enumerate(feedbacks) if fb)

    if method == "heuristic" or not feedbacks:
        consolidated = all_text
    else:
        messages = build_messages(
            template_file="feedback_consolidation.txt",
            extra_vars={"all_critic_outputs": all_text},
        )
        try:
            resp = client.complete(messages, use_cache=False)
            consolidated = resp.text.strip()
        except Exception as e:
            logger.error("Consolidation LLM call failed: %s — falling back to heuristic", e)
            consolidated = all_text

    save_text(consolidated, odir / "consolidated_feedback.txt")
    logger.info("Consolidated feedback: %d chars", len(consolidated))
    return consolidated
