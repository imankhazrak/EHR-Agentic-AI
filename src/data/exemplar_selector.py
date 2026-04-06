"""Few-shot exemplar selection from training data.

Strategies:
  - random_balanced: random sample of N positive + N negative (default)
  - prevalence_matched: sample proportional to target prevalence
  - similarity: placeholder for embedding-based retrieval
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.random_utils import set_seed

logger = get_logger(__name__)


def select_exemplars(
    train_df: pd.DataFrame,
    n_positive: int = 3,
    n_negative: int = 3,
    strategy: str = "random_balanced",
    seed: int = 42,
    label_col: str = "label_lipid_disorder",
    narrative_col: str = "narrative_current",
) -> pd.DataFrame:
    """Select few-shot exemplars from training data.

    Returns a DataFrame subset with columns preserved.
    """
    set_seed(seed)

    pos = train_df[train_df[label_col] == 1]
    neg = train_df[train_df[label_col] == 0]

    if strategy == "random_balanced":
        n_pos = min(n_positive, len(pos))
        n_neg = min(n_negative, len(neg))
        sampled_pos = pos.sample(n=n_pos, random_state=seed)
        sampled_neg = neg.sample(n=n_neg, random_state=seed)
        exemplars = pd.concat([sampled_pos, sampled_neg]).reset_index(drop=True)

    elif strategy == "prevalence_matched":
        total = n_positive + n_negative
        prevalence = len(pos) / len(train_df) if len(train_df) > 0 else 0.5
        n_pos = max(1, round(total * prevalence))
        n_neg = total - n_pos
        n_pos = min(n_pos, len(pos))
        n_neg = min(n_neg, len(neg))
        sampled_pos = pos.sample(n=n_pos, random_state=seed)
        sampled_neg = neg.sample(n=n_neg, random_state=seed)
        exemplars = pd.concat([sampled_pos, sampled_neg]).reset_index(drop=True)

    elif strategy == "similarity":
        # Placeholder — falls back to random balanced
        logger.warning("Similarity-based selection not implemented; falling back to random_balanced")
        return select_exemplars(train_df, n_positive, n_negative, "random_balanced", seed, label_col, narrative_col)

    else:
        raise ValueError(f"Unknown exemplar strategy: {strategy}")

    logger.info(
        "Selected %d exemplars (%d pos, %d neg) with strategy=%s",
        len(exemplars),
        int(exemplars[label_col].sum()),
        int((exemplars[label_col] == 0).sum()),
        strategy,
    )
    return exemplars


def format_exemplar_block(
    exemplars: pd.DataFrame,
    label_col: str = "label_lipid_disorder",
    narrative_col: str = "narrative_current",
) -> str:
    """Format exemplars into a text block suitable for prompt insertion.

    Each exemplar is rendered as:
        Case N:
        <narrative>
        Outcome: Yes / No
    """
    lines = []
    for i, (_, row) in enumerate(exemplars.iterrows(), 1):
        label_str = "Yes" if row[label_col] == 1 else "No"
        lines.append(f"Case {i}:")
        lines.append(row[narrative_col])
        lines.append(f"Outcome: {label_str}")
        lines.append("")  # blank line separator
    return "\n".join(lines)
