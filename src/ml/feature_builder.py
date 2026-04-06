"""Build feature matrices for classical ML baselines.

Feature types:
  - bag_of_codes: multi-hot over all observed diagnosis / procedure / medication tokens
  - tfidf: TF-IDF over the narrative text
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_bag_of_codes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    code_cols: list[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    """Build bag-of-codes features (multi-hot over semicolon-separated tokens).

    Combines diagnoses_codes_current, procedures_codes_current, medications_current
    into one token string per row, then applies CountVectorizer with binary=True.
    """
    if code_cols is None:
        code_cols = ["diagnoses_codes_current", "procedures_codes_current", "medications_current"]

    def _to_tokens(row: pd.Series) -> str:
        parts = []
        for col in code_cols:
            val = row.get(col, "")
            if pd.notna(val) and val:
                parts.extend(str(val).split(";"))
        return " ".join(parts)

    train_text = train_df.apply(_to_tokens, axis=1)
    test_text = test_df.apply(_to_tokens, axis=1)

    vec = CountVectorizer(binary=True, token_pattern=r"[^\s]+")
    X_train = vec.fit_transform(train_text)
    X_test = vec.transform(test_text)

    logger.info("Bag-of-codes features: %d train × %d features", X_train.shape[0], X_train.shape[1])
    return X_train, X_test, vec


def build_tfidf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "narrative_current",
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Build TF-IDF features over the narrative text."""
    train_text = train_df[text_col].fillna("")
    test_text = test_df[text_col].fillna("")

    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vec.fit_transform(train_text)
    X_test = vec.transform(test_text)

    logger.info("TF-IDF features: %d train × %d features", X_train.shape[0], X_train.shape[1])
    return X_train, X_test, vec


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_type: str = "bag_of_codes",
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to the requested feature builder. Returns (X_train, X_test)."""
    if feature_type == "bag_of_codes":
        X_tr, X_te, _ = build_bag_of_codes(train_df, test_df)
    elif feature_type == "tfidf":
        X_tr, X_te, _ = build_tfidf(train_df, test_df)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    return X_tr, X_te
