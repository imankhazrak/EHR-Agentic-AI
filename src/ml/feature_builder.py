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


DEFAULT_CODE_COLS = [
    "diagnoses_codes_current",
    "procedures_codes_current",
    "medications_current",
]


def row_to_code_tokens(row: pd.Series, code_cols: list[str] | None = None) -> str:
    """Join semicolon-separated code columns into a whitespace token string."""
    cols = code_cols or DEFAULT_CODE_COLS
    parts: list[str] = []
    for col in cols:
        val = row.get(col, "")
        if pd.notna(val) and val:
            parts.extend(str(val).split(";"))
    return " ".join(parts)


def dataframe_to_code_token_text(df: pd.DataFrame, code_cols: list[str] | None = None) -> pd.Series:
    """Return one tokenized bag-of-codes text string per row."""
    return df.apply(lambda row: row_to_code_tokens(row, code_cols=code_cols), axis=1)


def fit_bag_of_codes_vectorizer(
    train_df: pd.DataFrame,
    code_cols: list[str] | None = None,
) -> CountVectorizer:
    """Fit a bag-of-codes CountVectorizer on train rows only."""
    train_text = dataframe_to_code_token_text(train_df, code_cols=code_cols)
    vec = CountVectorizer(binary=True, token_pattern=r"[^\s]+")
    vec.fit(train_text)
    return vec


def transform_bag_of_codes(
    df: pd.DataFrame,
    vectorizer: CountVectorizer,
    code_cols: list[str] | None = None,
):
    """Transform rows into sparse bag-of-codes matrix using a fitted vectorizer."""
    text = dataframe_to_code_token_text(df, code_cols=code_cols)
    return vectorizer.transform(text)


def bag_of_codes_to_dataframe(
    matrix,
    vectorizer: CountVectorizer,
    *,
    prefix: str = "feat_",
) -> pd.DataFrame:
    """Convert sparse bag-of-codes matrix to a dense DataFrame with named columns."""
    feature_names = vectorizer.get_feature_names_out()
    out = pd.DataFrame.sparse.from_spmatrix(matrix, columns=[f"{prefix}{n}" for n in feature_names])
    return out


def build_feature_keep_mask_from_prefixes(
    feature_names: list[str],
    forbidden_prefixes: list[str],
    *,
    feature_prefix: str = "feat_",
) -> np.ndarray:
    """Return boolean keep-mask that excludes features matching forbidden ICD prefixes."""
    prefixes = [str(p).strip().replace(".", "") for p in forbidden_prefixes if str(p).strip()]
    keep = np.ones(len(feature_names), dtype=bool)
    if not prefixes:
        return keep
    for idx, feat in enumerate(feature_names):
        token = feat
        if token.startswith(feature_prefix):
            token = token[len(feature_prefix):]
        if any(token.startswith(p) for p in prefixes):
            keep[idx] = False
    return keep


def build_bag_of_codes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    code_cols: list[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    """Build bag-of-codes features (multi-hot over semicolon-separated tokens).

    Combines diagnoses_codes_current, procedures_codes_current, medications_current
    into one token string per row, then applies CountVectorizer with binary=True.
    """
    cols = code_cols or DEFAULT_CODE_COLS
    vec = fit_bag_of_codes_vectorizer(train_df, code_cols=cols)
    X_train = transform_bag_of_codes(train_df, vec, code_cols=cols)
    X_test = transform_bag_of_codes(test_df, vec, code_cols=cols)

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
