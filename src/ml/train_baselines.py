"""Train classical ML baselines: Decision Tree, Logistic Regression, Random Forest.

Two settings:
  1. Fully supervised — train on full training set
  2. Few-shot — train on N total exemplars (default N=6)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {
    "decision_tree": lambda seed: DecisionTreeClassifier(random_state=seed),
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
    "random_forest": lambda seed: RandomForestClassifier(n_estimators=100, random_state=seed),
}


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> Any:
    """Train a single ML model and return the fitted estimator."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    clf = MODEL_REGISTRY[model_name](seed)
    clf.fit(X_train, y_train)
    logger.info("Trained %s on %d samples", model_name, X_train.shape[0])
    return clf


def train_all_baselines(
    model_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train all requested baselines. Returns dict of name → fitted model."""
    models = {}
    for name in model_names:
        models[name] = train_model(name, X_train, y_train, seed=seed)
    return models
