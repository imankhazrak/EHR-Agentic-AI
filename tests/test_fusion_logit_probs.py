"""Unit tests for logit pair scoring and fusion threshold helper."""

from __future__ import annotations

import numpy as np
import torch

from src.llm.logit_pair_scorer import logits_to_yes_no_probs
from src.ml.fusion.thresholds import pick_threshold


def test_logits_to_yes_no_probs_softmax():
    vocab = 20
    logits = torch.zeros(1, vocab)
    logits[0, 3] = 2.0
    logits[0, 7] = 1.0
    yes_ids = {3}
    no_ids = {7}
    s = logits_to_yes_no_probs(logits, yes_ids, no_ids)
    assert abs(s.prob_yes + s.prob_no - 1.0) < 1e-5
    assert s.prob_yes > s.prob_no
    assert s.margin_logit == s.yes_logit - s.no_logit


def test_pick_threshold_favors_middle_class_balance():
    rng = np.random.default_rng(0)
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_score = rng.uniform(0.0, 1.0, size=len(y_true))
    pick = pick_threshold(y_true, y_score, mode="f1")
    assert 0.01 <= pick.threshold <= 0.99
