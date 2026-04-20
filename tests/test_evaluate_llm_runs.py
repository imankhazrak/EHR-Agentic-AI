"""Tests for ``src.evaluation.evaluate_llm_runs``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate_llm_runs import evaluate_llm_results


def test_evaluate_llm_results_with_probabilities() -> None:
    test_df = pd.DataFrame(
        {
            "pair_id": [1, 2, 3, 4],
            "label_lipid_disorder": [0, 1, 0, 1],
        }
    )
    pred_df = pd.DataFrame(
        {
            "pair_id": [1, 2, 3, 4],
            "parsed_prediction": ["No", "Yes", "No", "Yes"],
            "pred_binary": [0, 1, 0, 1],
            "parsed_probability": [0.20, 0.80, 0.15, 0.85],
            "parse_valid_probability": [True, True, True, True],
            "probability_parse_status": ["ok", "ok", "ok", "ok"],
            "raw_response": ["x"] * 4,
            "reasoning": [""] * 4,
            "parser_status": ["structured"] * 4,
            "model_name": ["test"] * 4,
            "prompt_mode": ["zero_shot"] * 4,
        }
    )
    with tempfile.TemporaryDirectory() as td:
        metrics = evaluate_llm_results(test_df, pred_df, "zero_shot", output_dir=td)
        assert metrics["n_total"] == 4
        assert metrics["n_valid_probability"] == 4
        assert metrics["auc"] is not None
        assert metrics["auprc"] is not None
        assert Path(td, "llm_zero_shot_roc_curve.json").exists()


def test_evaluate_llm_results_missing_probability_column() -> None:
    test_df = pd.DataFrame({"pair_id": [1, 2], "label_lipid_disorder": [1, 0]})
    pred_df = pd.DataFrame(
        {
            "pair_id": [1, 2],
            "parsed_prediction": ["Yes", "No"],
            "raw_response": ["Yes", "No"],
            "reasoning": ["", ""],
            "parser_status": ["first_line", "first_line"],
            "model_name": ["m", "m"],
            "prompt_mode": ["zero_shot", "zero_shot"],
        }
    )
    with tempfile.TemporaryDirectory() as td:
        m = evaluate_llm_results(test_df, pred_df, "zero_shot", output_dir=td)
        assert m["n_total"] == 2
        assert m["n_valid_probability"] == 0
        assert m["auc"] is None
