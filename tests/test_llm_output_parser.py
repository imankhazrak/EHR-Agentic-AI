"""Tests for ``src.llm.output_parser.parse_prediction``."""

from __future__ import annotations

import pytest

from src.llm.output_parser import (
    multitask_flat_column_names,
    parse_multitask_output,
    parse_prediction,
    test_parse_multitask_output_example,
)


def test_structured_triple() -> None:
    text = "Prediction: Yes\nProbability: 0.75\nReasoning: Lipid diagnosis documented."
    out = parse_prediction(text)
    assert out["prediction"] == "Yes"
    assert out["parser_status"] == "structured"
    assert out["probability"] == pytest.approx(0.75)
    assert out["probability_status"] == "ok"
    assert out["parse_valid_probability"] is True
    assert "Lipid" in out["reasoning"]


def test_structured_spacing() -> None:
    text = "Prediction:  No \nProbability:  0.10 \nReasoning:  No evidence "
    out = parse_prediction(text)
    assert out["prediction"] == "No"
    assert out["probability"] == pytest.approx(0.1)


def test_invalid_probability() -> None:
    text = "Prediction: Yes\nProbability: 2.0\nReasoning: x"
    out = parse_prediction(text)
    assert out["prediction"] == "Yes"
    assert out["probability"] is None
    assert out["probability_status"] == "out_of_range"
    assert out["parse_valid_probability"] is False


def test_legacy_plain_yes() -> None:
    out = parse_prediction("Yes")
    assert out["prediction"] == "Yes"
    assert out["parser_status"] == "first_line"
    assert out["probability_status"] == "missing"


def test_legacy_lowercase_yes() -> None:
    out = parse_prediction("yes")
    assert out["prediction"] == "Yes"


def test_probability_format_warning_single_decimal() -> None:
    text = "Prediction: No\nProbability: 0.5\nReasoning: x"
    out = parse_prediction(text)
    assert out["parse_valid_probability"] is True
    assert out["probability_format_warning"] is True


def test_two_decimal_probability_no_warning() -> None:
    text = "Prediction: No\nProbability: 0.50\nReasoning: x"
    out = parse_prediction(text)
    assert out["parse_valid_probability"] is True
    assert out["probability_format_warning"] is False


def test_parse_multitask_output_example_runs() -> None:
    test_parse_multitask_output_example()


def _valid_multitask_json() -> str:
    return (
        '{"lipid_next": {"prediction": "Yes", "probability": 0.72}, '
        '"diabetes_current": {"prediction": "No", "probability": 0.10}, '
        '"hypertension_current": {"prediction": "Yes", "probability": 0.80}, '
        '"obesity_current": {"prediction": "No", "probability": 0.20}, '
        '"cardio_next": {"prediction": "No", "probability": 0.30}, '
        '"kidney_next": {"prediction": "No", "probability": 0.15}, '
        '"stroke_next": {"prediction": "No", "probability": 0.10}, '
        '"reasoning": "ok"}'
    )


def test_parse_multitask_output_accepts_fenced_json() -> None:
    inner = _valid_multitask_json()
    text = "```json\n" + inner + "\n```"
    p = parse_multitask_output(text)
    assert p is not None
    assert p["lipid_prob"] == pytest.approx(0.72)
    assert set(multitask_flat_column_names()).issubset(p.keys())


def test_parse_multitask_output_rejects_trailing_text() -> None:
    assert parse_multitask_output(_valid_multitask_json() + "\nThanks") is None


def test_parse_multitask_output_rejects_missing_task() -> None:
    bad = (
        '{"lipid_next": {"prediction": "Yes", "probability": 0.5}, '
        '"reasoning": "x"}'
    )
    assert parse_multitask_output(bad) is None


def test_parse_multitask_output_rejects_bad_prediction_token() -> None:
    bad = _valid_multitask_json().replace('"Yes"', '"Maybe"', 1)
    assert parse_multitask_output(bad) is None
