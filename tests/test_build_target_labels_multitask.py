"""Tests for multitask ICD-9 helpers in ``build_target_labels``."""

from __future__ import annotations

import pandas as pd

from src.data.build_target_labels import (
    add_multitask_labels,
    codes_contain_prefix,
    normalize_icd9_code,
)


def test_normalize_icd9_code() -> None:
    assert normalize_icd9_code(" 278.0 ") == "2780"
    assert normalize_icd9_code("250.01") == "25001"
    assert normalize_icd9_code("") == ""
    assert normalize_icd9_code(float("nan")) == ""


def test_codes_contain_prefix_diabetes() -> None:
    assert codes_contain_prefix("250;401", ["250"]) is True
    assert codes_contain_prefix("150;200", ["250"]) is False


def test_codes_contain_prefix_hypertension() -> None:
    assert codes_contain_prefix("40301;486", ["401", "402", "403", "404", "405"]) is True


def test_add_multitask_labels_lipid_alias() -> None:
    df = pd.DataFrame(
        {
            "label_lipid_disorder": [0, 1],
            "diagnoses_codes_current": ["250;401", ""],
            "diagnoses_codes_next": ["410", "584"],
        }
    )
    out = add_multitask_labels(df)
    assert list(out["label_lipid_next"]) == [0, 1]
    assert (out["label_lipid_next"] == out["label_lipid_disorder"]).all()
    assert out["label_diabetes_current"].tolist() == [1, 0]
    assert out["label_hypertension_current"].tolist() == [1, 0]
    assert out["label_cardio_next"].tolist() == [1, 0]
    assert out["label_kidney_next"].tolist() == [0, 1]
