"""Smoke tests for constrained Yes/No decoding in finetuned test eval."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from src.llm.output_parser import parse_prediction
from src.scripts.evaluate_finetuned_gemma_on_test import (
    _FirstStepRestrictTokensLogitsProcessor,
    _collect_token_ids,
    _prompt,
)

_REPO = Path(__file__).resolve().parents[1]


class _StubTokenizer:
    """Minimal tokenizer.encode(first_token_only) for _collect_token_ids."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        key = text.strip().lower()
        mapping = {"yes": [101], "no": [102]}
        return mapping.get(key, [0])


def test_first_step_processor_masks_non_allowed_vocab() -> None:
    proc = _FirstStepRestrictTokensLogitsProcessor(prompt_length=4, allowed_token_ids=[2, 7])
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    scores = torch.arange(10, dtype=torch.float32).unsqueeze(0).expand(1, -1).clone()

    out = proc(input_ids, scores)
    assert out.shape == scores.shape
    for j in range(10):
        if j in (2, 7):
            assert out[0, j].item() == scores[0, j].item()
        else:
            assert out[0, j].item() == float("-inf")


def test_first_step_processor_leaves_scores_unchanged_after_first_token() -> None:
    proc = _FirstStepRestrictTokensLogitsProcessor(prompt_length=3, allowed_token_ids=[1])
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    scores = torch.randn(1, 50)
    out = proc(input_ids, scores)
    assert torch.allclose(out, scores)


def test_prompt_ends_with_prediction_newline() -> None:
    p = _prompt("short narrative")
    assert p.endswith("Prediction:\n")
    assert "Patient record:" in p


def test_parse_prediction_accepts_plain_yes_no() -> None:
    assert parse_prediction("Yes")["prediction"] == "Yes"
    assert parse_prediction("No")["parser_status"] == "first_line"
    assert parse_prediction("yes")["prediction"] == "Yes"


def test_collect_token_ids_stub_disjoint_yes_no() -> None:
    st = _StubTokenizer()
    yes_ids = _collect_token_ids(st, ["Yes", " yes"])
    no_ids = _collect_token_ids(st, ["No", " no"])
    assert yes_ids == {101}
    assert no_ids == {102}
    assert yes_ids.isdisjoint(no_ids)


def test_gemma_tokenizer_yes_no_ids_when_snapshot_loads() -> None:
    """Loads real Gemma tokenizer when snapshot + transformers/sentencepiece work."""
    snap = _REPO / "models" / "hf_snapshots" / "google--gemma-4-e4b-it"
    if not snap.is_dir():
        pytest.skip("Gemma snapshot not present")
    try:
        try:
            tok = AutoTokenizer.from_pretrained(str(snap))
        except (AttributeError, TypeError, ValueError, OSError):
            tok = AutoTokenizer.from_pretrained(str(snap), use_fast=False)
    except Exception as exc:
        pytest.skip(f"Gemma tokenizer unavailable in this environment: {exc}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes_ids = _collect_token_ids(tok, ["Yes", " yes", "Yes ", " yes "])
    no_ids = _collect_token_ids(tok, ["No", " no", "No ", " no "])
    assert yes_ids and no_ids and yes_ids.isdisjoint(no_ids)
