"""Logit-based Yes/No scoring at the first post-prompt position (no free-text probabilities).

Used by finetuned test evaluation and fusion train/test logit scoring pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

from src.llm.output_parser import parse_prediction


def build_eval_prompt(narrative: str) -> str:
    """Prompt through ``Prediction:\\n`` (matches training / finetuned eval)."""
    return (
        "Task: Predict whether Disorders of Lipid Metabolism will be present in the next visit.\n\n"
        f"Patient record:\n{narrative}\n\n"
        "Prediction:\n"
    )


def collect_first_token_ids(tokenizer: AutoTokenizer, texts: List[str]) -> Set[int]:
    ids: Set[int] = set()
    for t in texts:
        enc = tokenizer.encode(t, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return ids


def resolve_yes_no_token_ids(tokenizer: AutoTokenizer) -> Tuple[Set[int], Set[int], List[int]]:
    yes_ids = collect_first_token_ids(tokenizer, ["Yes", " yes", "Yes ", " yes "])
    no_ids = collect_first_token_ids(tokenizer, ["No", " no", "No ", " no "])
    if not yes_ids or not no_ids:
        raise RuntimeError("Failed to resolve Yes/No token ids for logit scoring.")
    yes_no_ids = sorted(yes_ids | no_ids)
    return yes_ids, no_ids, yes_no_ids


class FirstStepRestrictTokensLogitsProcessor(LogitsProcessor):
    """Restrict the first generated token to *allowed_token_ids* (then logits unchanged)."""

    def __init__(self, prompt_length: int, allowed_token_ids: List[int]) -> None:
        self.prompt_length = int(prompt_length)
        self.allowed_token_ids = list(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] != self.prompt_length:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask


@dataclass
class LogitPairScore:
    prob_yes: float
    prob_no: float
    yes_logit: float
    no_logit: float

    @property
    def margin_logit(self) -> float:
        return self.yes_logit - self.no_logit


def logits_to_yes_no_probs(
    logits_last: torch.Tensor,
    yes_ids: Set[int],
    no_ids: Set[int],
) -> LogitPairScore:
    """Two-way softmax over max Yes-token logit vs max No-token logit (vocab slice)."""
    if logits_last.dim() != 2 or logits_last.shape[0] != 1:
        raise ValueError(f"Expected logits shape (1, vocab), got {tuple(logits_last.shape)}")
    row = logits_last[0]
    yes_logit = max(float(row[idx].item()) for idx in yes_ids)
    no_logit = max(float(row[idx].item()) for idx in no_ids)
    z = torch.tensor([yes_logit, no_logit], device=logits_last.device, dtype=logits_last.dtype)
    p = torch.softmax(z, dim=0)
    return LogitPairScore(
        prob_yes=float(p[0].item()),
        prob_no=float(p[1].item()),
        yes_logit=yes_logit,
        no_logit=no_logit,
    )


def forward_logit_score(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    narrative: str,
    yes_ids: Set[int],
    no_ids: Set[int],
) -> Tuple[LogitPairScore, Dict[str, torch.Tensor], int]:
    """Single forward pass; returns logit pair score, token batch, prompt length."""
    ptxt = build_eval_prompt(str(narrative))
    toks = tokenizer(ptxt, return_tensors="pt")
    dev = next(model.parameters()).device
    toks = {k: v.to(dev) for k, v in toks.items()}
    prompt_len = int(toks["input_ids"].shape[1])
    with torch.no_grad():
        out = model(**toks)
        logits = out.logits[:, -1, :]
    score = logits_to_yes_no_probs(logits, yes_ids, no_ids)
    return score, toks, prompt_len


def hard_label_from_generation(
    tokenizer: AutoTokenizer,
    yes_ids: Set[int],
    no_ids: Set[int],
    new_tokens: torch.Tensor,
    generated: str,
    prob_yes: float,
    prob_no: float,
) -> Tuple[int, str, str]:
    """Match finetuned eval: first-token Yes/No, else parser, else logit fallback."""
    parsed = parse_prediction(generated.strip())
    first_id = int(new_tokens[0].item()) if new_tokens.numel() > 0 else None
    if first_id is not None and first_id in yes_ids:
        return 1, "Yes", "first_token_yes"
    if first_id is not None and first_id in no_ids:
        return 0, "No", "first_token_no"
    if parsed["prediction"] == "Yes":
        return 1, "Yes", str(parsed["parser_status"])
    if parsed["prediction"] == "No":
        return 0, "No", str(parsed["parser_status"])
    pred = 1 if prob_yes >= prob_no else 0
    return pred, ("Yes" if pred == 1 else "No"), "logits_fallback"


def generate_one_restricted_token(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    toks: Dict[str, torch.Tensor],
    prompt_len: int,
    yes_no_ids: List[int],
) -> torch.Tensor:
    proc = LogitsProcessorList([FirstStepRestrictTokensLogitsProcessor(prompt_len, yes_no_ids)])
    with torch.no_grad():
        gen = model.generate(
            **toks,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=proc,
        )
    return gen[0, prompt_len:]
