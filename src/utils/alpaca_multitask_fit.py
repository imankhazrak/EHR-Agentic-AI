"""Fit Alpaca multitask strings into a token budget without dropping the response slot.

TRL ``SFTTrainer`` and naive ``tokenizer(..., truncation=True, max_length=...)`` apply
**right** truncation on the full string. For long visit narratives that removes the
trailing ``\\n\\n### Response:\\n`` (and, for SFT, the gold JSON), which breaks both
training and eval.

Callers pass everything that must appear **after** the variable visit ``input`` text
as ``tail_after_input``:

- **Inference:** ``tail_after_input = "\\n\\n### Response:\\n"``
- **SFT:** ``tail_after_input = "\\n\\n### Response:\\n" + gold_output + eos_token``
"""

from __future__ import annotations

from typing import Any, List, Tuple


def resolve_tokenizer_for_encode(processing_class: Any) -> Any:
    """Return a Hugging Face tokenizer with ``.encode`` / ``__call__`` for text.

    Unsloth Gemma 4 checkpoints often return a **processor** (e.g. ``Gemma4Processor``)
    from ``FastLanguageModel.from_pretrained``; it exposes ``decode`` but not ``encode``.
    TRL and this helper need the inner ``PreTrainedTokenizer`` (usually ``.tokenizer``).
    """
    pc = processing_class
    if callable(getattr(pc, "encode", None)):
        return pc
    inner = getattr(pc, "tokenizer", None)
    if inner is not None and callable(getattr(inner, "encode", None)):
        return inner
    raise TypeError(
        "alpaca_multitask_fit.resolve_tokenizer_for_encode: expected a tokenizer with "
        f".encode or a processor with .tokenizer; got {type(pc)!r}"
    )


def _alpaca_head(instruction: str) -> str:
    return f"""### Instruction:
{instruction}

### Input:
"""


def fit_visit_input_for_token_cap(
    tokenizer: Any,
    *,
    instruction: str,
    input_text: str,
    tail_after_input: str,
    max_length: int,
) -> Tuple[str, List[int]]:
    """Return ``(truncated_input, input_ids)`` so ``head + truncated_input + tail`` fits ``max_length`` tokens.

    Shortens ``input_text`` by taking a **prefix** (keeps the start of the visit block).
    If the instruction and tail alone still exceed ``max_length``, falls back to HF
    **left** truncation on the assembled string.
    """
    tok = resolve_tokenizer_for_encode(tokenizer)
    head = _alpaca_head(instruction)

    def assembled(inp: str) -> str:
        return head + inp + tail_after_input

    def n_tokens(text: str) -> int:
        return int(len(tok.encode(text, add_special_tokens=True)))

    full_text = assembled(input_text)
    if n_tokens(full_text) <= max_length:
        enc = tok(full_text, return_tensors="pt", add_special_tokens=True)
        return input_text, enc["input_ids"][0].tolist()

    best = ""
    lo, hi = 0, len(input_text)
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = input_text[:mid]
        if n_tokens(assembled(chunk)) <= max_length:
            best = chunk
            lo = mid + 1
        else:
            hi = mid - 1

    final_text = assembled(best)
    enc = tok(final_text, return_tensors="pt", add_special_tokens=True)
    if enc["input_ids"].shape[1] > max_length:
        enc = tok(
            final_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            truncation_side="left",
        )
    return best, enc["input_ids"][0].tolist()
