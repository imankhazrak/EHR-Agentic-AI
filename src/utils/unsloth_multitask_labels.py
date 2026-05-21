"""Response-only label masks for Unsloth multitask Alpaca SFT (JSON after ``### Response:``).

TRL 0.24 ``SFTTrainer`` with a single ``text`` column uses language-modeling preprocessing:
all tokens are supervised (``completion_only_loss`` defaults to ``False``). To train only on
the strict multitask JSON (and optional EOS), we pre-tokenize rows and attach a
``completion_mask`` (1 = compute loss, 0 = ``-100`` in the collator) with supervision starting
at the first ``{`` of the assistant JSON — not at the ``### Response:`` header line.

Used by ``scripts/train_unsloth_router.py`` and its ``--debug-label-mask-check`` audit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.llm.output_parser import MULTITASK_JSON_TASK_KEYS, parse_multitask_output
from src.utils.alpaca_multitask_fit import resolve_tokenizer_for_encode

RESPONSE_MARKER = "### Response:"


def find_json_supervision_char_offset(formatted_text: str) -> int:
    """Return character index of the first ``{`` in the assistant JSON block, or ``-1``."""
    if RESPONSE_MARKER not in formatted_text:
        return -1
    tail = formatted_text.split(RESPONSE_MARKER, 1)[1]
    rel = tail.find("{")
    if rel < 0:
        return -1
    return formatted_text.index(RESPONSE_MARKER) + len(RESPONSE_MARKER) + rel


def build_completion_mask_for_input_ids(
    hf_tokenizer: Any,
    formatted_text: str,
    input_ids: List[int],
) -> Tuple[List[int], int, bool]:
    """Build ``completion_mask`` aligned with ``input_ids`` (supervision from first JSON ``{``).

    Returns ``(completion_mask, supervise_start_index, prefix_aligned)``.
    """
    json_char = find_json_supervision_char_offset(formatted_text)
    if json_char < 0:
        return [0] * len(input_ids), len(input_ids), False

    prefix_text = formatted_text[:json_char]
    prefix_ids = hf_tokenizer.encode(prefix_text, add_special_tokens=True)
    supervise_start = len(prefix_ids)
    prefix_aligned = input_ids[:supervise_start] == prefix_ids

    if not prefix_aligned:
        # Tokenizer boundary mismatch: scan for first token at/after JSON "{" in full decode.
        supervise_start = _fallback_supervise_start(hf_tokenizer, formatted_text, input_ids)

    n = len(input_ids)
    supervise_start = max(0, min(supervise_start, n))
    completion_mask = [0] * supervise_start + [1] * (n - supervise_start)
    return completion_mask, supervise_start, prefix_aligned


def _fallback_supervise_start(
    hf_tokenizer: Any,
    formatted_text: str,
    input_ids: List[int],
) -> int:
    """Find a reasonable supervision start when prefix encode does not match full encode."""
    json_char = find_json_supervision_char_offset(formatted_text)
    if json_char < 0:
        return len(input_ids)

    best = len(input_ids)
    for start in range(len(input_ids) + 1):
        chunk = hf_tokenizer.decode(input_ids[:start], skip_special_tokens=False)
        if len(chunk) >= json_char and chunk[:json_char] == formatted_text[:json_char]:
            best = start
            break
    return best


def labels_from_input_ids_and_mask(
    input_ids: List[int],
    completion_mask: List[int],
) -> List[int]:
    """``labels`` with ``-100`` on prompt/context tokens (mask 0)."""
    if len(completion_mask) != len(input_ids):
        raise ValueError(
            f"completion_mask length {len(completion_mask)} != input_ids length {len(input_ids)}"
        )
    return [tid if m == 1 else -100 for tid, m in zip(input_ids, completion_mask)]


def decode_masked_regions(
    hf_tokenizer: Any,
    input_ids: List[int],
    labels: List[int],
) -> Tuple[str, str]:
    """Decode prompt/context (label ``-100``) and supervised (label not ``-100``) token spans."""
    prompt_ids = [tid for tid, lab in zip(input_ids, labels) if lab == -100]
    supervised_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    prompt_text = hf_tokenizer.decode(prompt_ids, skip_special_tokens=True)
    supervised_text = hf_tokenizer.decode(supervised_ids, skip_special_tokens=True)
    return prompt_text, supervised_text


def audit_label_mask_row(
    *,
    row_index: int,
    pair_id: Any,
    formatted_text: str,
    input_ids: List[int],
    completion_mask: List[int],
    hf_tokenizer: Any,
    output_text: str,
) -> Dict[str, Any]:
    """Collect per-row label-mask audit fields for CLI / markdown reports."""
    labels = labels_from_input_ids_and_mask(input_ids, completion_mask)
    n_total = len(input_ids)
    n_supervised = int(sum(completion_mask))
    n_masked = n_total - n_supervised
    pct = (100.0 * n_supervised / n_total) if n_total else 0.0

    json_char = find_json_supervision_char_offset(formatted_text)
    resp_char = formatted_text.find(RESPONSE_MARKER)
    _, supervise_start, prefix_aligned = build_completion_mask_for_input_ids(
        hf_tokenizer, formatted_text, input_ids
    )

    prompt_decoded, supervised_decoded = decode_masked_regions(hf_tokenizer, input_ids, labels)

    has_response_marker = RESPONSE_MARKER in prompt_decoded or RESPONSE_MARKER in formatted_text
    supervised_starts_with_brace = supervised_decoded.lstrip().startswith("{")
    keys_present = all(k in supervised_decoded for k in MULTITASK_JSON_TASK_KEYS)
    parse_ok = parse_multitask_output(supervised_decoded) is not None

    return {
        "row_index": row_index,
        "pair_id": pair_id,
        "tokenized_length": n_total,
        "response_boundary_char_pos": resp_char,
        "json_supervision_char_pos": json_char,
        "supervise_start_token_index": supervise_start,
        "prefix_token_alignment_ok": prefix_aligned,
        "n_total_tokens": n_total,
        "n_masked_tokens": n_masked,
        "n_supervised_tokens": n_supervised,
        "pct_supervised": round(pct, 2),
        "has_response_marker": has_response_marker,
        "supervised_starts_with_brace": supervised_starts_with_brace,
        "supervised_has_all_task_keys": keys_present,
        "supervised_parse_ok": parse_ok,
        "prompt_context_decoded": prompt_decoded,
        "supervised_decoded": supervised_decoded,
        "failed_checks": _collect_mask_failures(
            has_response_marker=has_response_marker,
            supervised_starts_with_brace=supervised_starts_with_brace,
            keys_present=keys_present,
            parse_ok=parse_ok,
            output_text=output_text,
            supervised_decoded=supervised_decoded,
        ),
    }


def _collect_mask_failures(
    *,
    has_response_marker: bool,
    supervised_starts_with_brace: bool,
    keys_present: bool,
    parse_ok: bool,
    output_text: str,
    supervised_decoded: str,
) -> List[str]:
    failed: List[str] = []
    if not has_response_marker:
        failed.append("missing_response_marker_in_prompt")
    if not supervised_starts_with_brace:
        failed.append("supervised_not_starting_with_brace")
    if not keys_present:
        failed.append("supervised_missing_task_keys")
    if not parse_ok:
        failed.append("supervised_strict_parse_failed")
    if output_text.strip() not in supervised_decoded and output_text.strip() not in supervised_decoded.replace(
        " ", ""
    ):
        # Loose check: gold JSON should appear in supervised span (whitespace may differ).
        if output_text.strip()[:80] not in supervised_decoded:
            failed.append("gold_output_not_in_supervised_span")
    return failed
