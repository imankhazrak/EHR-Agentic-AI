#!/usr/bin/env python3
"""Audit Unsloth multitask JSONL → Alpaca SFT text → tokenization before retraining.

Loads sample rows from ``dataset_for_unsloth/*.jsonl``, applies the same formatting and
truncation as ``scripts/train_unsloth_router.py``, tokenizes with the Gemma4
tokenizer/processor, and writes a human-readable report for visual inspection.

Does **not** train or submit jobs.

**Environment:** prefer ``conda activate unsloth_env`` (matches Slurm / training). With
``--tokenizer-only``, Hugging Face ``AutoProcessor`` / ``AutoTokenizer`` may work on CPU,
but the processor should match what Unsloth returns at train time — verify in ``unsloth_env``.

Run from repo root::

    python scripts/audit_unsloth_multitask_tokenization.py \\
      --jsonl dataset_for_unsloth/train.jsonl \\
      --model-name unsloth/gemma-4-e2b-it-unsloth-bnb-4bit \\
      --max-seq-length 2048 \\
      --num-samples 5 \\
      --sample-mode head \\
      --output-md outputs/audits/unsloth_tokenization_audit_train.md \\
      --fail-on-error
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.llm.output_parser import MULTITASK_JSON_TASK_KEYS, parse_multitask_output
from src.utils.alpaca_multitask_fit import fit_visit_input_for_token_cap, resolve_tokenizer_for_encode

# Must stay identical to scripts/train_unsloth_router.py::build_multitask_sft_text
# (importing that module always loads unsloth; this copy keeps --tokenizer-only lightweight).
def build_multitask_sft_text(
    instruction: str,
    input_text: str,
    output_text: str,
    eos_token: str,
) -> str:
    body = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""
    if eos_token:
        return body + eos_token
    return body


RESPONSE_MARKER = "### Response:"
FLAT_DISPLAY_KEYS = [
    "lipid_pred",
    "lipid_prob",
    "diabetes_pred",
    "diabetes_prob",
    "hypertension_pred",
    "hypertension_prob",
    "obesity_pred",
    "obesity_prob",
    "cardio_pred",
    "cardio_prob",
    "kidney_pred",
    "kidney_prob",
    "stroke_pred",
    "stroke_prob",
]


@dataclass
class SampleAudit:
    row_index: int
    pair_id: Any
    instruction_len: int
    input_len: int
    output_len: int
    input_preview: str
    output_preview: str
    strict_parse_ok: bool
    parsed_flat: Optional[Dict[str, Any]]
    input_truncated: bool
    orig_input_chars: int
    trunc_input_chars: int
    formatted_chars: int
    tokenized_length: int
    formatted_length_ok: bool
    has_instruction_marker: bool
    has_input_marker: bool
    has_response_marker: bool
    has_json_open_after_response: bool
    has_json_close_before_eos: bool
    gold_output_in_formatted: bool
    decoded_has_response_marker: bool
    decoded_has_output_json: bool
    response_boundary_found: bool
    response_char_pos: int
    response_token_pos: Optional[int]
    boundary_window: str
    decode_roundtrip_parse_ok: bool
    token_window_text: str = ""
    failed_checks: List[str] = field(default_factory=list)


@dataclass
class AuditSummary:
    samples_checked: int = 0
    strict_output_parse_ok_count: int = 0
    formatted_length_ok_count: int = 0
    response_boundary_found_count: int = 0
    output_json_preserved_count: int = 0
    decode_roundtrip_parse_ok_count: int = 0
    any_failed_rows: List[int] = field(default_factory=list)


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _select_indices(
    rows: Sequence[Dict[str, Any]],
    num_samples: int,
    sample_mode: str,
    seed: int,
) -> List[int]:
    n = len(rows)
    k = max(0, min(num_samples, n))
    if k == 0:
        return []
    if sample_mode == "head":
        return list(range(k))
    if sample_mode == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(n), k))
    if sample_mode == "longest":
        scored = [
            (i, len(str(rows[i].get("input") or "")))
            for i in range(n)
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [i for i, _ in scored[:k]]
    raise ValueError(f"Unknown sample_mode: {sample_mode!r}")


def _truncate_preview(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n... [truncated for display] ..."


def _pretty_json_output(raw: str, max_chars: int) -> str:
    try:
        obj = json.loads(raw)
        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        return _truncate_preview(pretty, max_chars)
    except json.JSONDecodeError:
        return _truncate_preview(raw, max_chars)


def _format_parsed_flat(parsed: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in FLAT_DISPLAY_KEYS:
        if key in parsed:
            val = parsed[key]
            if key.endswith("_pred"):
                out[key] = "Yes" if val == 1 else ("No" if val == 0 else val)
            else:
                out[key] = val
    if "reasoning" in parsed:
        rs = parsed["reasoning"]
        out["reasoning"] = _truncate_preview(str(rs), 200)
    return out


def _extract_json_after_response(text: str, eos_token: str = "") -> str:
    if RESPONSE_MARKER not in text:
        return ""
    tail = text.split(RESPONSE_MARKER, 1)[-1]
    # Training puts JSON immediately after the marker line (optional leading newline).
    tail = tail.lstrip("\n").strip()
    if eos_token and tail.endswith(eos_token):
        tail = tail[: -len(eos_token)].strip()
    return tail


def _strip_trailing_eos(text: str, eos_token: str) -> str:
    if not eos_token:
        return text.rstrip()
    t = text.rstrip()
    if t.endswith(eos_token):
        return t[: -len(eos_token)].rstrip()
    return t


def _has_json_close_before_eos(formatted: str, output_text: str, eos_token: str) -> bool:
    body = _strip_trailing_eos(formatted, eos_token)
    json_part = _extract_json_after_response(body, eos_token=eos_token)
    if not json_part:
        return False
    stripped = json_part.strip()
    if not stripped.endswith("}"):
        return False
    # Gold output should be the JSON object (possibly with reasoning inside).
    return output_text.strip() in stripped or stripped.startswith("{")


def _boundary_window(formatted: str, before: int = 200, after: int = 500) -> Tuple[int, str]:
    pos = formatted.find(RESPONSE_MARKER)
    if pos < 0:
        return -1, ""
    start = max(0, pos - before)
    end = min(len(formatted), pos + len(RESPONSE_MARKER) + after)
    return pos, formatted[start:end]


def _find_response_token_pos(
    hf_tok: Any,
    formatted: str,
    input_ids: List[int],
) -> Optional[int]:
    pos = formatted.find(RESPONSE_MARKER)
    if pos < 0:
        return None
    prefix = formatted[: pos + len(RESPONSE_MARKER)]
    prefix_ids = hf_tok.encode(prefix, add_special_tokens=True)
    # First token index where decoded prefix length matches (approximate boundary).
    if len(prefix_ids) <= len(input_ids):
        return len(prefix_ids)
    return None


def _token_window_around_pos(
    hf_tok: Any,
    input_ids: List[int],
    center: Optional[int],
    window: int = 12,
    show_token_ids: bool = False,
) -> str:
    if center is None or center < 0:
        return ""
    lo = max(0, center - window)
    hi = min(len(input_ids), center + window)
    slice_ids = input_ids[lo:hi]
    decoded = hf_tok.decode(slice_ids, skip_special_tokens=False)
    if show_token_ids:
        id_str = ", ".join(str(t) for t in slice_ids)
        return f"token_ids[{lo}:{hi}]: [{id_str}]\ndecoded: {decoded!r}"
    return decoded


def _tokenizer_identity(processing_class: Any) -> Dict[str, Any]:
    inner = getattr(processing_class, "tokenizer", None)
    hf = resolve_tokenizer_for_encode(processing_class)
    info: Dict[str, Any] = {
        "processing_class_type": type(processing_class).__name__,
        "processing_class_module": type(processing_class).__module__,
        "has_encode_on_processing_class": callable(getattr(processing_class, "encode", None)),
        "has_tokenizer_attr": inner is not None,
        "inner_tokenizer_type": type(inner).__name__ if inner is not None else None,
        "encode_tokenizer_type": type(hf).__name__,
        "encode_tokenizer_class": hf.__class__.__name__,
        "eos_token": getattr(hf, "eos_token", None) or getattr(processing_class, "eos_token", None),
        "eos_token_id": getattr(hf, "eos_token_id", None) or getattr(processing_class, "eos_token_id", None),
        "pad_token": getattr(hf, "pad_token", None) or getattr(processing_class, "pad_token", None),
        "pad_token_id": getattr(hf, "pad_token_id", None) or getattr(processing_class, "pad_token_id", None),
        "model_max_length": getattr(hf, "model_max_length", None),
    }
    return info


def _load_processing_class(
    model_name: str,
    max_seq_length: int,
    tokenizer_only: bool,
) -> Tuple[Any, str]:
    if tokenizer_only:
        from transformers import AutoProcessor, AutoTokenizer

        try:
            proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            return proc, "transformers.AutoProcessor.from_pretrained (tokenizer-only)"
        except Exception as proc_exc:
            try:
                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                return tok, f"transformers.AutoTokenizer.from_pretrained (tokenizer-only; processor failed: {proc_exc})"
            except Exception as tok_exc:
                raise SystemExit(
                    "tokenizer-only load failed. Use conda env unsloth_env and omit --tokenizer-only:\n"
                    f"  processor error: {proc_exc}\n"
                    f"  tokenizer error: {tok_exc}"
                ) from tok_exc

    from unsloth import FastLanguageModel

    _model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    del _model
    return tokenizer, "unsloth.FastLanguageModel.from_pretrained"


def _format_training_text(
    processing_class: Any,
    instruction: str,
    input_text: str,
    output_text: str,
    max_seq_length: int,
    eos_token: str,
) -> Tuple[str, str, bool, List[int]]:
    tail = f"\n\n### Response:\n{output_text}{eos_token}"
    trunc_input, input_ids = fit_visit_input_for_token_cap(
        processing_class,
        instruction=instruction,
        input_text=input_text,
        tail_after_input=tail,
        max_length=max_seq_length,
    )
    formatted = build_multitask_sft_text(
        instruction,
        trunc_input,
        output_text,
        eos_token,
    )
    return formatted, trunc_input, trunc_input != input_text, input_ids


def _audit_one_sample(
    row_index: int,
    row: Dict[str, Any],
    processing_class: Any,
    max_seq_length: int,
    eos_token: str,
    max_input_chars_print: int,
    max_output_chars_print: int,
    show_token_ids: bool,
) -> SampleAudit:
    instruction = str(row.get("instruction") or "")
    input_text = str(row.get("input") or "")
    output_text = str(row.get("output") or "")
    pair_id = row.get("pair_id")

    parsed = parse_multitask_output(output_text)
    strict_ok = parsed is not None

    formatted, trunc_input, truncated, input_ids = _format_training_text(
        processing_class,
        instruction,
        input_text,
        output_text,
        max_seq_length,
        eos_token,
    )

    hf_tok = resolve_tokenizer_for_encode(processing_class)
    tokenized_length = len(input_ids)
    formatted_length_ok = tokenized_length <= max_seq_length

    resp_pos, boundary_window = _boundary_window(formatted)
    response_boundary_found = resp_pos >= 0
    resp_token_pos = _find_response_token_pos(hf_tok, formatted, input_ids)

    # Match eval decode (test_unsloth_router.py): skip special tokens, then strict-parse JSON only.
    decoded = hf_tok.decode(input_ids, skip_special_tokens=True)
    json_after = _extract_json_after_response(decoded, eos_token=eos_token)
    roundtrip_parsed = parse_multitask_output(json_after) if json_after else None

    failed: List[str] = []
    if not strict_ok:
        failed.append("strict_output_parse")
    if not formatted_length_ok:
        failed.append("formatted_length")
    if not response_boundary_found:
        failed.append("response_boundary")
    if output_text not in formatted:
        failed.append("gold_output_in_formatted")
    if roundtrip_parsed is None:
        failed.append("decode_roundtrip_parse")

    return SampleAudit(
        row_index=row_index,
        pair_id=pair_id,
        instruction_len=len(instruction),
        input_len=len(input_text),
        output_len=len(output_text),
        input_preview=_truncate_preview(input_text, max_input_chars_print),
        output_preview=_pretty_json_output(output_text, max_output_chars_print),
        strict_parse_ok=strict_ok,
        parsed_flat=_format_parsed_flat(parsed) if parsed else None,
        input_truncated=truncated,
        orig_input_chars=len(input_text),
        trunc_input_chars=len(trunc_input),
        formatted_chars=len(formatted),
        tokenized_length=tokenized_length,
        formatted_length_ok=formatted_length_ok,
        has_instruction_marker="### Instruction:" in formatted,
        has_input_marker="### Input:" in formatted,
        has_response_marker=RESPONSE_MARKER in formatted,
        has_json_open_after_response=(
            "{" in _extract_json_after_response(formatted, eos_token=eos_token)
        ),
        has_json_close_before_eos=_has_json_close_before_eos(
            formatted, output_text, eos_token
        ),
        gold_output_in_formatted=output_text in formatted,
        decoded_has_response_marker=RESPONSE_MARKER in decoded,
        decoded_has_output_json=output_text in decoded or (
            output_text.strip()
            in _extract_json_after_response(decoded, eos_token=eos_token)
        ),
        response_boundary_found=response_boundary_found,
        response_char_pos=resp_pos,
        response_token_pos=resp_token_pos,
        boundary_window=boundary_window,
        decode_roundtrip_parse_ok=roundtrip_parsed is not None,
        token_window_text=_token_window_around_pos(
            hf_tok, input_ids, resp_token_pos, show_token_ids=show_token_ids
        ),
        failed_checks=failed,
    )


def _update_summary(summary: AuditSummary, audit: SampleAudit) -> None:
    summary.samples_checked += 1
    if audit.strict_parse_ok:
        summary.strict_output_parse_ok_count += 1
    if audit.formatted_length_ok:
        summary.formatted_length_ok_count += 1
    if audit.response_boundary_found:
        summary.response_boundary_found_count += 1
    if audit.gold_output_in_formatted:
        summary.output_json_preserved_count += 1
    if audit.decode_roundtrip_parse_ok:
        summary.decode_roundtrip_parse_ok_count += 1
    if audit.failed_checks:
        summary.any_failed_rows.append(audit.row_index)


def _print_tokenizer_summary(tok_info: Dict[str, Any], load_source: str) -> None:
    print("=" * 72)
    print("TOKENIZER / PROCESSOR SUMMARY")
    print("=" * 72)
    print(f"load_source: {load_source}")
    for key, val in tok_info.items():
        print(f"  {key}: {val!r}")
    print()


def _print_sample(audit: SampleAudit, show_token_ids: bool) -> None:
    print("=" * 72)
    print(f"SAMPLE row_index={audit.row_index}  pair_id={audit.pair_id!r}")
    print("=" * 72)
    print(f"instruction_len_chars: {audit.instruction_len}")
    print(f"input_len_chars: {audit.input_len}")
    print(f"output_len_chars: {audit.output_len}")
    print()
    print("--- Raw input preview ---")
    print(audit.input_preview)
    print()
    print("--- Raw output JSON ---")
    print(audit.output_preview)
    print()
    print(f"strict_parse_ok: {audit.strict_parse_ok}")
    if audit.parsed_flat:
        print("parsed_flat:")
        for k, v in audit.parsed_flat.items():
            print(f"  {k}: {v!r}")
    else:
        print("parsed_flat: (strict parser returned None)")
    print()
    print("--- Formatted SFT / tokenization ---")
    print(f"input_truncated: {audit.input_truncated}")
    print(f"orig_input_chars: {audit.orig_input_chars}")
    print(f"trunc_input_chars: {audit.trunc_input_chars}")
    print(f"formatted_chars: {audit.formatted_chars}")
    print(f"tokenized_length: {audit.tokenized_length}")
    print(f"formatted_length_ok (<= cap): {audit.formatted_length_ok}")
    print(f"has ### Instruction:: {audit.has_instruction_marker}")
    print(f"has ### Input:: {audit.has_input_marker}")
    print(f"has ### Response:: {audit.has_response_marker}")
    print(f"has '{{' after ### Response:: {audit.has_json_open_after_response}")
    print(f"has final '}}' before EOS: {audit.has_json_close_before_eos}")
    print(f"gold_output_in_formatted: {audit.gold_output_in_formatted}")
    print(f"decoded_has_response_marker: {audit.decoded_has_response_marker}")
    print(f"decoded_has_output_json: {audit.decoded_has_output_json}")
    print()
    print("--- Response boundary (char window) ---")
    print(f"response_char_pos: {audit.response_char_pos}")
    print(f"response_token_pos (approx): {audit.response_token_pos}")
    print(audit.boundary_window)
    print()
    if show_token_ids and audit.token_window_text:
        print("--- Token window at response boundary ---")
        print(audit.token_window_text)
        print()
    print(f"decode_roundtrip_parse_ok: {audit.decode_roundtrip_parse_ok}")
    if audit.failed_checks:
        print(f"FAILED CHECKS: {audit.failed_checks}")
    print()


def _render_markdown(
    jsonl_path: Path,
    model_name: str,
    max_seq_length: int,
    sample_mode: str,
    tok_info: Dict[str, Any],
    load_source: str,
    audits: List[SampleAudit],
    summary: AuditSummary,
    eos_token: str,
) -> str:
    lines: List[str] = [
        "# Unsloth Multitask Tokenization Audit",
        "",
        f"- **JSONL:** `{jsonl_path}`",
        f"- **Model:** `{model_name}`",
        f"- **max_seq_length:** {max_seq_length}",
        f"- **sample_mode:** {sample_mode}",
        f"- **EOS appended:** {bool(eos_token)} ({eos_token!r})",
        "",
        "## Tokenizer Summary",
        "",
        f"**Load source:** {load_source}",
        "",
        "| Field | Value |",
        "|-------|-------|",
    ]
    for key, val in tok_info.items():
        lines.append(f"| `{key}` | `{val!r}` |")

    for n, audit in enumerate(audits, start=1):
        lines.extend(
            [
                "",
                f"## Sample {n} — row index {audit.row_index}, pair_id {audit.pair_id!r}",
                "",
                "### Raw instruction",
                "",
                f"Length: {audit.instruction_len} characters",
                "",
                "### Raw input preview",
                "",
                "```text",
                audit.input_preview,
                "```",
                "",
                "### Raw output JSON",
                "",
                "```json",
                audit.output_preview,
                "```",
                "",
                "### Strict parser result",
                "",
                f"- **strict_parse_ok:** `{audit.strict_parse_ok}`",
                "",
            ]
        )
        if audit.parsed_flat:
            lines.append("```json")
            lines.append(json.dumps(audit.parsed_flat, indent=2, ensure_ascii=False))
            lines.append("```")
        else:
            lines.append("_Strict parser returned None._")

        lines.extend(
            [
                "",
                "### Formatted SFT text around response boundary",
                "",
                f"- response_char_pos: `{audit.response_char_pos}`",
                f"- response_token_pos (approx): `{audit.response_token_pos}`",
                f"- input_truncated: `{audit.input_truncated}`",
                "",
                "```text",
                audit.boundary_window,
                "```",
                "",
                "### Tokenization summary",
                "",
                "| Check | Value |",
                "|-------|-------|",
                f"| orig_input_chars | {audit.orig_input_chars} |",
                f"| trunc_input_chars | {audit.trunc_input_chars} |",
                f"| formatted_chars | {audit.formatted_chars} |",
                f"| tokenized_length | {audit.tokenized_length} |",
                f"| formatted_length_ok | {audit.formatted_length_ok} |",
                f"| has ### Instruction: | {audit.has_instruction_marker} |",
                f"| has ### Input: | {audit.has_input_marker} |",
                f"| has ### Response: | {audit.has_response_marker} |",
                f"| has `{{` after response | {audit.has_json_open_after_response} |",
                f"| has final `}}` before EOS | {audit.has_json_close_before_eos} |",
                f"| gold_output_in_formatted | {audit.gold_output_in_formatted} |",
                f"| decoded_has_response_marker | {audit.decoded_has_response_marker} |",
                f"| decoded_has_output_json | {audit.decoded_has_output_json} |",
                "",
                "### Decoded roundtrip check",
                "",
                f"- **decode_roundtrip_parse_ok:** `{audit.decode_roundtrip_parse_ok}`",
                "",
            ]
        )
        if audit.token_window_text:
            lines.extend(
                [
                    "### Token window at response boundary",
                    "",
                    "```text",
                    audit.token_window_text,
                    "```",
                    "",
                ]
            )
        if audit.failed_checks:
            lines.append(f"- **failed_checks:** `{audit.failed_checks}`")
            lines.append("")

    lines.extend(
        [
            "## Validation summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| samples_checked | {summary.samples_checked} |",
            f"| strict_output_parse_ok_count | {summary.strict_output_parse_ok_count} |",
            f"| formatted_length_ok_count | {summary.formatted_length_ok_count} |",
            f"| response_boundary_found_count | {summary.response_boundary_found_count} |",
            f"| output_json_preserved_count | {summary.output_json_preserved_count} |",
            f"| decode_roundtrip_parse_ok_count | {summary.decode_roundtrip_parse_ok_count} |",
            "",
            f"**any_failed_rows:** `{summary.any_failed_rows}`",
            "",
        ]
    )
    return "\n".join(lines)


def _print_summary(summary: AuditSummary) -> None:
    print("=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    print(f"samples_checked: {summary.samples_checked}")
    print(f"strict_output_parse_ok_count: {summary.strict_output_parse_ok_count}")
    print(f"formatted_length_ok_count: {summary.formatted_length_ok_count}")
    print(f"response_boundary_found_count: {summary.response_boundary_found_count}")
    print(f"output_json_preserved_count: {summary.output_json_preserved_count}")
    print(f"decode_roundtrip_parse_ok_count: {summary.decode_roundtrip_parse_ok_count}")
    print(f"any_failed_rows: {summary.any_failed_rows}")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit multitask JSONL tokenization before Unsloth SFT (no training)."
    )
    p.add_argument("--jsonl", type=Path, required=True, help="train.jsonl or test.jsonl")
    p.add_argument(
        "--model-name",
        type=str,
        default="unsloth/gemma-4-e2b-it-unsloth-bnb-4bit",
        help="Same base model as train_unsloth_router.py",
    )
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument(
        "--sample-mode",
        choices=("head", "random", "longest"),
        default="head",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional markdown audit report path",
    )
    p.add_argument(
        "--show-token-ids",
        action="store_true",
        help="Print token IDs in a window around the response boundary",
    )
    p.add_argument("--max-input-chars-print", type=int, default=1500)
    p.add_argument("--max-output-chars-print", type=int, default=2000)
    p.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Load HF processor/tokenizer only (no full 4-bit model). Prefer unsloth_env for Gemma4 parity.",
    )
    p.add_argument(
        "--no-append-eos",
        action="store_true",
        help="Match train_unsloth_router.py --no-append-eos (default: append eos_token).",
    )
    p.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit 1 if any sample fails strict parse, length, boundary, or roundtrip checks.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = args.jsonl.resolve()
    if not jsonl_path.is_file():
        raise SystemExit(f"JSONL not found: {jsonl_path}")

    print(f"Loading JSONL: {jsonl_path}")
    rows = _load_jsonl_rows(jsonl_path)
    indices = _select_indices(rows, args.num_samples, args.sample_mode, args.seed)
    if not indices:
        raise SystemExit("No rows to audit (empty JSONL or num-samples=0).")

    print(f"Loading tokenizer/processor: {args.model_name}")
    processing_class, load_source = _load_processing_class(
        args.model_name,
        args.max_seq_length,
        args.tokenizer_only,
    )
    tok_info = _tokenizer_identity(processing_class)
    _print_tokenizer_summary(tok_info, load_source)

    eos_token = "" if args.no_append_eos else (
        getattr(resolve_tokenizer_for_encode(processing_class), "eos_token", None) or ""
    )
    if eos_token:
        print(f"Appending EOS to formatted SFT text: {eos_token!r}\n")
    else:
        print("Not appending EOS (--no-append-eos).\n")

    summary = AuditSummary()
    audits: List[SampleAudit] = []

    for row_index in indices:
        audit = _audit_one_sample(
            row_index,
            rows[row_index],
            processing_class,
            args.max_seq_length,
            eos_token,
            args.max_input_chars_print,
            args.max_output_chars_print,
            args.show_token_ids,
        )
        audits.append(audit)
        _update_summary(summary, audit)
        _print_sample(audit, args.show_token_ids)

    _print_summary(summary)

    if args.output_md:
        out_path = args.output_md.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md = _render_markdown(
            jsonl_path,
            args.model_name,
            args.max_seq_length,
            args.sample_mode,
            tok_info,
            load_source,
            audits,
            summary,
            eos_token,
        )
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote markdown report: {out_path}")

    if args.fail_on_error and summary.any_failed_rows:
        raise SystemExit(
            f"Audit failed for row indices: {summary.any_failed_rows}. "
            "Inspect markdown / stdout above."
        )


if __name__ == "__main__":
    main()
