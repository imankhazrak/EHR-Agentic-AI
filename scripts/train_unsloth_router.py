#!/usr/bin/env python3
"""LoRA fine-tune Gemma for MIMIC multitask JSON (seven tasks + reasoning).

Expects Alpaca-style JSONL from ``dataset_for_unsloth`` (instruction, input, output).
Build that folder first (separate step — not imported here)::

    python -m src.scripts.export_multitask_unsloth_jsonl

That exporter writes ``output`` strings validated by ``parse_multitask_output`` in
``src/llm/output_parser.py``; this trainer imports the same parser to reject bad rows.

The formatted ``text`` field matches ``scripts/test_unsloth_router.py`` so eval matches train.
Long ``input`` narratives are shortened with ``src.utils.alpaca_multitask_fit`` so the
``### Response:`` block and gold JSON are not truncated by the 2048 token cap.

**Loss masking (default):** only multitask JSON tokens (+ EOS) are supervised; instruction,
visit input, and the ``### Response:`` header are masked (``-100``). See
``src.utils.unsloth_multitask_labels`` and ``--debug-label-mask-check``. TRL 0.24 with a
plain ``text`` column otherwise trains on the full sequence (including long clinical input).

**Environment:** use the Conda env ``unsloth_env`` (Unsloth, TRL, CUDA-enabled PyTorch).
Do not use the bare system Python; it will miss ``unsloth`` and match Slurm jobs under
``slurm/``.

Run from repo root (so ``src`` imports resolve)::

    conda activate unsloth_env
    cd /path/to/EHR-Agentic-AI
    python scripts/train_unsloth_router.py
    python scripts/train_unsloth_router.py --max-steps 500 --eval-jsonl dataset_for_unsloth/test.jsonl
    python scripts/train_unsloth_router.py --debug-only --debug-label-mask-check 5 --train-max-samples 5

Slurm::

    sbatch slurm/train_unsloth_mt_smoke.slurm          # quick pipeline check
    sbatch slurm/train_unsloth_mt_natural_dist.slurm   # full train + test eval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from datasets import Dataset, load_dataset
# Schema + strict JSON checks: src/llm/output_parser.py (used by export + eval stack).
# JSONL files: produced by src/scripts/export_multitask_unsloth_jsonl.py (run as __main__).
from src.llm.output_parser import MULTITASK_JSON_TASK_KEYS, parse_multitask_output
from src.utils.alpaca_multitask_fit import fit_visit_input_for_token_cap, resolve_tokenizer_for_encode
from src.utils.unsloth_multitask_labels import (
    RESPONSE_MARKER,
    audit_label_mask_row,
    build_completion_mask_for_input_ids,
)


def build_multitask_sft_text(
    instruction: str,
    input_text: str,
    output_text: str,
    eos_token: str,
) -> str:
    """Same layout as ``test_unsloth_router.py`` generation prompt + completed response."""
    body = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""
    if eos_token:
        return body + eos_token
    return body


def _validate_output_json(raw: str, row_index: int) -> None:
    """Ensure multitask JSON matches strict eval parser."""
    parsed = parse_multitask_output(raw)
    if parsed is None:
        raise ValueError(
            f"Row {row_index}: output is not valid strict multitask JSON "
            f"(keys {list(MULTITASK_JSON_TASK_KEYS)} + reasoning). First 200 chars: {raw[:200]!r}"
        )


def _make_sft_format_fn(tokenizer: Any, max_seq_length: int, eos_token: str):
    """Batched map: shorten long ``input`` so ``text`` never exceeds ``max_seq_length`` tokens."""

    def _format_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        inst = examples["instruction"]
        n = len(inst)
        inp = examples.get("input")
        if inp is None:
            inp = [""] * n
        out = examples["output"]
        texts: List[str] = []
        for i in range(n):
            instruction = str(inst[i])
            input_text = str(inp[i] if i < len(inp) else "")
            output_text = str(out[i])
            tail = f"\n\n### Response:\n{output_text}{eos_token}"
            trunc_input, _ = fit_visit_input_for_token_cap(
                tokenizer,
                instruction=instruction,
                input_text=input_text,
                tail_after_input=tail,
                max_length=max_seq_length,
            )
            texts.append(
                build_multitask_sft_text(
                    instruction,
                    trunc_input,
                    output_text,
                    eos_token,
                )
            )
        return {"text": texts}

    return _format_batch


def _format_one_row(
    tokenizer: Any,
    max_seq_length: int,
    eos_token: str,
    instruction: str,
    input_text: str,
    output_text: str,
) -> tuple[str, List[int]]:
    """Return ``(formatted_text, input_ids)`` using the same fit path as training."""
    tail = f"\n\n### Response:\n{output_text}{eos_token}"
    trunc_input, input_ids = fit_visit_input_for_token_cap(
        tokenizer,
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
    return formatted, input_ids


def _prepare_split_lm_text(
    ds: Dataset,
    tokenizer: Any,
    max_seq_length: int,
    tokenizer_eos: str,
    validate: bool,
    desc: str,
) -> Dataset:
    """TRL language-modeling path: ``text`` column only → loss on all tokens by default."""
    if validate:
        for i, row in enumerate(ds):
            _validate_output_json(str(row.get("output", "")), i)
    cols = list(ds.column_names)
    fmt = _make_sft_format_fn(tokenizer, max_seq_length, tokenizer_eos)
    return ds.map(
        fmt,
        batched=True,
        remove_columns=cols,
        desc=f"Format {desc}",
    )


def _prepare_split_response_only(
    ds: Dataset,
    tokenizer: Any,
    max_seq_length: int,
    tokenizer_eos: str,
    validate: bool,
    desc: str,
) -> Dataset:
    """Pre-tokenize with ``completion_mask`` (supervise JSON + EOS only)."""
    hf_tok = resolve_tokenizer_for_encode(tokenizer)

    def _tokenize_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        inst = examples["instruction"]
        n = len(inst)
        inp = examples.get("input") or [""] * n
        out = examples["output"]
        input_ids_col: List[List[int]] = []
        completion_mask_col: List[List[int]] = []
        text_col: List[str] = []
        for i in range(n):
            formatted, ids = _format_one_row(
                tokenizer,
                max_seq_length,
                tokenizer_eos,
                str(inst[i]),
                str(inp[i] if i < len(inp) else ""),
                str(out[i]),
            )
            mask, _, _ = build_completion_mask_for_input_ids(hf_tok, formatted, ids)
            text_col.append(formatted)
            input_ids_col.append(ids)
            completion_mask_col.append(mask)
        return {
            "text": text_col,
            "input_ids": input_ids_col,
            "completion_mask": completion_mask_col,
        }

    if validate:
        for i, row in enumerate(ds):
            _validate_output_json(str(row.get("output", "")), i)
    cols = list(ds.column_names)
    return ds.map(
        _tokenize_batch,
        batched=True,
        remove_columns=cols,
        desc=f"Tokenize+mask {desc}",
    )


def _prepare_split(
    ds: Dataset,
    tokenizer: Any,
    max_seq_length: int,
    tokenizer_eos: str,
    validate: bool,
    desc: str,
    *,
    response_only: bool,
) -> Dataset:
    if response_only:
        return _prepare_split_response_only(
            ds, tokenizer, max_seq_length, tokenizer_eos, validate, desc
        )
    return _prepare_split_lm_text(
        ds, tokenizer, max_seq_length, tokenizer_eos, validate, desc
    )


def _run_label_mask_audit(
    raw_ds: Dataset,
    tokenizer: Any,
    max_seq_length: int,
    eos_token: str,
    n_check: int,
    *,
    output_md: Optional[Path] = None,
) -> bool:
    """Print/save label-mask audit for the first ``n_check`` raw JSONL rows. Returns all_ok."""
    hf_tok = resolve_tokenizer_for_encode(tokenizer)
    n = min(n_check, len(raw_ds))
    lines: List[str] = [
        "# Unsloth multitask label-mask audit",
        "",
        f"- samples: {n}",
        f"- max_seq_length: {max_seq_length}",
        f"- supervision starts at first `{{` after `{RESPONSE_MARKER}`",
        "",
    ]
    all_ok = True
    failed_rows: List[int] = []

    print("=" * 72)
    print("LABEL-MASK AUDIT (response-only supervision)")
    print("=" * 72)

    for i in range(n):
        row = raw_ds[i]
        instruction = str(row.get("instruction") or "")
        input_text = str(row.get("input") or "")
        output_text = str(row.get("output") or "")
        formatted, input_ids = _format_one_row(
            tokenizer, max_seq_length, eos_token, instruction, input_text, output_text
        )
        mask, supervise_start, prefix_aligned = build_completion_mask_for_input_ids(
            hf_tok, formatted, input_ids
        )
        audit = audit_label_mask_row(
            row_index=i,
            pair_id=row.get("pair_id"),
            formatted_text=formatted,
            input_ids=input_ids,
            completion_mask=mask,
            hf_tokenizer=hf_tok,
            output_text=output_text,
        )

        print(f"\n--- row_index={audit['row_index']} pair_id={audit['pair_id']!r} ---")
        print(f"tokenized_length: {audit['tokenized_length']}")
        print(f"response_boundary_char_pos: {audit['response_boundary_char_pos']}")
        print(f"json_supervision_char_pos: {audit['json_supervision_char_pos']}")
        print(f"supervise_start_token_index: {audit['supervise_start_token_index']}")
        print(f"prefix_token_alignment_ok: {audit['prefix_token_alignment_ok']}")
        print(f"n_masked_tokens: {audit['n_masked_tokens']}")
        print(f"n_supervised_tokens: {audit['n_supervised_tokens']}")
        print(f"pct_supervised: {audit['pct_supervised']}")
        print(f"has_response_marker: {audit['has_response_marker']}")
        print(f"supervised_starts_with_brace: {audit['supervised_starts_with_brace']}")
        print(
            f"supervised_starts_with_lipid_next: {audit.get('supervised_starts_with_lipid_next')}"
        )
        print(f"supervised_has_all_task_keys: {audit['supervised_has_all_task_keys']}")
        print(f"supervised_parse_ok: {audit['supervised_parse_ok']}")
        print("\n[A] Prompt/context (masked tokens):")
        print(audit["prompt_context_decoded"][:2000])
        print("\n[B] Supervised label region (JSON + EOS):")
        print(audit["supervised_decoded"][:2500])
        if audit["failed_checks"]:
            all_ok = False
            failed_rows.append(i)
            print(f"FAILED: {audit['failed_checks']}")

        lines.extend(
            [
                f"## Row {i} (pair_id={audit['pair_id']!r})",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| tokenized_length | {audit['tokenized_length']} |",
                f"| n_masked_tokens | {audit['n_masked_tokens']} |",
                f"| n_supervised_tokens | {audit['n_supervised_tokens']} |",
                f"| pct_supervised | {audit['pct_supervised']} |",
                f"| supervised_parse_ok | {audit['supervised_parse_ok']} |",
                "",
                "### [A] Prompt/context (masked)",
                "",
                "```text",
                audit["prompt_context_decoded"][:3000],
                "```",
                "",
                "### [B] Supervised JSON region",
                "",
                "```text",
                audit["supervised_decoded"][:3500],
                "```",
                "",
            ]
        )

    print("\n" + "=" * 72)
    print(f"Label-mask audit: {'PASS' if all_ok else 'FAIL'}  failed_rows={failed_rows}")
    print("=" * 72)

    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        lines.append(f"\n**Result:** {'PASS' if all_ok else 'FAIL'}  failed_rows={failed_rows}\n")
        output_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {output_md}")

    return all_ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsloth LoRA SFT for multitask clinical JSON.")
    p.add_argument(
        "--model-name",
        type=str,
        default="unsloth/gemma-4-e2b-it-unsloth-bnb-4bit",
        help="Base model id (Unsloth Gemma build).",
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        default=_ROOT / "dataset_for_unsloth/train.jsonl",
        help="Training JSONL (instruction, input, output).",
    )
    p.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help="Optional eval JSONL; default None (no eval). Use dataset_for_unsloth/test.jsonl for held-out eval.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="natural",
        help="Logical run id; used in default output folder name (sanitized).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint root. If omitted, uses outputs/unsloth_mt_lora_<run_name>_<UTC_timestamp>/",
    )
    p.add_argument(
        "--train-max-samples",
        type=int,
        default=None,
        help="Use only the first N training rows (smoke tests).",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Use only the first N eval rows (smoke tests).",
    )
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Max checkpoints to keep on disk (Hugging Face TrainingArguments).",
    )
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-validate-multitask-json",
        action="store_true",
        help="Skip strict parse_multitask_output check on each training row.",
    )
    p.add_argument(
        "--no-append-eos",
        action="store_true",
        help="Do not append tokenizer eos after the assistant JSON.",
    )
    loss = p.add_mutually_exclusive_group()
    loss.add_argument(
        "--train-on-response-only",
        dest="response_only",
        action="store_true",
        default=True,
        help="Supervise only JSON (+ EOS); mask instruction/input/### Response: (default).",
    )
    loss.add_argument(
        "--train-on-full-text",
        dest="response_only",
        action="store_false",
        help="Legacy: TRL LM path — loss on all tokens in ``text`` (debug only).",
    )
    p.add_argument(
        "--debug-label-mask-check",
        type=int,
        default=None,
        metavar="N",
        help="Audit label masking on N raw train rows before training; exit 1 if any fail.",
    )
    p.add_argument(
        "--debug-label-mask-md",
        type=Path,
        default=None,
        help="Optional markdown path for --debug-label-mask-check report.",
    )
    p.add_argument(
        "--debug-only",
        action="store_true",
        help="Run --debug-label-mask-check (if set) and exit before trainer.train().",
    )
    p.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Load HF processor/tokenizer only (no 4-bit model). For --debug-only on CPU.",
    )
    return p.parse_args()


def _load_tokenizer_or_processor(
    model_name: str,
    max_seq_length: int,
    *,
    tokenizer_only: bool,
) -> Any:
    """Load Unsloth processor (training) or HF processor (CPU debug audits)."""
    if tokenizer_only:
        from transformers import AutoProcessor, AutoTokenizer

        try:
            return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception as exc:
            print(f"AutoProcessor failed ({exc}); falling back to AutoTokenizer.")
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Unsloth must load before transformers/trl/peft when training on GPU.
    from unsloth import FastLanguageModel

    _model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    del _model
    return tokenizer


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (args.run_name or "run")).strip("_") or "run"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (_ROOT / "outputs" / f"unsloth_mt_lora_{safe}_{ts}").resolve()


def main() -> None:
    args = parse_args()
    train_path = args.train_jsonl.resolve()
    if not train_path.is_file():
        raise SystemExit(f"Train JSONL not found: {train_path}")

    response_only = bool(args.response_only)
    if not response_only:
        print(
            "WARNING: --train-on-full-text — TRL 0.24 will compute loss on ALL tokens in "
            "each ``text`` example (instruction + visit narrative + JSON). Not recommended "
            "for multitask JSON."
        )
    else:
        print(
            "Response-only loss: JSON + EOS supervised; instruction/input/### Response: masked."
        )

    out_dir = _resolve_output_dir(args)
    print(f"Output directory: {out_dir}")

    # V100 (CC 7.x) and older GPUs must use fp16; bf16 needs Ampere+ (CC >= 8.0).
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        use_bf16 = major >= 8 and bool(torch.cuda.is_bf16_supported())
        use_fp16 = not use_bf16
        print(f"Training precision: bf16={use_bf16} fp16={use_fp16} (CC {major}.x)")

    tokenizer_only = bool(args.tokenizer_only) or (
        args.debug_only and args.debug_label_mask_check is not None
    )
    if tokenizer_only:
        print("Loading tokenizer/processor only (--tokenizer-only or debug-only audit)...")
        tokenizer = _load_tokenizer_or_processor(
            args.model_name, args.max_seq_length, tokenizer_only=True
        )
    else:
        from unsloth import FastLanguageModel

        print("Loading model + tokenizer...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )

    hf_tok = resolve_tokenizer_for_encode(tokenizer)
    eos = "" if args.no_append_eos else (hf_tok.eos_token or getattr(tokenizer, "eos_token", None) or "")
    if eos:
        print(f"Appending eos token to each training example ({eos!r}).")

    if not tokenizer_only:
        from unsloth import FastLanguageModel

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
        )

    print(f"Loading train dataset: {train_path}")
    train_raw = load_dataset("json", data_files=str(train_path), split="train")
    if args.train_max_samples is not None:
        n = max(0, min(int(args.train_max_samples), len(train_raw)))
        train_raw = train_raw.select(range(n))
        print(f"Using train subset: first {n} rows (--train-max-samples).")
    validate = not args.no_validate_multitask_json
    if validate:
        print("Validating each row with parse_multitask_output (multitask experiment).")

    if args.debug_label_mask_check is not None:
        n_audit = int(args.debug_label_mask_check)
        md_path = args.debug_label_mask_md
        if md_path is None and response_only:
            md_path = out_dir / "label_mask_audit.md"
        ok = _run_label_mask_audit(
            train_raw,
            tokenizer,
            args.max_seq_length,
            eos,
            n_audit,
            output_md=md_path,
        )
        if not ok:
            raise SystemExit("Label-mask audit failed (see stdout / markdown report).")

    train_ds = _prepare_split(
        train_raw,
        tokenizer,
        args.max_seq_length,
        eos,
        validate,
        "train",
        response_only=response_only,
    )

    eval_ds: Optional[Dataset] = None
    if args.eval_jsonl is not None:
        eval_path = args.eval_jsonl.resolve()
        if not eval_path.is_file():
            raise SystemExit(f"Eval JSONL not found: {eval_path}")
        print(f"Loading eval dataset: {eval_path}")
        eval_raw = load_dataset("json", data_files=str(eval_path), split="train")
        if args.eval_max_samples is not None:
            n_e = max(0, min(int(args.eval_max_samples), len(eval_raw)))
            eval_raw = eval_raw.select(range(n_e))
            print(f"Using eval subset: first {n_e} rows (--eval-max-samples).")
        eval_ds = _prepare_split(
            eval_raw,
            tokenizer,
            args.max_seq_length,
            eos,
            validate,
            "eval",
            response_only=response_only,
        )

    if args.debug_only:
        print("--debug-only: skipping trainer.train().")
        return

    if tokenizer_only:
        raise SystemExit(
            "Full training requires a GPU and Unsloth model load. "
            "Omit --tokenizer-only / --debug-only for training."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir = out_dir / "final_lora"

    # TRL 0.24: ``text``-only datasets → language modeling → loss on full sequence unless we
    # pre-tokenize with ``completion_mask`` and set ``completion_only_loss=True``.
    dataset_kwargs: Optional[Dict[str, Any]] = None
    completion_only_loss = False
    if response_only:
        dataset_kwargs = {"skip_prepare_dataset": True}
        completion_only_loss = True
        print(
            "SFTConfig: completion_only_loss=True, dataset_kwargs.skip_prepare_dataset=True "
            "(DataCollatorForLanguageModeling masks prompt tokens via completion_mask)."
        )
    else:
        print(
            "SFTConfig: completion_only_loss=False (default LM path on ``text`` column)."
        )

    ta_kwargs: Dict[str, Any] = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_8bit",
        report_to="none",
        seed=args.seed,
        save_total_limit=args.save_total_limit,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        completion_only_loss=completion_only_loss,
        dataset_kwargs=dataset_kwargs,
    )
    if eval_ds is not None:
        ta_kwargs.update(
            eval_strategy="steps",
            eval_steps=args.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            do_eval=True,
        )
    else:
        ta_kwargs["eval_strategy"] = "no"

    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(**ta_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting training (multitask JSON SFT)...")
    trainer.train()

    print(f"Saving LoRA + tokenizer -> {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    manifest = {
        "run_name": args.run_name,
        "output_dir": str(out_dir),
        "model_name": args.model_name,
        "train_jsonl": str(train_path),
        "train_max_samples": args.train_max_samples,
        "eval_jsonl": str(args.eval_jsonl.resolve()) if args.eval_jsonl else None,
        "eval_max_samples": args.eval_max_samples,
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "multitask_keys": list(MULTITASK_JSON_TASK_KEYS),
        "format": "Alpaca ### Instruction / ### Input / ### Response (matches test_unsloth_router.py)",
        "input_truncation": "visit_input_prefix_fit_via_src.utils.alpaca_multitask_fit",
        "loss_masking": (
            "response_only_json_via_completion_mask"
            if response_only
            else "full_text_lm_default_trl"
        ),
        "supervision_starts_at": "first_json_brace_after_response_marker" if response_only else "all_tokens",
        "final_lora": str(final_dir),
    }
    with (out_dir / "train_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_dir / 'train_manifest.json'}")


if __name__ == "__main__":
    main()
