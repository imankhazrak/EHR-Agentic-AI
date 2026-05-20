#!/usr/bin/env python3
"""Run multitask JSON generation with a saved Unsloth LoRA adapter (eval).

Prompt layout matches ``scripts/train_unsloth_router.py`` (Alpaca blocks, no
completion after ``### Response:``). Gold and predictions are parsed with
``parse_multitask_output`` from ``src.llm.output_parser`` (same strict rules as
training data validation).

**Environment:** ``conda activate unsloth_env`` (see ``slurm/test_unsloth_router.slurm``).

Run from repo root::

    python scripts/test_unsloth_router.py \\
      --model-path outputs/.../final_lora \\
      --output-jsonl outputs/.../test_predictions_2nd_try_fixed.jsonl

After a Slurm TIMEOUT, continue appending from the next index (same paths as the
interrupted run)::

    python scripts/test_unsloth_router.py --resume \\
      --model-path outputs/.../final_lora \\
      --output-jsonl outputs/.../partial.jsonl

Score predictions (CPU only)::

    python scripts/score_unsloth_multitask_jsonl.py outputs/.../test_predictions_2nd_try_fixed.jsonl --all-modes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.llm.output_parser import parse_multitask_output
from src.utils.alpaca_multitask_fit import fit_visit_input_for_token_cap, resolve_tokenizer_for_encode


def build_multitask_generation_prompt(instruction: str, input_text: str) -> str:
    """Prefix only — matches training ``build_multitask_sft_text`` up to the assistant slot."""
    return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""


def extract_response(decoded_text: str) -> str:
    if "### Response:" in decoded_text:
        return decoded_text.split("### Response:", 1)[-1].strip()
    return decoded_text.strip()


def _truncate_input_to_fit_prompt(
    tokenizer: Any,
    instruction: str,
    input_text: str,
    max_length: int,
) -> Tuple[str, List[int]]:
    """Delegate to ``fit_visit_input_for_token_cap`` (shared with ``train_unsloth_router.py``)."""
    return fit_visit_input_for_token_cap(
        tokenizer,
        instruction=instruction,
        input_text=input_text,
        tail_after_input="\n\n### Response:\n",
        max_length=max_length,
    )


def _serialize_parse_result(parsed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if parsed is None:
        return None
    out: Dict[str, Any] = {}
    for k, v in parsed.items():
        if isinstance(v, float):
            out[k] = float(v)
        elif isinstance(v, (int, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _resume_start_index(output_path: Path) -> int:
    """Next dataset row index to write, from existing JSONL (skips invalid lines)."""
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return 0
    nxt = 0
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                nxt = max(nxt, int(rec["index"]) + 1)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return nxt


def parse_args() -> argparse.Namespace:
    default_model = (
        _ROOT
        / "outputs"
        / "unsloth_mt_lora_natural_dist_job47427323_20260513_021406"
        / "final_lora"
    )
    default_out = default_model.parent / "test_predictions_2nd_try_fixed.jsonl"
    p = argparse.ArgumentParser(description="Multitask JSON eval with Unsloth LoRA.")
    p.add_argument(
        "--model-path",
        type=Path,
        default=default_model,
        help="Directory with saved LoRA + tokenizer (final_lora).",
    )
    p.add_argument(
        "--test-jsonl",
        type=Path,
        default=_ROOT / "dataset_for_unsloth" / "test.jsonl",
        help="Test JSONL (instruction, input, output; optional pair_id).",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=default_out,
        help="Where to write one JSON record per line.",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Must match training max_length (see train_unsloth_router.py).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Max tokens generated for the multitask JSON completion (JSON is ~150–400 tokens).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append to output JSONL, skipping rows before the next index after the last valid line.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    test_path = args.test_jsonl.resolve()
    output_path = args.output_jsonl.resolve()

    if not model_path.is_dir():
        raise SystemExit(f"Model path not found or not a directory: {model_path}")
    if not test_path.is_file():
        raise SystemExit(f"Test JSONL not found: {test_path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available; pass --device cpu.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    if args.resume:
        if not output_path.is_file():
            raise SystemExit("--resume requires an existing --output-jsonl file.")
        start_idx = _resume_start_index(output_path)
        print(f"Resume: next dataset index = {start_idx} (from {output_path})")

    print(f"Loading LoRA from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    hf_tok = resolve_tokenizer_for_encode(tokenizer)

    print(f"Loading test JSONL: {test_path}")
    test_dataset = load_dataset("json", data_files=str(test_path), split="train")
    n_total = len(test_dataset)
    print(f"Samples: {n_total}  max_seq_length={args.max_seq_length}")

    if start_idx >= n_total:
        print(f"Resume: nothing to do (indices 0..{n_total - 1} already covered).")
        return

    tail = test_dataset.select(range(start_idx, n_total))
    file_mode = "a" if start_idx > 0 else "w"

    dev = torch.device(device)

    with output_path.open(file_mode, encoding="utf-8") as f:
        for k, example in enumerate(tqdm(tail, desc="Generating", total=len(tail))):
            idx = start_idx + k
            instruction = str(example["instruction"])
            input_text = str(example.get("input") or "")
            gold_raw = str(example.get("output") or "")

            _trunc_input, input_ids_list = _truncate_input_to_fit_prompt(
                tokenizer, instruction, input_text, args.max_seq_length
            )
            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=dev)
            attn = torch.ones_like(input_ids)
            inputs = {"input_ids": input_ids, "attention_mask": attn}
            prompt_len = int(input_ids.shape[1])

            gen_kwargs: Dict[str, Any] = dict(
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=hf_tok.pad_token_id or hf_tok.eos_token_id,
                eos_token_id=hf_tok.eos_token_id,
            )

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            gen_only = outputs[0, prompt_len:]
            prediction_text = hf_tok.decode(gen_only, skip_special_tokens=True).strip()
            if "### Response:" in prediction_text:
                prediction_text = prediction_text.split("### Response:", 1)[-1].strip()

            gold_parsed = parse_multitask_output(gold_raw)
            pred_parsed = parse_multitask_output(prediction_text)

            record: Dict[str, Any] = {
                "index": idx,
                "pair_id": example.get("pair_id"),
                "instruction": instruction,
                "input": input_text,
                "input_was_truncated": _trunc_input != input_text,
                "gold_output": example.get("output", ""),
                "prediction_text": prediction_text,
                "gold_parse_ok": gold_parsed is not None,
                "pred_parse_ok": pred_parsed is not None,
                "gold_parsed_flat": _serialize_parse_result(gold_parsed),
                "pred_parsed_flat": _serialize_parse_result(pred_parsed),
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    print(f"Wrote predictions to: {output_path}")


if __name__ == "__main__":
    main()
