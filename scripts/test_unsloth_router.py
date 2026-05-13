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
      --output-jsonl outputs/.../test_predictions_2nd_try.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.llm.output_parser import parse_multitask_output


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


def parse_args() -> argparse.Namespace:
    default_model = (
        _ROOT
        / "outputs"
        / "unsloth_mt_lora_natural_dist_job47427323_20260513_021406"
        / "final_lora"
    )
    default_out = default_model.parent / "test_predictions_2nd_try.jsonl"
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
        default=512,
        help="Max tokens generated for the multitask JSON completion.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu.",
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

    print(f"Loading LoRA from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print(f"Loading test JSONL: {test_path}")
    test_dataset = load_dataset("json", data_files=str(test_path), split="train")
    print(f"Samples: {len(test_dataset)}  max_seq_length={args.max_seq_length}")

    dev = torch.device(device)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(tqdm(test_dataset, desc="Generating")):
            instruction = str(example["instruction"])
            input_text = str(example.get("input") or "")
            gold_raw = str(example.get("output") or "")

            prompt = build_multitask_generation_prompt(instruction, input_text)
            inputs = tokenizer(
                text=prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_length,
            ).to(dev)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction_text = extract_response(decoded)

            gold_parsed = parse_multitask_output(gold_raw)
            pred_parsed = parse_multitask_output(prediction_text)

            record: Dict[str, Any] = {
                "index": idx,
                "pair_id": example.get("pair_id"),
                "instruction": instruction,
                "input": input_text,
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
