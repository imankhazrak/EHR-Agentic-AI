"""Sanity-check instruction text, answer-only label masking, and eval prompt alignment.

Run from repo root:
  python -m src.scripts.verify_finetune_feed --config configs/default.yaml
  python -m src.scripts.verify_finetune_feed --config configs/default.yaml --overrides configs/finetune_answer_only_5epochs.yaml

Exit code 1 if any sampled row fails critical checks (truncation ate the label, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from transformers import AutoTokenizer

from src.llm.finetune_gemma import _row_to_instruction
from src.utils.config_utils import load_config


def _apply_answer_only_labels(
    input_ids: List[int],
    full_text: str,
    tokenizer,
    max_length: int,
) -> Tuple[List[int], int]:
    """Mirror ``finetune_gemma._tokenize`` answer-only masking. Returns (labels, prompt_len_used)."""
    labels = list(input_ids)
    marker = "\nPrediction:\n"
    idx = full_text.find(marker)
    prompt = full_text[: idx + len(marker)] if idx >= 0 else full_text
    prompt_tok = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    prompt_ids: List[int] = prompt_tok["input_ids"]
    plen = min(len(prompt_ids), len(labels))
    for j in range(plen):
        labels[j] = -100
    return labels, plen


def _supervised_tokens(
    labels: List[int],
    pad_id: int | None,
) -> List[int]:
    out: List[int] = []
    for t in labels:
        if t == -100:
            continue
        if pad_id is not None and t == pad_id:
            continue
        out.append(t)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--n", type=int, default=30, help="Rows to check from train_ft or train.")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer directory (defaults to llm.model_name / model from config).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides or None)
    processed = Path(cfg["paths"]["processed"])
    llm_cfg = cfg.get("llm", {})
    train_cfg = llm_cfg.get("finetune", {})
    max_length = int(train_cfg.get("max_length", 2048))
    answer_only = bool(train_cfg.get("answer_only_loss", True))
    model_name = args.tokenizer_path or llm_cfg.get("model_name") or llm_cfg.get("model")

    train_ft = processed / "train_ft.csv"
    train_path = processed / "train.csv"
    if train_ft.exists():
        df = pd.read_csv(train_ft).head(args.n)
        src = str(train_ft)
    else:
        df = pd.read_csv(train_path).head(args.n)
        src = str(train_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_name))
    except (AttributeError, OSError, ValueError, TypeError) as exc:
        print(f"Tokenizer fast load failed ({exc}); retrying use_fast=False.")
        tokenizer = AutoTokenizer.from_pretrained(str(model_name), use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    failures = 0
    print(f"Tokenizer: {model_name}\nmax_length={max_length}  answer_only_loss={answer_only}\nrows from {src}\n")

    for pos, (i, row) in enumerate(df.iterrows()):
        label = int(row["label_lipid_disorder"])
        expected = "Yes" if label == 1 else "No"
        narrative = str(row["narrative_current"])
        full_text = _row_to_instruction(narrative, label)

        enc = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = enc["input_ids"]
        labels, plen = _apply_answer_only_labels(input_ids, full_text, tokenizer, max_length)
        sup = _supervised_tokens(labels, pad_id)
        decoded = tokenizer.decode(sup, skip_special_tokens=True).strip()

        # Truncation removed tail: full string contains answer but encoded tail may not
        truncated_answer = False
        if expected in full_text:
            tail_ids = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding=False,
            )["input_ids"]
            tail_decoded = tokenizer.decode(tail_ids, skip_special_tokens=True)
            truncated_answer = expected not in tail_decoded

        ok = decoded.replace(" ", "") == expected or decoded == expected
        if not ok and sup:
            ok = expected.lower() in decoded.lower() and len(decoded) <= len(expected) + 4

        status = "OK" if ok and not truncated_answer else "FAIL"
        if status == "FAIL":
            failures += 1

        if status == "FAIL" or pos < 5:
            print(f"--- row index {i} pair_id={row.get('pair_id', '?')} label={label} ---")
            print(f"  supervised_token_count={len(sup)}  prompt_masked_len={plen}")
            print(f"  decoded_supervised={decoded!r}  expected={expected!r}")
            print(f"  truncated_answer_gone_from_tail={truncated_answer}")
            if not ok:
                print("  [check] answer-only region does not decode to Yes/No — training signal may be wrong.")

    def _eval_style_prompt(narrative: str) -> str:
        return (
            "Task: Predict whether Disorders of Lipid Metabolism will be present in the next visit.\n\n"
            f"Patient record:\n{narrative}\n\n"
            "Prediction:\n"
        )

    def _train_through_marker(narrative: str, label: int) -> str:
        t = _row_to_instruction(narrative, label)
        m = "\nPrediction:\n"
        k = t.rfind(m)
        return t[: k + len(m)] if k >= 0 else t

    probe = "probe-narrative-123"
    if _eval_style_prompt(probe) != _train_through_marker(probe, 0) or _eval_style_prompt(probe) != _train_through_marker(
        probe, 1
    ):
        print("WARN: eval-style prompt differs from training text through `Prediction:\\n` (review templates).")
    else:
        print("\nTraining instruction prefix matches finetuned eval prompt through `Prediction:\\n`.")

    print(f"\nDone. failures={failures} / {len(df)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
