"""Fine-tuning utilities for local Gemma models using LoRA."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _label_to_text(label: int) -> str:
    return "Yes" if int(label) == 1 else "No"


def _row_to_instruction(narrative: str, label: int) -> str:
    return (
        "Task: Predict whether Disorders of Lipid Metabolism will be present in the next visit.\n\n"
        f"Patient record:\n{narrative}\n\n"
        f"Prediction:\n{_label_to_text(label)}"
    )


def build_instruction_dataset(df: pd.DataFrame) -> Dataset:
    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "text": _row_to_instruction(
                    narrative=str(row["narrative_current"]),
                    label=int(row["label_lipid_disorder"]),
                )
            }
        )
    return Dataset.from_list(records)


def finetune_gemma_lora(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    model_name: str,
    output_dir: str = "models/gemma_finetuned",
    max_length: int = 1024,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    num_train_epochs: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    max_steps: Optional[int] = None,
    gradient_checkpointing: bool = False,
    force_fp32: bool = False,
    max_grad_norm: float = 1.0,
    smoke: bool = False,
) -> Path:
    """Fine-tune Gemma with LoRA and save adapter/tokenizer locally."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except AttributeError as exc:
        # Some transformers versions fail on Gemma tokenizer_config extra_special_tokens list format.
        model_path = Path(model_name)
        patched_loaded = False
        if model_path.exists() and "list" in str(exc).lower() and "keys" in str(exc).lower():
            cfg_path = model_path / "tokenizer_config.json"
            tok_json_path = model_path / "tokenizer.json"
            if cfg_path.exists() and tok_json_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                extra = cfg.get("extra_special_tokens")
                if isinstance(extra, list):
                    cfg["extra_special_tokens"] = {f"extra_token_{i}": tok for i, tok in enumerate(extra)}
                    tmp_dir = Path(tempfile.mkdtemp(prefix="gemma_tokenizer_patch_"))
                    (tmp_dir / "tokenizer_config.json").write_text(
                        json.dumps(cfg, indent=2), encoding="utf-8"
                    )
                    shutil.copy2(tok_json_path, tmp_dir / "tokenizer.json")
                    logger.warning(
                        "Patched tokenizer_config extra_special_tokens format in temp dir: %s",
                        tmp_dir,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(str(tmp_dir))
                    patched_loaded = True
        if not patched_loaded:
            logger.warning(
                "Fast tokenizer load failed for %s (%s); retrying with use_fast=False.",
                model_name,
                exc,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    except ValueError as exc:
        if "model type `gemma4`" in str(exc):
            raise RuntimeError(
                "Your installed transformers version does not support Gemma 4 checkpoints yet. "
                "Upgrade dependencies (for example: `pip install -r requirements.txt --upgrade`) "
                "and re-run the smoke test."
            ) from exc
        raise
    # Gemma-4 requires explicit LoRA target modules with current peft releases.
    default_target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules or default_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    try:
        model = get_peft_model(model, lora_cfg)
    except ValueError as exc:
        if "Gemma4ClippableLinear" in str(exc):
            logger.warning(
                "LoRA target modules are incompatible with Gemma4ClippableLinear. "
                "Reloading base model and retrying with target_modules=['linear']."
            )
            # Reload a clean base model before second PEFT attempt to avoid double-wrap side effects.
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["linear"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
        else:
            raise
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    logger.info("LoRA parameters prepared for model=%s", model_name)

    if max_train_samples is not None:
        train_df = train_df.head(max_train_samples).copy()
    if val_df is not None and max_eval_samples is not None:
        val_df = val_df.head(max_eval_samples).copy()
    if train_df.empty:
        raise ValueError("Training split is empty after sampling.")
    if val_df is not None and val_df.empty:
        logger.warning("Validation split is empty after sampling; proceeding without eval_dataset.")
        val_df = None

    train_ds = build_instruction_dataset(train_df)
    val_ds = build_instruction_dataset(val_df) if val_df is not None else None

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        out = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized_train = train_ds.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized_val = (
        val_ds.map(_tokenize, batched=True, remove_columns=["text"]) if val_ds is not None else None
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    effective_max_steps = max_steps if max_steps is not None else -1
    use_cuda = torch.cuda.is_available() and not force_fp32
    args_kwargs = dict(
        output_dir=str(output_path / "checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        max_steps=effective_max_steps,
        eval_steps=1 if smoke else 20,
        logging_steps=1 if smoke else 20,
        save_steps=200,
        save_total_limit=2,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        bf16=False,
        fp16=use_cuda,
        report_to=[],
    )
    eval_mode = "steps" if tokenized_val is not None else "no"
    try:
        args = TrainingArguments(evaluation_strategy=eval_mode, **args_kwargs)
    except TypeError:
        # transformers>=5 renamed this argument.
        args = TrainingArguments(eval_strategy=eval_mode, **args_kwargs)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    try:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        # transformers>=5 removed `tokenizer` from Trainer.__init__.
        trainer = Trainer(**trainer_kwargs)
    trainer.train()
    if trainer.state.log_history:
        for row in trainer.state.log_history:
            for key in ("loss", "eval_loss", "grad_norm"):
                val = row.get(key)
                if isinstance(val, float) and not math.isfinite(val):
                    raise RuntimeError(
                        f"Smoke training produced non-finite metric: {key}={val}. "
                        "Reduce LR / disable fp16 / adjust LoRA settings before full run."
                    )

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Saved fine-tuned Gemma LoRA adapter to %s", output_path)
    return output_path
