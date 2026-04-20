"""Fine-tuning utilities for local Gemma models using LoRA."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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

from src.evaluation.metrics import compute_metrics
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _causal_lm_from_pretrained(model_name: str, *, smoke: bool) -> AutoModelForCausalLM:
    """Load causal LM; smoke + CUDA uses a single device map to avoid meta/offload tensors that break PEFT save."""
    if smoke and torch.cuda.is_available():
        idx = int(torch.cuda.current_device())
        kwargs: Dict[str, object] = {"device_map": {"": idx}}
    else:
        kwargs = {"device_map": "auto"}
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def _finetune_eval_prompt(narrative: str) -> str:
    """Prompt through ``Prediction:\\n`` (matches finetuned test eval scoring)."""
    return (
        "Task: Predict whether Disorders of Lipid Metabolism will be present in the next visit.\n\n"
        f"Patient record:\n{narrative}\n\n"
        "Prediction:\n"
    )


def _collect_first_token_ids(tokenizer: AutoTokenizer, texts: List[str]) -> set[int]:
    ids: set[int] = set()
    for t in texts:
        enc = tokenizer.encode(t, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return ids


def _model_inference_device(module: torch.nn.Module) -> torch.device:
    dev = getattr(module, "device", None)
    if isinstance(dev, torch.device) and dev.type != "meta":
        return dev
    p = next(module.parameters(), None)
    if p is not None and p.device.type != "meta":
        return p.device
    return torch.device("cpu")


def _compute_val_rank_metrics(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    val_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Accuracy / recall / F1 plus AUC and AUPRC from Yes-vs-No logit softmax at the answer position."""
    if val_df.empty:
        raise ValueError("val_df is empty.")
    for col in ("narrative_current", "label_lipid_disorder"):
        if col not in val_df.columns:
            raise KeyError(f"val_df missing required column {col!r}")
    yes_ids = _collect_first_token_ids(tokenizer, ["Yes", " yes", "Yes ", " yes "])
    no_ids = _collect_first_token_ids(tokenizer, ["No", " no", "No ", " no "])
    if not yes_ids or not no_ids:
        raise RuntimeError("Failed to resolve Yes/No token ids for validation rank metrics.")
    dev = _model_inference_device(model)
    was_training = model.training
    model.eval()
    ys: List[int] = []
    preds: List[int] = []
    scores: List[float] = []
    try:
        for _, row in val_df.iterrows():
            true_label = int(row["label_lipid_disorder"])
            ptxt = _finetune_eval_prompt(str(row["narrative_current"]))
            toks = tokenizer(ptxt, return_tensors="pt")
            toks = {k: v.to(dev) for k, v in toks.items()}
            with torch.no_grad():
                logits = model(**toks).logits[:, -1, :]
            yes_logit = max(float(logits[0, idx].item()) for idx in yes_ids)
            no_logit = max(float(logits[0, idx].item()) for idx in no_ids)
            z = torch.tensor([yes_logit, no_logit], device=logits.device, dtype=logits.dtype)
            p_pair = torch.softmax(z, dim=0)
            prob_yes = float(p_pair[0].item())
            prob_no = float(p_pair[1].item())
            pred = 1 if prob_yes >= prob_no else 0
            ys.append(true_label)
            preds.append(pred)
            scores.append(prob_yes)
    finally:
        if was_training:
            model.train()
    y_true = np.asarray(ys, dtype=int)
    y_pred = np.asarray(preds, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    out = compute_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)
    out["n_val_scored"] = int(len(ys))
    out["metric_family"] = "yes_no_logit_softmax"
    return out


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
    evaluation_strategy: str = "epoch",
    eval_steps: int = 200,
    logging_steps: int = 20,
    save_strategy: str = "epoch",
    save_steps: int = 1000,
    answer_only_loss: bool = True,
    smoke: bool = False,
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "linear",
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.0,
    load_best_model_at_end: bool = False,
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
        model = _causal_lm_from_pretrained(model_name, smoke=smoke)
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
            model = _causal_lm_from_pretrained(model_name, smoke=smoke)
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

    def _tokenize_dataset(ds: Dataset, split_name: str) -> Dataset:
        debug_samples_remaining = 10
        stats = {
            "total": 0,
            "truncated_answer": 0,
            "supervised_sum": 0,
            "supervised_min": None,
            "supervised_max": 0,
        }

        def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
            out = tokenizer(
                batch["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
            labels = [ids.copy() for ids in out["input_ids"]]
            if answer_only_loss:
                prompts: List[str] = []
                for txt in batch["text"]:
                    marker = "\nPrediction:\n"
                    idx = txt.find(marker)
                    if idx >= 0:
                        prompts.append(txt[: idx + len(marker)])
                    else:
                        prompts.append(txt)
                prompt_tok = tokenizer(
                    prompts,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    add_special_tokens=False,
                )
                full_tok_no_special = tokenizer(
                    batch["text"],
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    add_special_tokens=False,
                )
                for i, prompt_ids in enumerate(prompt_tok["input_ids"]):
                    attention = out["attention_mask"][i]
                    seq_len = int(sum(attention))
                    if seq_len <= 0:
                        labels[i] = [-100] * len(labels[i])
                        continue

                    first_real = next((j for j, m in enumerate(attention) if m == 1), len(attention))
                    answer_len = max(len(full_tok_no_special["input_ids"][i]) - len(prompt_ids), 0)
                    answer_len = min(answer_len, seq_len)

                    if answer_len <= 0:
                        tail_preview = batch["text"][i][-220:].replace("\n", "\\n")
                        msg = (
                            f"Answer span truncated in split={split_name}, batch_index={i}, "
                            f"seq_len={seq_len}, prompt_len={len(prompt_ids)}, answer_len={answer_len}. "
                            f"text_tail={tail_preview!r}"
                        )
                        if smoke:
                            logger.warning(msg)
                            stats["truncated_answer"] += 1
                        else:
                            raise RuntimeError(msg)

                    # Supervise only the answer tail (Yes/No), independent of padding side.
                    supervise_start = first_real + (seq_len - answer_len)
                    for j in range(first_real, min(supervise_start, len(labels[i]))):
                        labels[i][j] = -100

                    # Always ignore padding positions regardless of collator behavior.
                    for j, m in enumerate(attention):
                        if m == 0:
                            labels[i][j] = -100

                    non_ignored_ids = [tid for tid, lab in zip(out["input_ids"][i], labels[i]) if lab != -100]
                    supervised_count = len(non_ignored_ids)
                    stats["total"] += 1
                    stats["supervised_sum"] += supervised_count
                    if stats["supervised_min"] is None:
                        stats["supervised_min"] = supervised_count
                    else:
                        stats["supervised_min"] = min(stats["supervised_min"], supervised_count)
                    stats["supervised_max"] = max(stats["supervised_max"], supervised_count)
                    nonlocal debug_samples_remaining
                    if debug_samples_remaining > 0:
                        debug_samples_remaining -= 1
                        logger.info(
                            "Mask debug sample: split=%s supervised_tokens=%d supervised_text=%r",
                            split_name,
                            supervised_count,
                            tokenizer.decode(non_ignored_ids, skip_special_tokens=False)[:160],
                        )
            out["labels"] = labels
            return out

        tokenized = ds.map(_tokenize, batched=True, remove_columns=["text"])
        if answer_only_loss:
            avg_supervised = (
                float(stats["supervised_sum"]) / float(stats["total"]) if stats["total"] > 0 else 0.0
            )
            logger.info(
                "Mask summary split=%s total_samples=%d truncated_answer_samples=%d "
                "avg_supervised_tokens=%.4f min_supervised_tokens=%s max_supervised_tokens=%d",
                split_name,
                stats["total"],
                stats["truncated_answer"],
                avg_supervised,
                stats["supervised_min"] if stats["supervised_min"] is not None else "n/a",
                stats["supervised_max"],
            )
        return tokenized

    tokenized_train = _tokenize_dataset(train_ds, "train")
    tokenized_val = _tokenize_dataset(val_ds, "val") if val_ds is not None else None
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
        eval_steps=1 if smoke else int(eval_steps),
        logging_steps=1 if smoke else int(logging_steps),
        save_steps=10 if smoke else int(save_steps),
        save_total_limit=2,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        bf16=False,
        fp16=use_cuda,
        report_to=[],
        warmup_ratio=float(warmup_ratio),
        lr_scheduler_type=str(lr_scheduler_type),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        weight_decay=float(weight_decay),
    )
    if load_best_model_at_end and not smoke and tokenized_val is not None:
        args_kwargs["load_best_model_at_end"] = True
        args_kwargs["metric_for_best_model"] = "eval_loss"
        args_kwargs["greater_is_better"] = False
    eval_mode = evaluation_strategy if tokenized_val is not None else "no"
    if eval_mode not in {"no", "steps", "epoch"}:
        raise ValueError(
            f"Unsupported evaluation_strategy={eval_mode!r}; expected one of: no, steps, epoch."
        )
    # Smoke: no mid-run checkpoints (avoids safetensors/meta-device errors under accelerate offload).
    effective_save_strategy = "no" if smoke else save_strategy
    if effective_save_strategy not in {"no", "steps", "epoch"}:
        raise ValueError(
            f"Unsupported save_strategy={effective_save_strategy!r}; expected one of: no, steps, epoch."
        )
    try:
        args = TrainingArguments(
            evaluation_strategy=eval_mode,
            save_strategy=effective_save_strategy,
            **args_kwargs,
        )
    except TypeError:
        # transformers>=5 renamed this argument.
        args = TrainingArguments(
            eval_strategy=eval_mode,
            save_strategy=effective_save_strategy,
            **args_kwargs,
        )
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

    if val_df is not None:
        try:
            rank_metrics = _compute_val_rank_metrics(model, tokenizer, val_df)
            logger.info(
                "Validation rank metrics (AUC/AUPRC on held-out val_ft rows): %s",
                {
                    k: rank_metrics[k]
                    for k in sorted(rank_metrics)
                    if k not in {"tp", "fp", "tn", "fn", "confusion_matrix"}
                },
            )
            logger.info(
                "Validation confusion counts: tp=%s fp=%s tn=%s fn=%s",
                rank_metrics.get("tp"),
                rank_metrics.get("fp"),
                rank_metrics.get("tn"),
                rank_metrics.get("fn"),
            )
            (output_path / "finetune_val_rank_metrics.json").write_text(
                json.dumps(rank_metrics, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Could not compute validation AUC/AUPRC (rank metrics): %s", exc)

    try:
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info("Saved fine-tuned Gemma LoRA adapter to %s", output_path)
    except NotImplementedError as exc:
        if smoke:
            logger.warning(
                "Smoke run: skipped final adapter save (%s). Training/eval metrics above are still valid.",
                exc,
            )
        else:
            raise
    return output_path
