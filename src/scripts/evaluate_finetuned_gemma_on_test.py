"""Evaluate finetuned Gemma LoRA adapter on held-out test set.

The first token after ``Prediction:\\n`` is constrained to tokenizer ids for ``Yes`` / ``No``
(plus common leading-space variants), matching answer-only training. Probabilities use a
two-way softmax over the best Yes vs best No logit at that position.

Outputs:
  - data/outputs/mimiciii_gemma/llm_finetuned_test_results.csv
  - data/outputs/mimiciii_gemma/llm_finetuned_test_metrics.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from src.evaluation.metrics import compute_metrics
from src.llm.output_parser import parse_prediction
from src.utils.config_utils import load_config
from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger
from src.utils.random_utils import set_seed

logger = get_logger(__name__, log_file="data/outputs/finetuned_test_eval.log")


def _prompt(narrative: str) -> str:
    return (
        "Task: Predict whether Disorders of Lipid Metabolism will be present in the next visit.\n\n"
        f"Patient record:\n{narrative}\n\n"
        "Prediction:\n"
    )


def _collect_token_ids(tokenizer: AutoTokenizer, texts: list[str]) -> set[int]:
    ids: set[int] = set()
    for t in texts:
        enc = tokenizer.encode(t, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return ids


class _FirstStepRestrictTokensLogitsProcessor(LogitsProcessor):
    """Allow only *allowed_token_ids* as the first generated token (then leave logits unchanged)."""

    def __init__(self, prompt_length: int, allowed_token_ids: list[int]) -> None:
        self.prompt_length = int(prompt_length)
        self.allowed_token_ids = list(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] != self.prompt_length:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask


def main(config_path: str = "configs/default.yaml", overrides: list | None = None) -> None:
    cfg = load_config(config_path, overrides)
    set_seed(cfg.get("seed", 42))

    processed_dir = Path(cfg["paths"]["processed"])
    output_dir = Path(cfg["paths"]["outputs"])
    llm_cfg = cfg.get("llm", {})
    adapter_dir = Path(llm_cfg.get("finetune_output_dir", "models/gemma_finetuned"))
    test_df = pd.read_csv(processed_dir / "test.csv")
    lim = llm_cfg.get("max_test_samples")
    if lim is not None and lim > 0:
        test_df = test_df.head(int(lim)).copy()

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Loading base model from %s", llm_cfg.get("model_name") or llm_cfg.get("model"))
    base_model_name = llm_cfg.get("model_name") or llm_cfg.get("model")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    yes_ids = _collect_token_ids(tokenizer, ["Yes", " yes", "Yes ", " yes "])
    no_ids = _collect_token_ids(tokenizer, ["No", " no", "No ", " no "])
    if not yes_ids or not no_ids:
        raise RuntimeError("Failed to resolve Yes/No token ids for scoring.")
    yes_no_ids = sorted(yes_ids | no_ids)

    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Finetuned test eval"):
        sample_id = int(row["pair_id"])
        true_label = int(row["label_lipid_disorder"])
        ptxt = _prompt(str(row["narrative_current"]))
        toks = tokenizer(ptxt, return_tensors="pt")
        toks = {k: v.to(model.device) for k, v in toks.items()}
        prompt_len = int(toks["input_ids"].shape[1])

        with torch.no_grad():
            logits = model(**toks).logits[:, -1, :]
            yes_logit = max(float(logits[0, idx].item()) for idx in yes_ids)
            no_logit = max(float(logits[0, idx].item()) for idx in no_ids)
            z = torch.tensor([yes_logit, no_logit], device=logits.device, dtype=logits.dtype)
            p_pair = torch.softmax(z, dim=0)
            prob_yes = float(p_pair[0].item())
            prob_no = float(p_pair[1].item())

            proc = LogitsProcessorList(
                [_FirstStepRestrictTokensLogitsProcessor(prompt_len, yes_no_ids)]
            )
            gen = model.generate(
                **toks,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=proc,
            )

        new_tokens = gen[0, prompt_len:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parsed = parse_prediction(generated)

        first_id = int(new_tokens[0].item()) if new_tokens.numel() > 0 else None
        if first_id is not None and first_id in yes_ids:
            pred_binary = 1
            parsed_prediction = "Yes"
            parser_status = "first_token_yes"
        elif first_id is not None and first_id in no_ids:
            pred_binary = 0
            parsed_prediction = "No"
            parser_status = "first_token_no"
        elif parsed["prediction"] == "Yes":
            pred_binary = 1
            parsed_prediction = "Yes"
            parser_status = parsed["parser_status"]
        elif parsed["prediction"] == "No":
            pred_binary = 0
            parsed_prediction = "No"
            parser_status = parsed["parser_status"]
        else:
            pred_binary = 1 if prob_yes >= prob_no else 0
            parsed_prediction = "Yes" if pred_binary == 1 else "No"
            parser_status = "logits_fallback"

        rows.append(
            {
                "pair_id": sample_id,
                "true_label": true_label,
                "pred_binary": pred_binary,
                "prob_yes": prob_yes,
                "prob_no": prob_no,
                "generated_text": generated,
                "parsed_prediction": parsed_prediction,
                "parser_status": parser_status,
                "model_name": str(adapter_dir),
                "prompt_mode": "finetuned_test",
            }
        )

    pred_df = pd.DataFrame(rows)
    y_true = pred_df["true_label"].to_numpy()
    y_pred = pred_df["pred_binary"].to_numpy()
    y_score = pred_df["prob_yes"].to_numpy()
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)
    metrics["mode"] = "finetuned_test"
    metrics["model"] = str(adapter_dir)
    metrics["n_total"] = int(len(pred_df))
    metrics["n_valid"] = int(len(pred_df))
    metrics["n_unparseable"] = int((pred_df["parser_status"] == "logits_fallback").sum())

    save_dataframe(pred_df, output_dir / "llm_finetuned_test_results.csv")
    save_json(metrics, output_dir / "llm_finetuned_test_metrics.json")
    logger.info("Finetuned test metrics: %s", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    main(args.config, args.overrides)
