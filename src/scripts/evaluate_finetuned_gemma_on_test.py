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

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import compute_metrics
from src.llm.logit_pair_scorer import (
    generate_one_restricted_token,
    hard_label_from_generation,
    forward_logit_score,
    resolve_yes_no_token_ids,
)
from src.utils.config_utils import load_config
from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger
from src.utils.random_utils import set_seed

logger = get_logger(__name__, log_file="data/outputs/finetuned_test_eval.log")


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

    yes_ids, no_ids, yes_no_ids = resolve_yes_no_token_ids(tokenizer)

    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Finetuned test eval"):
        sample_id = int(row["pair_id"])
        true_label = int(row["label_lipid_disorder"])
        score, toks, prompt_len = forward_logit_score(
            model, tokenizer, str(row["narrative_current"]), yes_ids, no_ids
        )
        toks = {k: v.to(model.device) for k, v in toks.items()}
        new_tokens = generate_one_restricted_token(model, tokenizer, toks, prompt_len, yes_no_ids)
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred_binary, parsed_prediction, parser_status = hard_label_from_generation(
            tokenizer,
            yes_ids,
            no_ids,
            new_tokens,
            generated,
            score.prob_yes,
            score.prob_no,
        )

        rows.append(
            {
                "pair_id": sample_id,
                "true_label": true_label,
                "pred_binary": pred_binary,
                "prob_yes": score.prob_yes,
                "prob_no": score.prob_no,
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
