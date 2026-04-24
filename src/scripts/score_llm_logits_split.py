"""Score train or test split with logits-based Yes/No probabilities (no free-text probs).

This script does **not** parse free-text ``Prediction:`` / ``Probability:`` model replies;
that lives in ``src.llm.output_parser`` (``parse_prediction``, ``parse_multitask_output``)
and is applied in ``src.llm.predictor.run_predictions`` when calling API-based LLMs.

Writes CSV under ``{paths.outputs}/fusion/llm_logit_scores_{split}.csv`` with:
pair_id, label_lipid_disorder, prob_yes, prob_no, margin_logit, pred_hard, split.

Usage:
  python -m src.scripts.score_llm_logits_split --config configs/default.yaml \\
      --overrides configs/fusion_default.yaml --split train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm.logit_pair_scorer import forward_logit_score, resolve_yes_no_token_ids
from src.utils.config_utils import load_config
from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger
from src.utils.random_utils import set_seed

logger = get_logger(__name__, log_file="data/outputs/fusion_llm_score.log")


def main(
    config_path: str,
    overrides: list[str] | None,
    split: str,
    input_csv: str | None,
    max_samples: int | None,
) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("--split must be train or test")
    cfg = load_config(config_path, overrides or [])
    set_seed(int(cfg.get("seed", 42)))

    processed = Path(cfg["paths"]["processed"])
    output_root = Path(cfg["paths"]["outputs"])
    fusion_cfg = cfg.get("fusion") or {}
    sub = str(fusion_cfg.get("output_subdir", "fusion"))
    out_dir = output_root if fusion_cfg.get("llm_scores_at_output_root") else output_root / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_csv:
        csv_path = Path(input_csv)
    else:
        csv_path = processed / ("train.csv" if split == "train" else "test.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    lim = max_samples
    if lim is None and fusion_cfg.get("llm_score_max_samples") is not None:
        lim = int(fusion_cfg["llm_score_max_samples"])
    if lim is not None and lim > 0:
        df = df.head(int(lim)).copy()

    llm_cfg = cfg.get("llm", {})
    adapter_dir = Path(llm_cfg.get("finetune_output_dir", "models/gemma_finetuned"))
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    base_model_name = llm_cfg.get("model_name") or llm_cfg.get("model")
    logger.info("Loading base model %s + adapter %s", base_model_name, adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    yes_ids, no_ids, _ = resolve_yes_no_token_ids(tokenizer)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"LLM logit score ({split})"):
        score, _, _ = forward_logit_score(
            model, tokenizer, str(row["narrative_current"]), yes_ids, no_ids
        )
        pred_hard = 1 if score.prob_yes >= score.prob_no else 0
        rows.append(
            {
                "pair_id": int(row["pair_id"]),
                "label_lipid_disorder": int(row["label_lipid_disorder"]),
                "prob_yes": score.prob_yes,
                "prob_no": score.prob_no,
                "margin_logit": score.margin_logit,
                "pred_hard": pred_hard,
                "split": split,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = out_dir / f"llm_logit_scores_{split}.csv"
    save_dataframe(out_df, out_path)
    logger.info("Wrote %d rows -> %s", len(out_df), out_path)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--overrides", nargs="*", default=[])
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--input-csv", default=None, help="Override default train.csv / test.csv path")
    p.add_argument("--max-samples", type=int, default=None, help="Cap rows (debug); else fusion.llm_score_max_samples")
    args = p.parse_args()
    main(args.config, list(args.overrides), args.split, args.input_csv, args.max_samples)
