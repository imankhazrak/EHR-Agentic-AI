"""Stratified K-fold out-of-fold LLM logit scores on train.csv (frozen model, no test rows).

Each train row receives a score computed only when it lies in the validation fold of a CV split,
so train-side fusion can use scores that were not produced using a self-predictive shortcut.
Still uses the same frozen Gemma+LoRA forward (no fine-tuning on folds).

Writes CSV to ``{scores_parent}/llm_logit_scores_train_oof.csv`` (see fusion.llm_scores_at_output_root).

Usage:
  python -m src.scripts.score_llm_logits_train_oof --config configs/default.yaml \\
      --overrides configs/fusion_holdout_mimiciii.yaml --n-splits 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm.logit_pair_scorer import forward_logit_score, resolve_yes_no_token_ids
from src.utils.config_utils import load_config
from src.utils.io import save_dataframe
from src.utils.logging_utils import get_logger
from src.utils.random_utils import set_seed

logger = get_logger(__name__)


def main(
    config_path: str,
    overrides: list[str] | None,
    n_splits: int,
    max_samples: int | None,
) -> Path:
    cfg = load_config(config_path, overrides or [])
    set_seed(int(cfg.get("seed", 42)))
    processed = Path(cfg["paths"]["processed"])
    output_root = Path(cfg["paths"]["outputs"])
    fusion_cfg = cfg.get("fusion") or {}
    sub = str(fusion_cfg.get("output_subdir", "fusion"))
    parent = output_root if fusion_cfg.get("llm_scores_at_output_root") else output_root / sub
    parent.mkdir(parents=True, exist_ok=True)

    train_path = processed / "train.csv"
    train_df = pd.read_csv(train_path)
    lim = max_samples
    if lim is None and fusion_cfg.get("llm_score_max_samples") is not None:
        lim = int(fusion_cfg["llm_score_max_samples"])
    if lim is not None and lim > 0:
        train_df = train_df.head(int(lim)).copy()

    y = train_df["label_lipid_disorder"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.get("seed", 42)))

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

    by_pos: dict[int, dict] = {}
    for fold_id, (_, val_idx) in enumerate(skf.split(np.zeros(len(train_df)), y)):
        sub_df = train_df.iloc[val_idx].reset_index(drop=True)
        for _, row in tqdm(
            sub_df.iterrows(),
            total=len(sub_df),
            desc=f"OOF fold {fold_id + 1}/{n_splits}",
        ):
            score, _, _ = forward_logit_score(
                model, tokenizer, str(row["narrative_current"]), yes_ids, no_ids
            )
            pid = int(row["pair_id"])
            pred_hard = 1 if score.prob_yes >= score.prob_no else 0
            by_pos[pid] = {
                "pair_id": pid,
                "label_lipid_disorder": int(row["label_lipid_disorder"]),
                "prob_yes": score.prob_yes,
                "prob_no": score.prob_no,
                "margin_logit": score.margin_logit,
                "pred_hard": pred_hard,
                "oof_fold": int(fold_id),
                "split": "train_oof",
            }

    if len(by_pos) != len(train_df):
        missing = len(train_df) - len(by_pos)
        raise RuntimeError(f"OOF coverage incomplete: expected {len(train_df)} rows, got {len(by_pos)} ({missing} missing).")

    out_rows = [by_pos[int(pid)] for pid in train_df["pair_id"].tolist()]
    out_df = pd.DataFrame(out_rows)
    out_path = parent / "llm_logit_scores_train_oof.csv"
    save_dataframe(out_df, out_path)
    logger.info("Wrote OOF train scores -> %s", out_path)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--overrides", nargs="*", default=[])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()
    main(args.config, list(args.overrides), args.n_splits, args.max_samples)
