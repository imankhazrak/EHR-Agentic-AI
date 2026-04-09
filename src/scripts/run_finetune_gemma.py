"""Script: Fine-tune Gemma with LoRA on narrative_current -> Yes/No labels.

Usage:
    python -m src.scripts.run_finetune_gemma --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.llm.finetune_gemma import finetune_gemma_lora
from src.utils.io import save_dataframe, save_json
from src.utils.config_utils import load_config
from src.utils.logging_utils import get_logger
from src.utils.random_utils import set_seed

logger = get_logger(__name__, log_file="data/outputs/finetune_gemma.log")


def _prepare_finetune_splits(
    processed_dir: Path,
    val_ratio: float,
    split_seed: int,
    force_rebuild: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ft_path = processed_dir / "train_ft.csv"
    val_ft_path = processed_dir / "val_ft.csv"
    split_summary_path = processed_dir / "finetune_split_summary.json"

    if train_ft_path.exists() and val_ft_path.exists() and not force_rebuild:
        train_ft_df = pd.read_csv(train_ft_path)
        val_ft_df = pd.read_csv(val_ft_path)
        logger.info("Reusing existing fine-tune splits: %s, %s", train_ft_path, val_ft_path)
        return train_ft_df, val_ft_df

    train_path = processed_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Run preprocessing first to generate train.csv/test.csv."
        )

    train_df = pd.read_csv(train_path)
    if "label_lipid_disorder" not in train_df.columns:
        raise ValueError("Expected column 'label_lipid_disorder' in train.csv for stratified split.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}.")

    train_ft_df, val_ft_df = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=split_seed,
        stratify=train_df["label_lipid_disorder"],
    )
    train_ft_df = train_ft_df.reset_index(drop=True).copy()
    val_ft_df = val_ft_df.reset_index(drop=True).copy()
    train_ft_df["split"] = "train_ft"
    val_ft_df["split"] = "val_ft"

    save_dataframe(train_ft_df, train_ft_path)
    save_dataframe(val_ft_df, val_ft_path)

    summary = {
        "source_train_rows": int(len(train_df)),
        "val_ratio": float(val_ratio),
        "split_seed": int(split_seed),
        "train_ft_rows": int(len(train_ft_df)),
        "val_ft_rows": int(len(val_ft_df)),
        "train_ft_prevalence": round(float(train_ft_df["label_lipid_disorder"].mean()), 4),
        "val_ft_prevalence": round(float(val_ft_df["label_lipid_disorder"].mean()), 4),
    }
    save_json(summary, split_summary_path)
    logger.info("Prepared fine-tune train/val split: %s", summary)
    return train_ft_df, val_ft_df


def main(
    config_path: str = "configs/default.yaml",
    overrides: list | None = None,
    smoke: bool = False,
    force_rebuild_split: bool = False,
) -> None:
    cfg = load_config(config_path, overrides)
    set_seed(cfg.get("seed", 42))

    if cfg.get("llm", {}).get("mode", "inference") != "finetune":
        logger.warning("llm.mode is not 'finetune'; proceeding anyway for explicit script run.")

    processed_dir = Path(cfg["paths"]["processed"])
    llm_cfg = cfg.get("llm", {})
    model_name = llm_cfg.get("model_name") or llm_cfg.get("model")
    output_dir = llm_cfg.get("finetune_output_dir", "models/gemma_finetuned")
    train_cfg = llm_cfg.get("finetune", {})
    val_ratio = float(train_cfg.get("val_ratio_within_train", 0.15))
    split_seed = int(train_cfg.get("split_seed", cfg.get("seed", 42)))
    use_smoke = bool(smoke or train_cfg.get("smoke", False))
    train_ft_df, val_ft_df = _prepare_finetune_splits(
        processed_dir=processed_dir,
        val_ratio=val_ratio,
        split_seed=split_seed,
        force_rebuild=bool(force_rebuild_split or train_cfg.get("force_rebuild_split", False)),
    )

    if use_smoke:
        logger.info("Running fine-tune in smoke mode with reduced samples/steps.")

    finetune_gemma_lora(
        train_df=train_ft_df,
        val_df=val_ft_df,
        model_name=model_name,
        output_dir=output_dir,
        max_length=int(train_cfg.get("max_length", 1024)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        batch_size=int(train_cfg.get("batch_size", 2)),
        num_train_epochs=int(train_cfg.get("num_train_epochs", 1)),
        lora_r=int(train_cfg.get("lora_r", 8)),
        lora_alpha=int(train_cfg.get("lora_alpha", 16)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=train_cfg.get("lora_target_modules"),
        max_train_samples=int(train_cfg.get("smoke_max_train_samples", 64)) if use_smoke else None,
        max_eval_samples=int(train_cfg.get("smoke_max_eval_samples", 32)) if use_smoke else None,
        max_steps=int(train_cfg.get("smoke_max_steps", 3)) if use_smoke else None,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        force_fp32=bool(train_cfg.get("force_fp32", False)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        smoke=use_smoke,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke fine-tune.")
    parser.add_argument(
        "--force-rebuild-split",
        action="store_true",
        help="Regenerate train_ft/val_ft from processed/train.csv even if they already exist.",
    )
    args = parser.parse_args()
    main(args.config, args.overrides, smoke=args.smoke, force_rebuild_split=args.force_rebuild_split)
