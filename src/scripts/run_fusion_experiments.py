"""End-to-end fusion experiments: optional LLM scoring, build dataset, train/eval models.

Usage:
  python -m src.scripts.run_fusion_experiments --config configs/default.yaml \\
      --overrides configs/fusion_default.yaml

  # Skip LLM scoring if CSVs already exist:
  python -m src.scripts.run_fusion_experiments --config configs/default.yaml \\
      --overrides configs/fusion_default.yaml --skip-llm-score
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.ml.fusion.dataset import FusionArtifactPaths
from src.ml.fusion.train_eval import run_fusion_experiments
from src.scripts.build_fusion_dataset import main as build_main
from src.utils.config_utils import load_config
from src.utils.logging_utils import get_logger
from src.utils.random_utils import set_seed

logger = get_logger(__name__)


def _maybe_run_llm_scores(config_path: str, overrides: list[str], cfg: dict, skip: bool) -> None:
    if skip:
        logger.info("--skip-llm-score: not invoking score_llm_logits_split")
        return
    fusion = cfg.get("fusion") or {}
    if not fusion.get("run_llm_score_subprocess", True):
        return
    out_root = Path(cfg["paths"]["outputs"])
    sub = str(fusion.get("output_subdir", "fusion"))
    parent = out_root if fusion.get("llm_scores_at_output_root") else out_root / sub
    tr = parent / "llm_logit_scores_train.csv"
    te = parent / "llm_logit_scores_test.csv"
    if tr.exists() and te.exists():
        logger.info("LLM score CSVs already exist; skipping subprocess.")
        return
    base = [sys.executable, "-m", "src.scripts.score_llm_logits_split", "--config", config_path]
    for o in overrides:
        base.extend(["--overrides", o])
    cmd_tr = base + ["--split", "train"]
    cmd_te = base + ["--split", "test"]
    logger.info("Running: %s", " ".join(cmd_tr))
    subprocess.run(cmd_tr, check=True)
    logger.info("Running: %s", " ".join(cmd_te))
    subprocess.run(cmd_te, check=True)


def main(
    config_path: str,
    overrides: list[str] | None,
    skip_llm_score: bool,
    skip_build: bool,
) -> None:
    cfg = load_config(config_path, overrides or [])
    set_seed(int(cfg.get("seed", 42)))
    _maybe_run_llm_scores(config_path, overrides or [], cfg, skip_llm_score)
    if not skip_build:
        paths: FusionArtifactPaths = build_main(config_path, overrides or [])
    else:
        fusion = cfg.get("fusion") or {}
        out_root = Path(cfg["paths"]["outputs"])
        sub = str(fusion.get("output_subdir", "fusion"))
        art_root_cfg = fusion.get("artifact_root")
        if art_root_cfg:
            root = Path(art_root_cfg) if Path(art_root_cfg).is_absolute() else out_root / art_root_cfg
        else:
            root = out_root / sub
        man = root / "fusion_manifest.json"
        if not man.exists():
            raise FileNotFoundError(f"--skip-build requires existing fusion bundle at {root}")
        art = root / "artifacts"
        paths = FusionArtifactPaths(
            root=root,
            manifest=man,
            feature_stage=art / "feature_stage.joblib",
            x_struct_train=art / "X_struct_train.npz",
            x_struct_test=art / "X_struct_test.npz",
            x_train_svd=art / "X_train_svd.npy",
            x_test_svd=art / "X_test_svd.npy",
            llm_train=art / "llm_train.npy",
            llm_test=art / "llm_test.npy",
            y_train=art / "y_train.npy",
            y_test=art / "y_test.npy",
            meta_csv=root / "fusion_row_meta.csv",
            idx_train_fit=art / "idx_train_fit.npy",
            idx_val_tune=art / "idx_val_tune.npy",
            train_pair_ids=art / "train_pair_ids.npy",
            test_pair_ids=art / "test_pair_ids.npy",
        )
    run_fusion_experiments(paths, cfg, out_dir=paths.root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--skip-llm-score", action="store_true")
    ap.add_argument("--skip-build", action="store_true", help="Reuse existing fusion/ artifacts")
    args = ap.parse_args()
    main(args.config, list(args.overrides), args.skip_llm_score, args.skip_build)
