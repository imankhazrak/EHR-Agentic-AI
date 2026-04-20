"""Build fusion dataset artifacts (structured + LLM scores, SVD, train/val indices).

Usage:
  python -m src.scripts.build_fusion_dataset --config configs/default.yaml \\
      --overrides configs/fusion_default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.fusion.dataset import FusionArtifactPaths, build_fusion_artifacts
from src.utils.config_utils import load_config
from src.utils.logging_utils import get_logger
from src.utils.random_utils import set_seed

logger = get_logger(__name__)


def main(config_path: str, overrides: list[str] | None) -> FusionArtifactPaths:
    cfg = load_config(config_path, overrides or [])
    set_seed(int(cfg.get("seed", 42)))
    fusion = cfg.get("fusion") or {}
    processed = Path(cfg["paths"]["processed"])
    out_root = Path(cfg["paths"]["outputs"])
    sub = str(fusion.get("output_subdir", "fusion"))
    scores_parent = out_root if fusion.get("llm_scores_at_output_root") else out_root / sub
    scores_tr = fusion.get("llm_scores_train_csv")
    scores_te = fusion.get("llm_scores_test_csv")
    if scores_tr:
        p_tr = Path(scores_tr)
    else:
        p_tr = scores_parent / "llm_logit_scores_train.csv"
    if scores_te:
        p_te = Path(scores_te)
    else:
        p_te = scores_parent / "llm_logit_scores_test.csv"

    use_oof = bool(fusion.get("use_oof_train_scores", False))
    oof_path = fusion.get("llm_scores_train_oof_csv")
    if oof_path:
        p_oof = Path(oof_path)
    elif use_oof:
        p_oof = scores_parent / "llm_logit_scores_train_oof.csv"
    else:
        p_oof = None

    art_root_cfg = fusion.get("artifact_root")
    if art_root_cfg:
        artifact_root = Path(art_root_cfg) if Path(art_root_cfg).is_absolute() else out_root / art_root_cfg
    else:
        artifact_root = None

    paths = build_fusion_artifacts(
        processed,
        out_root,
        p_tr,
        p_te,
        feature_type=str(fusion.get("feature_type", cfg.get("ml", {}).get("feature_type", "bag_of_codes"))),
        val_ratio=float(fusion.get("val_ratio", 0.15)),
        split_seed=int(fusion.get("split_seed", cfg.get("llm", {}).get("finetune", {}).get("split_seed", 42))),
        svd_dim=int(fusion.get("svd_dim", 256)),
        use_oof_train_scores=use_oof,
        oof_scores_path=p_oof,
        artifact_root=artifact_root,
    )
    logger.info("Fusion dataset ready: %s", paths.manifest)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()
    main(args.config, list(args.overrides))
