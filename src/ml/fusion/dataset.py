"""Build leakage-safe fusion artifacts: structured sparse features, SVD, LLM scores, train/val indices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from src.ml.feature_builder import build_bag_of_codes, build_tfidf
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FusionArtifactPaths:
    root: Path
    manifest: Path
    feature_stage: Path
    x_struct_train: Path
    x_struct_test: Path
    x_train_svd: Path
    x_test_svd: Path
    llm_train: Path
    llm_test: Path
    y_train: Path
    y_test: Path
    meta_csv: Path
    idx_train_fit: Path
    idx_val_tune: Path
    train_pair_ids: Path
    test_pair_ids: Path


def _merge_llm_scores(df: pd.DataFrame, scores: pd.DataFrame, split_name: str) -> pd.DataFrame:
    need = {"pair_id", "prob_yes", "prob_no", "margin_logit"}
    miss = need - set(scores.columns)
    if miss:
        raise ValueError(f"LLM scores CSV missing columns: {miss}")
    sub = scores[list(need | ({"pred_hard"} & set(scores.columns)))].copy()
    merged = df.merge(sub, on="pair_id", how="inner")
    if len(merged) != len(df):
        logger.warning(
            "%s: merged %d/%d rows after inner join on pair_id (dropped %d)",
            split_name,
            len(merged),
            len(df),
            len(df) - len(merged),
        )
    if merged.empty:
        raise RuntimeError(f"No rows left after merging LLM scores for {split_name}.")
    return merged


def _safe_svd_components(n_samples: int, n_features: int, requested: int) -> int:
    """TruncatedSVD n_components must be < min(n_samples, n_features) in common sklearn versions."""
    upper = min(n_samples - 1, n_features) if n_samples > 1 and n_features > 0 else 1
    upper = max(upper, 1)
    return max(2, min(int(requested), upper))


def build_fusion_artifacts(
    processed_dir: Path,
    output_dir: Path,
    scores_train_path: Path,
    scores_test_path: Path,
    *,
    feature_type: str = "bag_of_codes",
    val_ratio: float = 0.15,
    split_seed: int = 42,
    svd_dim: int = 256,
    use_oof_train_scores: bool = False,
    oof_scores_path: Path | None = None,
    artifact_root: Path | None = None,
) -> FusionArtifactPaths:
    """Fit vectorizer on train-only rows, SVD on full train sparse design, stratified train_fit/val_tune indices.

    If *artifact_root* is set, fusion matrices and manifest are written there; otherwise ``{output_dir}/fusion``.
    """
    root = Path(artifact_root) if artifact_root is not None else Path(output_dir) / "fusion"
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    st_path = oof_scores_path if (use_oof_train_scores and oof_scores_path is not None) else scores_train_path
    if not st_path.exists():
        raise FileNotFoundError(f"Train LLM scores not found: {st_path}")
    if not scores_test_path.exists():
        raise FileNotFoundError(f"Test LLM scores not found: {scores_test_path}")

    scores_tr = pd.read_csv(st_path)
    scores_te = pd.read_csv(scores_test_path)

    train_m = _merge_llm_scores(train_df, scores_tr, "train")
    test_m = _merge_llm_scores(test_df, scores_te, "test")

    if feature_type == "bag_of_codes":
        X_tr_sp, X_te_sp, vec = build_bag_of_codes(train_m, test_m)
        featurizer = vec
    elif feature_type == "tfidf":
        X_tr_sp, X_te_sp, vec = build_tfidf(train_m, test_m)
        featurizer = vec
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    y_train = train_m["label_lipid_disorder"].to_numpy(dtype=np.int64)
    y_test = test_m["label_lipid_disorder"].to_numpy(dtype=np.int64)

    n_comp = _safe_svd_components(X_tr_sp.shape[0], X_tr_sp.shape[1], svd_dim)
    svd = TruncatedSVD(n_components=n_comp, random_state=split_seed)
    svd.fit(X_tr_sp)
    X_tr_svd = svd.transform(X_tr_sp).astype(np.float32)
    X_te_svd = svd.transform(X_te_sp).astype(np.float32)

    idx_all = np.arange(len(train_m))
    idx_fit, idx_val = train_test_split(
        idx_all,
        test_size=val_ratio,
        random_state=split_seed,
        stratify=y_train,
    )
    mask_fit = np.zeros(len(train_m), dtype=bool)
    mask_fit[idx_fit] = True

    llm_cols = ["prob_yes", "prob_no", "margin_logit"]
    if "pred_hard" in train_m.columns:
        llm_cols.append("pred_hard")
    Z_tr = train_m[llm_cols].to_numpy(dtype=np.float32)
    Z_te = test_m[llm_cols].to_numpy(dtype=np.float32)

    meta_rows: List[Dict[str, Any]] = []
    for i in range(len(train_m)):
        spl = "train_fit" if mask_fit[i] else "val_tune"
        r = train_m.iloc[i]
        meta_rows.append(
            {
                "pair_id": int(r["pair_id"]),
                "y": int(r["label_lipid_disorder"]),
                "fusion_split": spl,
                "prob_yes": float(r["prob_yes"]),
                "prob_no": float(r["prob_no"]),
                "margin_logit": float(r["margin_logit"]),
                "pred_hard_llm": int(r.get("pred_hard", (r["prob_yes"] >= r["prob_no"]))),
            }
        )
    for i in range(len(test_m)):
        r = test_m.iloc[i]
        meta_rows.append(
            {
                "pair_id": int(r["pair_id"]),
                "y": int(r["label_lipid_disorder"]),
                "fusion_split": "test",
                "prob_yes": float(r["prob_yes"]),
                "prob_no": float(r["prob_no"]),
                "margin_logit": float(r["margin_logit"]),
                "pred_hard_llm": int(r.get("pred_hard", (r["prob_yes"] >= r["prob_no"]))),
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    meta_path = root / "fusion_row_meta.csv"
    meta_df.to_csv(meta_path, index=False)

    sparse.save_npz(art / "X_struct_train.npz", X_tr_sp.tocsr())
    sparse.save_npz(art / "X_struct_test.npz", X_te_sp.tocsr())
    np.save(art / "X_train_svd.npy", X_tr_svd)
    np.save(art / "X_test_svd.npy", X_te_svd)
    np.save(art / "llm_train.npy", Z_tr)
    np.save(art / "llm_test.npy", Z_te)
    y_train_path = art / "y_train.npy"
    y_test_path = art / "y_test.npy"
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    np.save(art / "idx_train_fit.npy", idx_fit)
    np.save(art / "idx_val_tune.npy", idx_val)
    np.save(art / "train_pair_ids.npy", train_m["pair_id"].to_numpy())
    np.save(art / "test_pair_ids.npy", test_m["pair_id"].to_numpy())

    stage = {"feature_type": feature_type, "featurizer": featurizer, "svd": svd, "svd_dim_requested": svd_dim, "svd_dim_used": n_comp}
    feat_path = art / "feature_stage.joblib"
    joblib.dump(stage, feat_path)

    manifest: Dict[str, Any] = {
        "processed_dir": str(processed_dir),
        "scores_train": str(st_path),
        "scores_test": str(scores_test_path),
        "use_oof_train_scores": use_oof_train_scores,
        "feature_type": feature_type,
        "val_ratio": val_ratio,
        "split_seed": split_seed,
        "n_train": int(len(train_m)),
        "n_test": int(len(test_m)),
        "n_struct_features": int(X_tr_sp.shape[1]),
        "svd_dim_used": int(n_comp),
        "n_train_fit": int(len(idx_fit)),
        "n_val_tune": int(len(idx_val)),
    }
    man_path = root / "fusion_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Fusion artifacts written under %s", root)

    return FusionArtifactPaths(
        root=root,
        manifest=man_path,
        feature_stage=feat_path,
        x_struct_train=art / "X_struct_train.npz",
        x_struct_test=art / "X_struct_test.npz",
        x_train_svd=art / "X_train_svd.npy",
        x_test_svd=art / "X_test_svd.npy",
        llm_train=art / "llm_train.npy",
        llm_test=art / "llm_test.npy",
        y_train=y_train_path,
        y_test=y_test_path,
        meta_csv=meta_path,
        idx_train_fit=art / "idx_train_fit.npy",
        idx_val_tune=art / "idx_val_tune.npy",
        train_pair_ids=art / "train_pair_ids.npy",
        test_pair_ids=art / "test_pair_ids.npy",
    )


def load_fusion_bundle(paths: FusionArtifactPaths) -> Dict[str, Any]:
    """Load matrices and indices for training scripts."""
    stage = joblib.load(paths.feature_stage)
    X_tr = sparse.load_npz(paths.x_struct_train).tocsr()
    X_te = sparse.load_npz(paths.x_struct_test).tocsr()
    return {
        "feature_stage": stage,
        "X_struct_train": X_tr,
        "X_struct_test": X_te,
        "X_train_svd": np.load(paths.x_train_svd),
        "X_test_svd": np.load(paths.x_test_svd),
        "llm_train": np.load(paths.llm_train),
        "llm_test": np.load(paths.llm_test),
        "y_train": np.load(paths.y_train),
        "y_test": np.load(paths.y_test),
        "idx_train_fit": np.load(paths.idx_train_fit),
        "idx_val_tune": np.load(paths.idx_val_tune),
        "train_pair_ids": np.load(paths.train_pair_ids),
        "test_pair_ids": np.load(paths.test_pair_ids),
    }
