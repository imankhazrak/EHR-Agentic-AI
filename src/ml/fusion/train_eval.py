"""Train and evaluate fusion models (LR, MLP, Hybrid) across feature regimes."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics
from src.ml.fusion.dataset import FusionArtifactPaths, load_fusion_bundle
from src.ml.fusion.models_hybrid import FusionHybridCNNTransformer
from src.ml.fusion.models_mlp import FusionMLP
from src.ml.fusion.thresholds import pick_threshold
from src.utils.io import save_dataframe, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

FeatureRegime = Literal["structured_only", "llm_only", "fused"]


def _hstack_struct_llm(X_sp: csr_matrix, Z: np.ndarray) -> csr_matrix:
    Z_csr = csr_matrix(Z.astype(np.float64))
    return hstack([X_sp.tocsr(), Z_csr], format="csr")


def _metrics_block(
    y_true: np.ndarray,
    y_pred_default: np.ndarray,
    y_score: np.ndarray,
    y_pred_tuned: np.ndarray,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    base = compute_metrics(y_true, y_pred_default, y_score=y_score)
    out: Dict[str, Any] = {**{k: v for k, v in base.items()}, **extra}
    out["pred_positive_rate_default"] = round(100.0 * float(np.mean(y_pred_default)), 4)
    mt = compute_metrics(y_true, y_pred_tuned, y_score=y_score)
    out["threshold_tuned_metrics"] = {
        k: v
        for k, v in mt.items()
        if k in ("accuracy", "sensitivity", "specificity", "precision", "recall", "f1", "tp", "fp", "tn", "fn")
    }
    out["pred_positive_rate_tuned"] = round(100.0 * float(np.mean(y_pred_tuned)), 4)
    return out


def _lr_solver(n_features: int) -> str:
    if n_features < 100:
        return "liblinear"
    if n_features >= 5000:
        return "saga"
    return "lbfgs"


def _train_torch_classifier(
    model: nn.Module,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    patience: int = 5,
) -> Tuple[nn.Module, int]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    def _loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    tr_loader = _loader(X_fit, y_fit, True)
    va_loader = _loader(X_val, y_val, False)
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_ep = 1
    stall = 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += float(loss_fn(model(xb), yb).item()) * len(xb)
                n += len(xb)
        val_loss /= max(n, 1)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_ep = ep
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_ep


def _refit_torch_full(
    model_ctor: Callable[[], nn.Module],
    X_full: np.ndarray,
    y_full: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
) -> nn.Module:
    model = model_ctor()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(X_full).float(), torch.from_numpy(y_full).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def _torch_predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outs: List[float] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            outs.extend(p.tolist())
    return np.asarray(outs, dtype=np.float64)


def _dense_block(reg: FeatureRegime, X_svd: np.ndarray, Z: np.ndarray) -> np.ndarray:
    if reg == "structured_only":
        return X_svd.astype(np.float32)
    if reg == "llm_only":
        return Z.astype(np.float32)
    return np.hstack([X_svd, Z]).astype(np.float32)


def _sparse_block(reg: FeatureRegime, X_sp: csr_matrix, Z: np.ndarray) -> csr_matrix:
    if reg == "structured_only":
        return X_sp.tocsr()
    if reg == "llm_only":
        return csr_matrix(Z.astype(np.float64))
    return _hstack_struct_llm(X_sp, Z)


def _save_pred_csv(path: Path, pair_ids: np.ndarray, y_true: np.ndarray, proba: np.ndarray, pred05: np.ndarray, pred_t: np.ndarray, thr: float) -> None:
    save_dataframe(
        pd.DataFrame(
            {
                "pair_id": pair_ids,
                "y_true": y_true,
                "score": proba,
                "pred_default_0p5": pred05,
                "pred_val_threshold": pred_t,
                "val_threshold": thr,
            }
        ),
        path,
    )


def _load_comparison_baselines(cfg: Dict[str, Any], fusion: Dict[str, Any], out_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ml_path = fusion.get("compare_ml_results_csv")
    if not ml_path:
        ml_path = out_root / "ml_results_fully_supervised.csv"
    if Path(ml_path).exists():
        mdf = pd.read_csv(ml_path)
        for _, r in mdf.iterrows():
            rows.append(
                {
                    "source": "ml_baseline",
                    "model": r.get("model", r.get("setting", "")),
                    "accuracy": r.get("accuracy"),
                    "sensitivity": r.get("sensitivity"),
                    "f1": r.get("f1"),
                    "auc": r.get("auc"),
                    "auprc": r.get("auprc"),
                }
            )
    llm_path = fusion.get("compare_llm_test_results_csv")
    if not llm_path:
        llm_path = out_root / "llm_finetuned_test_results.csv"
    if Path(llm_path).exists():
        df = pd.read_csv(llm_path)
        if {"true_label", "pred_binary", "prob_yes"}.issubset(df.columns):
            m = compute_metrics(
                df["true_label"].to_numpy(),
                df["pred_binary"].to_numpy(),
                y_score=df["prob_yes"].to_numpy(),
            )
            rows.append(
                {
                    "source": "llm_hard_decision",
                    "model": "finetuned_gemma_test_csv",
                    "accuracy": m.get("accuracy"),
                    "sensitivity": m.get("sensitivity"),
                    "f1": m.get("f1"),
                    "auc": m.get("auc"),
                    "auprc": m.get("auprc"),
                }
            )
            prob_pred = (df["prob_yes"].to_numpy() >= 0.5).astype(int)
            m2 = compute_metrics(df["true_label"].to_numpy(), prob_pred, y_score=df["prob_yes"].to_numpy())
            rows.append(
                {
                    "source": "llm_prob_threshold_0p5",
                    "model": "finetuned_gemma_prob",
                    "accuracy": m2.get("accuracy"),
                    "sensitivity": m2.get("sensitivity"),
                    "f1": m2.get("f1"),
                    "auc": m2.get("auc"),
                    "auprc": m2.get("auprc"),
                }
            )
    return rows


def run_fusion_experiments(paths: FusionArtifactPaths, cfg: Dict[str, Any], out_dir: Path | None = None) -> pd.DataFrame:
    fusion = cfg.get("fusion") or {}
    out_dir = out_dir or paths.root
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    bundle = load_fusion_bundle(paths)
    X_tr_sp: csr_matrix = bundle["X_struct_train"]
    X_te_sp: csr_matrix = bundle["X_struct_test"]
    X_tr_svd: np.ndarray = bundle["X_train_svd"]
    X_te_svd: np.ndarray = bundle["X_test_svd"]
    Z_tr: np.ndarray = bundle["llm_train"]
    Z_te: np.ndarray = bundle["llm_test"]
    y_train: np.ndarray = bundle["y_train"]
    y_test: np.ndarray = bundle["y_test"]
    idx_fit: np.ndarray = bundle["idx_train_fit"]
    idx_val: np.ndarray = bundle["idx_val_tune"]
    test_pair_ids: np.ndarray = bundle["test_pair_ids"]

    lr_cfg = fusion.get("logistic_regression", {})
    lr_c = float(lr_cfg.get("C", 1.0))
    lr_max_iter = int(lr_cfg.get("max_iter", 500))
    lr_class_weight = lr_cfg.get("class_weight", "balanced")

    mlp_cfg = fusion.get("mlp", {})
    mlp_hidden = tuple(mlp_cfg.get("hidden", [128, 64]))
    mlp_dropout = float(mlp_cfg.get("dropout", 0.2))
    mlp_epochs = int(mlp_cfg.get("epochs", 40))
    mlp_lr = float(mlp_cfg.get("lr", 1e-3))
    mlp_wd = float(mlp_cfg.get("weight_decay", 1e-4))
    mlp_batch = int(mlp_cfg.get("batch_size", 256))

    hy_cfg = fusion.get("hybrid", {})
    hy_tokens = int(hy_cfg.get("n_tokens", 8))
    hy_d_model = int(hy_cfg.get("d_model", 128))
    hy_layers = int(hy_cfg.get("num_layers", 2))
    hy_ff = int(hy_cfg.get("dim_feedforward", 256))
    hy_drop = float(hy_cfg.get("dropout", 0.1))
    hy_epochs = int(hy_cfg.get("epochs", 35))
    hy_lr = float(hy_cfg.get("lr", 1e-3))
    hy_wd = float(hy_cfg.get("weight_decay", 1e-4))
    hy_batch = int(hy_cfg.get("batch_size", 256))
    refit_nn = bool(fusion.get("refit_full_train_nn", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thr_mode = str(fusion.get("threshold_objective", "f1"))

    regimes: List[FeatureRegime] = ["structured_only", "llm_only", "fused"]
    summary_rows: List[Dict[str, Any]] = []

    for reg in regimes:
        Xtr_sp = _sparse_block(reg, X_tr_sp, Z_tr)
        Xte_sp = _sparse_block(reg, X_te_sp, Z_te)
        X_fit_sp = Xtr_sp[idx_fit]
        X_val_sp = Xtr_sp[idx_val]
        y_fit = y_train[idx_fit]
        y_val = y_train[idx_val]

        Xtr_den = _dense_block(reg, X_tr_svd, Z_tr)
        Xte_den = _dense_block(reg, X_te_svd, Z_te)
        X_fit_den = Xtr_den[idx_fit]
        X_val_den = Xtr_den[idx_val]

        # ----- Logistic regression -----
        lr = LogisticRegression(
            max_iter=lr_max_iter,
            C=lr_c,
            class_weight=lr_class_weight,
            random_state=seed,
            solver=_lr_solver(X_fit_sp.shape[1]),
        )
        lr.fit(X_fit_sp, y_fit)
        val_proba = lr.predict_proba(X_val_sp)[:, 1]
        pick = pick_threshold(y_val, val_proba, mode="f1" if thr_mode == "f1" else "recall")
        lr_full = LogisticRegression(
            max_iter=lr_max_iter,
            C=lr_c,
            class_weight=lr_class_weight,
            random_state=seed,
            solver=_lr_solver(Xtr_sp.shape[1]),
        )
        lr_full.fit(Xtr_sp, y_train)
        te_proba = lr_full.predict_proba(Xte_sp)[:, 1]
        pred05 = (te_proba >= 0.5).astype(int)
        pred_t = (te_proba >= pick.threshold).astype(int)
        m_lr = _metrics_block(
            y_test,
            pred05,
            te_proba,
            pred_t,
            {"learner": "logistic_regression", "feature_regime": reg, "val_threshold": float(pick.threshold)},
        )
        key = f"lr_{reg}"
        save_json(m_lr, out_dir / f"metrics_{key}.json")
        _save_pred_csv(out_dir / f"preds_{key}.csv", test_pair_ids, y_test, te_proba, pred05, pred_t, float(pick.threshold))
        summary_rows.append({"key": key, "learner": "LR", "regime": reg, "auc": m_lr.get("auc"), "auprc": m_lr.get("auprc"), "sensitivity": m_lr.get("sensitivity"), "f1": m_lr.get("f1")})

        # ----- MLP -----
        in_dim = Xtr_den.shape[1]
        mlp, best_ep = _train_torch_classifier(
            FusionMLP(in_dim, hidden=mlp_hidden, dropout=mlp_dropout),
            X_fit_den,
            y_fit.astype(np.float32),
            X_val_den,
            y_val.astype(np.float32),
            epochs=mlp_epochs,
            lr=mlp_lr,
            weight_decay=mlp_wd,
            batch_size=mlp_batch,
            device=device,
        )
        val_mlp_p = _torch_predict_proba(mlp, X_val_den, device, mlp_batch)
        pick_mlp = pick_threshold(y_val, val_mlp_p, mode="f1" if thr_mode == "f1" else "recall")
        if refit_nn:
            mlp = _refit_torch_full(
                lambda: FusionMLP(in_dim, hidden=mlp_hidden, dropout=mlp_dropout),
                Xtr_den,
                y_train.astype(np.float32),
                best_ep,
                mlp_lr,
                mlp_wd,
                mlp_batch,
                device,
            )
        te_mlp = _torch_predict_proba(mlp, Xte_den, device, mlp_batch)
        p05 = (te_mlp >= 0.5).astype(int)
        pt = (te_mlp >= pick_mlp.threshold).astype(int)
        m_mlp = _metrics_block(
            y_test,
            p05,
            te_mlp,
            pt,
            {"learner": "mlp", "feature_regime": reg, "val_threshold": float(pick_mlp.threshold), "best_epoch": best_ep},
        )
        key = f"mlp_{reg}"
        save_json(m_mlp, out_dir / f"metrics_{key}.json")
        _save_pred_csv(out_dir / f"preds_{key}.csv", test_pair_ids, y_test, te_mlp, p05, pt, float(pick_mlp.threshold))
        summary_rows.append({"key": key, "learner": "MLP", "regime": reg, "auc": m_mlp.get("auc"), "auprc": m_mlp.get("auprc"), "sensitivity": m_mlp.get("sensitivity"), "f1": m_mlp.get("f1")})

        # ----- Hybrid CNN + Transformer (nhead=4) -----
        hy = FusionHybridCNNTransformer(
            in_dim,
            n_tokens=hy_tokens,
            d_model=hy_d_model,
            nhead=4,
            num_layers=hy_layers,
            dim_feedforward=hy_ff,
            dropout=hy_drop,
        )
        hy, best_ep_h = _train_torch_classifier(
            hy,
            X_fit_den,
            y_fit.astype(np.float32),
            X_val_den,
            y_val.astype(np.float32),
            epochs=hy_epochs,
            lr=hy_lr,
            weight_decay=hy_wd,
            batch_size=hy_batch,
            device=device,
        )
        val_h_p = _torch_predict_proba(hy, X_val_den, device, hy_batch)
        pick_h = pick_threshold(y_val, val_h_p, mode="f1" if thr_mode == "f1" else "recall")
        if refit_nn:
            hy = _refit_torch_full(
                lambda: FusionHybridCNNTransformer(
                    in_dim,
                    n_tokens=hy_tokens,
                    d_model=hy_d_model,
                    nhead=4,
                    num_layers=hy_layers,
                    dim_feedforward=hy_ff,
                    dropout=hy_drop,
                ),
                Xtr_den,
                y_train.astype(np.float32),
                best_ep_h,
                hy_lr,
                hy_wd,
                hy_batch,
                device,
            )
        te_h = _torch_predict_proba(hy, Xte_den, device, hy_batch)
        p05h = (te_h >= 0.5).astype(int)
        pth = (te_h >= pick_h.threshold).astype(int)
        m_h = _metrics_block(
            y_test,
            p05h,
            te_h,
            pth,
            {"learner": "hybrid_cnn_transformer", "feature_regime": reg, "val_threshold": float(pick_h.threshold), "best_epoch": best_ep_h},
        )
        key = f"hybrid_{reg}"
        save_json(m_h, out_dir / f"metrics_{key}.json")
        _save_pred_csv(out_dir / f"preds_{key}.csv", test_pair_ids, y_test, te_h, p05h, pth, float(pick_h.threshold))
        summary_rows.append({"key": key, "learner": "Hybrid", "regime": reg, "auc": m_h.get("auc"), "auprc": m_h.get("auprc"), "sensitivity": m_h.get("sensitivity"), "f1": m_h.get("f1")})

    summ = pd.DataFrame(summary_rows)
    save_dataframe(summ, out_dir / "fusion_summary.csv")

    comp = _load_comparison_baselines(cfg, fusion, Path(cfg["paths"]["outputs"]))
    if comp:
        save_dataframe(pd.DataFrame(comp), out_dir / "fusion_comparison_table.csv")
    (out_dir / "fusion_config_snapshot.json").write_text(json.dumps({"fusion": fusion, "seed": seed}, indent=2), encoding="utf-8")
    logger.info("Fusion experiments complete; summary -> %s", out_dir / "fusion_summary.csv")
    return summ
