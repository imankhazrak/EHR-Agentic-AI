"""Integration test: fusion dataset builder on synthetic CSVs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.fusion.dataset import build_fusion_artifacts
from src.ml.fusion.train_eval import run_fusion_experiments


def test_build_fusion_artifacts_smoke(tmp_path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    train = pd.DataFrame(
        {
            "pair_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "label_lipid_disorder": [0, 1, 0, 1, 0, 1, 0, 1],
            "diagnoses_codes_current": ["25000", "4019", "2724", "25000", "25000", "4019", "2724", "25000"],
            "procedures_codes_current": ["0066", "", "0066", "", "0066", "", "0066", ""],
            "medications_current": ["Metformin", "Aspirin", "Insulin", "Metformin", "Aspirin", "Insulin", "X", "Y"],
            "narrative_current": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )
    test = pd.DataFrame(
        {
            "pair_id": [9, 10],
            "label_lipid_disorder": [0, 1],
            "diagnoses_codes_current": ["25000", "2724"],
            "procedures_codes_current": ["", ""],
            "medications_current": ["Z", "W"],
            "narrative_current": ["i", "j"],
        }
    )
    train.to_csv(proc / "train.csv", index=False)
    test.to_csv(proc / "test.csv", index=False)
    py = np.linspace(0.2, 0.8, len(train))
    st = pd.DataFrame(
        {
            "pair_id": train["pair_id"].to_numpy(),
            "prob_yes": py,
            "prob_no": 1.0 - py,
            "margin_logit": np.zeros(len(train)),
            "pred_hard": (py >= 0.5).astype(int),
        }
    )
    te = pd.DataFrame(
        {
            "pair_id": test["pair_id"].to_numpy(),
            "prob_yes": [0.3, 0.7],
            "prob_no": [0.7, 0.3],
            "margin_logit": [0.0, 0.0],
            "pred_hard": [0, 1],
        }
    )
    st_path = out / "llm_train.csv"
    te_path = out / "llm_test.csv"
    st.to_csv(st_path, index=False)
    te.to_csv(te_path, index=False)

    paths = build_fusion_artifacts(
        proc,
        out,
        st_path,
        te_path,
        feature_type="bag_of_codes",
        val_ratio=0.25,
        split_seed=42,
        svd_dim=8,
    )
    assert paths.manifest.exists()
    bundle_np = np.load(paths.x_train_svd)
    assert bundle_np.shape[0] == len(train)
    assert bundle_np.shape[1] <= 8

    cfg = {
        "seed": 42,
        "paths": {"outputs": str(out)},
        "fusion": {
            "mlp": {"epochs": 3, "batch_size": 4, "hidden": [16, 8]},
            "hybrid": {"epochs": 3, "batch_size": 4, "d_model": 32, "n_tokens": 4, "dim_feedforward": 64, "num_layers": 1},
            "logistic_regression": {"max_iter": 200},
            "refit_full_train_nn": True,
            "run_llm_score_subprocess": False,
        },
    }
    summ = run_fusion_experiments(paths, cfg, out_dir=paths.root)
    assert len(summ) == 9
    assert (paths.root / "fusion_summary.csv").exists()
