# Fusion experiment report (structured + LLM logit probabilities)

## Design

- **Structured:** Same as ML baselines — `CountVectorizer` binary bag-of-codes on current-visit diagnosis / procedure / medication strings (`src/ml/feature_builder.py`). Vocabulary is **fit on full `train.csv` only**; `test.csv` is transformed (no test leakage into the vectorizer).
- **LLM features:** `prob_yes`, `prob_no`, `margin_logit` (and optional `pred_hard`) from a **single forward pass** at `Prediction:\n` — two-way softmax over max Yes-token vs max No-token logits (`src/llm/logit_pair_scorer.py`). No free-text numeric probabilities.
- **Neural input:** `TruncatedSVD` on the **full train** sparse design matrix → dense `svd_dim` (default 256); MLP and hybrid consume `[SVD | LLM features]`. Logistic regression uses **sparse** `hstack(structured, LLM)` where applicable.
- **Train / val_tune / test:** Stratified `train_fit` / `val_tune` split inside `train.csv` (default 85% / 15%, seed 42). Fusion **weights** are fit on `train_fit` only; **classification threshold** is tuned on `val_tune` scores (default: maximize F1). **Neural nets** optionally **refit on full `train.csv`** for the same number of epochs as the best early-stopping epoch (`fusion.refit_full_train_nn: true`). Final metrics are on **held-out `test.csv`** only.
- **Models:** (1) sklearn `LogisticRegression`; (2) MLP 128→64; (3) Hybrid: linear projection to `(n_tokens=8, d_model=128)`, Conv1d kernel 3, `TransformerEncoder` with **`nhead=4` exactly**, 2 layers, mean pool, linear head.

## Leakage prevention

1. **Test LLM scores** are produced only from `test.csv` narratives and are used **only** for final evaluation rows — never for fitting vectorizers, SVD, fusion weights, or threshold selection.
2. **Train LLM scores (infold bundle):** Each train row is scored with the **frozen** Gemma+LoRA model on **that same row’s** narrative (`score_llm_logits_split --split train`). The LLM was **not** fine-tuned during fusion, but the score is still **in-fold** in the stacking sense: the train-row LLM logit is not cross-validated, so any subtle train-specific overfitting of the LLM to phrasing can leak into fusion inputs. Treat the **`fusion_infold/`** runs as **validation-informed / mildly optimistic fusion** unless you only trust the OOF bundle.
3. **Train LLM scores (OOF bundle):** `score_llm_logits_train_oof.py` assigns each train row a score from the model only when that row falls in the **validation** fold of a stratified K-fold split on `train.csv` (default K=5). Each row is scored exactly once. The **`fusion_oof/`** runs are the **scientifically preferred** train-side fusion for comparing to classical stacking, because train labels are not used to pick the forward used for that row’s own score (beyond stratified fold assignment).
4. **Threshold** is chosen using **validation** predictions (`val_tune`) from models trained on `train_fit`.

## Held-out experiment bundle (MIMIC-III)

All artifacts for this study live under a **single named output root** (analogous to a finetuning run directory):

| Path | Role |
|------|------|
| `data/outputs/mimiciii_fusion_holdout_run/` | Experiment root (`configs/fusion_holdout_mimiciii.yaml` sets `paths.outputs`) |
| `.../llm_logit_scores_train.csv` | In-fold Gemma+LoRA logits on **all** `train.csv` rows |
| `.../llm_logit_scores_test.csv` | Logits on **held-out** `test.csv` |
| `.../llm_logit_scores_train_oof.csv` | Stratified **out-of-fold** logits on `train.csv` |
| `.../fusion_infold/` | Built with infold train LLM scores + all fusion models/metrics |
| `.../fusion_oof/` | Built with OOF train LLM scores + all fusion models/metrics |
| `.../fusion_final_comparison_table.csv` | Produced by `python -m src.scripts.summarize_fusion_holdout` (infold + OOF metrics + ML/LLM baselines) |

**Driver script:** `scripts/run_fusion_holdout_full.sh` (train score, test score, OOF score, build+train infold, build+train OOF, summarize).

**Slurm:** `slurm/mimiciii_fusion_holdout_full.slurm` — loads `conda` env `gemma4-smoke`, runs the driver. Wall time is set to **80 hours** because scoring is **row-wise** forward passes (~30k train + ~30k OOF + ~7.5k test ≈ **6.7×10⁴** forwards; duration is hardware- and narrative-length dependent).

### Cluster execution status

The full pipeline was **submitted to Slurm** on the **`gpu`** partition (`account=pcs0229`). At the time of the last agent check, the job was **still pending** (queue backlog / priority). Latest submission uses an **80-hour** wall time (verify with `squeue -u $USER -n mimiciii_fusion_holdout` and `sacct -j <jobid>`).

**After the job completes**, refresh this report by opening:

`data/outputs/mimiciii_fusion_holdout_run/fusion_final_comparison_table.csv`

and copying the headline metrics into the table below. If the job **fails or times out**, inspect `logs/slurm-mimiciii-fusion-holdout-<jobid>.out` / `.err`, then re-submit or split work (e.g. score splits in separate jobs).

## Outputs (generic layout)

Under each `artifact_root` directory (e.g. `fusion_infold/`, `fusion_oof/`):

| File | Content |
|------|---------|
| `artifacts/feature_stage.joblib` | Vectorizer + `TruncatedSVD` |
| `artifacts/*.npy`, `*.npz` | Design matrices, labels, indices, pair_ids |
| `fusion_row_meta.csv` | `pair_id`, `y`, `fusion_split`, LLM columns |
| `fusion_manifest.json` | Run metadata |
| `metrics_{lr,mlp,hybrid}_{regime}.json` | Test AUC/AUPRC + `threshold_tuned_metrics` (accuracy, sensitivity/recall, specificity, precision, F1, TP/FP/TN/FN) |
| `preds_*.csv` | Test `pair_id`, scores, default vs val-tuned preds |
| `fusion_summary.csv` | Short overview |

The **final comparison table** merges all `metrics_*.json` files from both bundles and adds baseline rows from:

- `data/outputs/mimiciii_gpt4o_mini/ml_results_fully_supervised.csv` (structured ML baselines),
- `data/outputs/mimiciii_gemma/llm_finetuned_test_results.csv` (LLM hard vs prob@0.5 on **test only**).

Column semantics align with the requested science table: `model_family`, `feature_set` (regime), `threshold_val_tuned`, classification metrics, AUC, AUPRC, confusion counts. Baseline rows use `train_llm_score_mode` = `n/a_ml_baseline`, `test_only_llm_hard`, or `test_only_llm_prob`.

## Smoke verification

`pytest tests/test_fusion_dataset_build.py` builds synthetic `train.csv` / `test.csv`, synthetic LLM score CSVs, runs `build_fusion_artifacts` + full `run_fusion_experiments` (all nine model/regime combinations) successfully. This does **not** substitute for MIMIC GPU scoring.

## Final comparison table (MIMIC held-out test)

**Populate from** `data/outputs/mimiciii_fusion_holdout_run/fusion_final_comparison_table.csv` **after the Slurm job finishes.**

| train_llm_score_mode | model_family | feature_set | threshold (val-tuned) | accuracy | recall | specificity | precision | F1 | AUC | AUPRC | TP | FP | TN | FN |
|----------------------|--------------|-------------|------------------------|------------|--------|-------------|-----------|-----|-----|-------|----|----|----|-----|
| *(pending run)* | | | | | | | | | | | | | | | |

## Scientific questions (to answer from the CSV)

| Question | Where to look | Answer (fill after run) |
|----------|---------------|-------------------------|
| Does LLM **probability** help over hard Yes/No? | Compare `test_only_llm_hard` vs `test_only_llm_prob` rows (same `llm_finetuned_test_results.csv`; hard uses `pred_binary`, prob uses `prob_yes >= 0.5` — **accuracy/F1 may tie** if decode matches 0.5 rule; **AUC/AUPRC are identical** here because the score curve is the same `prob_yes`; substantive gain is from **using continuous scores inside fusion**, not from re-thresholding the LLM alone.) | *(pending)* |
| Does **fusion** improve **recall**, **AUC**, and **AUPRC** vs `structured_only` and `llm_only`? | Within each `train_llm_score_mode`, compare `fused` vs `structured_only` / `llm_only` for each learner. Prefer **`oof_train_llm`** rows for unbiased train LLM inputs. | *(pending)* |
| Which **model** is best overall? | Among fusion rows, rank by **AUPRC** (primary for rare positives) and **recall** at val-tuned threshold; use **AUC** as tie-breaker. | *(pending)* |
| Is **Hybrid CNN+Transformer** worth the complexity? | Compare `Hybrid_CNN_Transformer` vs `MLP` vs `LogisticRegression` at the same `feature_set` and `train_llm_score_mode`; weigh **marginal metric gain vs training time / code surface**. | *(pending)* |

### Infold vs OOF (mandatory interpretation)

- **`infold_train_llm`:** Train-row LLM logits are computed on the **same** narrative used in fusion for that row. This is **not** out-of-fold with respect to the LLM’s exposure to train phrasing. Label it as **validation-threshold fusion on in-fold LLM scores** in any publication-style writeup.
- **`oof_train_llm`:** Preferred for answering whether **fusion** helps when train LLM features are **not** self-scores in a CV sense.

## Recommendation

Re-run `summarize_fusion_holdout` if you add new `metrics_*.json` files. Once `fusion_final_comparison_table.csv` exists, prefer **OOF** rows for conclusions about fusion value; use **infold** as a sensitivity analysis.
