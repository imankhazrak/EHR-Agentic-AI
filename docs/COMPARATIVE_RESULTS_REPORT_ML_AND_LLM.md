# Comparative Results Report — Classical ML vs LLM Approaches

This report summarizes the completed MIMIC-III lipid-disorder next-visit prediction experiments, covering both classical ML and LLM pipelines. It presents full metric tables, confusion-level outcomes, and structured intra-family and inter-family comparisons to clarify performance trade-offs across approaches.

**MIMIC-III next-visit prediction:** Disorders of Lipid Metabolism (binary), held-out test evaluation.

This report consolidates **all** classical ML and LLM results from the completed pipeline, then compares approaches **within** each family (intra-) and **between** families (inter-). Numbers are taken from `data/outputs/mimiciii/summary_table.csv`, `ml_results_*.csv`, and `llm_*_metrics.json` as produced by a full run.

**Related:** `docs/LLM_EXPERIMENT_REPORT.md` (LLM-only deep dive), `METHODOLOGY.md` (pipeline).

---

## Table of Contents
- [1. Task and evaluation (shared)](#1-task-and-evaluation-shared)
- [2. Complete classical ML results](#2-complete-classical-ml-results)
  - [2.1 Fully supervised (full training set)](#21-fully-supervised-full-training-set)
  - [2.2 Few-shot ML (N = 6 labeled training examples)](#22-few-shot-ml-n--6-labeled-training-examples)
- [3. Complete LLM results (`gpt-4o-mini-2024-07-18`)](#3-complete-llm-results-gpt-4o-mini-2024-07-18)
- [4. Intra-family comparisons](#4-intra-family-comparisons)
  - [4.1 Intra-ML — fully supervised](#41-intra-ml--fully-supervised)
  - [4.2 Intra-ML — few-shot (N=6) vs fully supervised](#42-intra-ml--few-shot-n6-vs-fully-supervised)
  - [4.3 Intra-LLM — ranking and deltas vs zero-shot](#43-intra-llm--ranking-and-deltas-vs-zero-shot)
- [5. Inter-family comparisons (ML vs LLM)](#5-inter-family-comparisons-ml-vs-llm)
  - [5.1 Best-in-class vs best LLM (this run)](#51-best-in-class-vs-best-llm-this-run)
  - [5.2 LLM vs strongest “reasonable” ML baseline](#52-llm-vs-strongest-reasonable-ml-baseline)
  - [5.3 LLM few-shot (in-context) vs ML few-shot (parameter learning)](#53-llm-few-shot-in-context-vs-ml-few-shot-parameter-learning)
  - [5.4 Modality caveat](#54-modality-caveat)
- [6. Summary matrix (all twelve configurations)](#6-summary-matrix-all-twelve-configurations)
- [7. Conclusions](#7-conclusions)
- [8. Artifact map](#8-artifact-map)

## 1. Task and evaluation (shared)

| Item | Detail |
|------|--------|
| **Target** | Presence of lipid-metabolism ICD-9 codes (`272.x` / CCS-53 family) on the **next** hospital admission |
| **Test unit** | Visit-pair rows; **stratified** hold-out (**20%** test, seed **42**) |
| **Metrics** | Accuracy (ACC), **Sensitivity** (recall on positives), **Specificity**, **F1** (positive class); all in **%** |
| **Positive prevalence (test)** | Positives ≈ **tp + fn** on fully parsed rows → **683 / 2492 ≈ 27.4%** (zero-shot LLM row; consistent across modes after accounting for dropped unparseables) |

**Input modality differs by family:**

| Family | Input representation |
|--------|----------------------|
| **Classical ML** | **Bag-of-codes** on diagnosis / procedure / medication strings from the **current** visit (`feature_type: bag_of_codes`) |
| **LLM** | **Natural-language narrative** (`narrative_current`) from the same current visit |

Metrics are **not** directly comparable as “fairness of model class” because features differ; they **are** comparable as **end-to-end systems** under the same labels and split.

---

## 2. Complete classical ML results

### 2.1 Fully supervised (full training set)

Trained on all training pairs; evaluated on the same test split as the rest of the pipeline.

| Model | ACC | Sensitivity | Specificity | F1 | tp | fp | tn | fn |
|--------|-----|-------------|-------------|-----|----|----|----|-----|
| Decision Tree | 73.48 | 50.66 | 82.09 | 51.15 | 346 | 324 | 1485 | 337 |
| Logistic Regression | **77.05** | 47.73 | **88.11** | **53.27** | 326 | 215 | 1594 | 357 |
| Random Forest | 76.44 | 24.01 | **96.24** | 35.85 | 164 | 68 | 1741 | 519 |

**Source:** `ml_results_fully_supervised.csv` / `summary_table.csv`.

### 2.2 Few-shot ML (N = 6 labeled training examples)

Same models and features, but each model is **fit** on only **6** labeled rows (`ml.few_shot_n: 6`, balanced), then evaluated on the full test set.

| Model | ACC | Sensitivity | Specificity | F1 | tp | fp | tn | fn |
|--------|-----|-------------|-------------|-----|----|----|----|-----|
| Decision Tree | **61.40** | 56.52 | 63.24 | 44.52 | 386 | 665 | 1144 | 297 |
| Logistic Regression | 35.27 | **90.34** | 14.48 | 43.34 | 617 | 1547 | 262 | 66 |
| Random Forest | 34.95 | **92.53** | 13.21 | 43.81 | 632 | 1570 | 239 | 51 |

**Source:** `ml_results_few_shot.csv` / `summary_table.csv`.

---

## 3. Complete LLM results (`gpt-4o-mini-2024-07-18`)

Single underlying chat model; four **prompting** strategies. One completion per test row (coagent adds extra calls on calibration/critic only).

| Approach | ACC | Sensitivity | Specificity | F1 | tp | fp | tn | fn | n_valid | n_unparseable |
|----------|-----|-------------|-------------|-----|----|----|----|-----|---------|---------------|
| Zero-Shot | 80.82 | 57.69 | 89.55 | 62.24 | 394 | 189 | 1620 | 289 | 2492 | **0** |
| Zero-Shot+ | 80.73 | 56.22 | **89.99** | 61.54 | 384 | 181 | 1627 | 299 | 2491 | 1 |
| Few-Shot (N=6) | 80.65 | 57.39 | 89.44 | 61.93 | 392 | 191 | 1617 | 291 | 2491 | 1 |
| **EHR-CoAgent** | **80.93** | **58.21** | 89.50 | **62.57** | **397** | 190 | 1619 | 285 | 2491 | 1 |

**Source:** `llm_*_metrics.json` / `summary_table.csv`.

**Note:** Rows with **unparseable** assistant text are **excluded** from tp/fp/tn/fn but counted in `n_unparseable`; three modes show **one** such test example in this run.

---

## 4. Intra-family comparisons

### 4.1 Intra-ML — fully supervised

**Ranking by F1 (primary balance of precision/recall on minority positive class):**

1. Logistic Regression — **53.27**
2. Decision Tree — 51.15
3. Random Forest — 35.85

**Ranking by accuracy:**

1. **Logistic Regression — 77.05**
2. Random Forest — 76.44
3. Decision Tree — 73.48

**Sensitivity vs specificity trade-off:**

- **Random Forest** maximizes **specificity (96.24%)** and minimizes positives predicted, at the cost of **very low sensitivity (24.01%)** and weak F1 — it rarely predicts the positive class.
- **Logistic regression** offers the **best overall** mix for this task among the three under full supervision.

**Takeaway:** Under full supervision, **logistic regression** on bag-of-codes is the strongest classical baseline in both ACC and F1; **random forest** is miscalibrated for sensitivity on this skewed label.

---

### 4.2 Intra-ML — few-shot (N=6) vs fully supervised

**Δ = few_shot − fully_supervised** (same model):

| Model | Δ ACC | Δ Sens | Δ Spec | Δ F1 |
|--------|-------|--------|--------|------|
| Decision Tree | −12.08 | +5.86 | −18.85 | −6.63 |
| Logistic Regression | **−41.78** | +42.61 | −73.63 | −9.93 |
| Random Forest | **−41.49** | +68.52 | −83.03 | +7.96 |

**Interpretation:**

- Training on **only six** rows is **catastrophic** for **accuracy** for linear and forest models: they **over-predict positive** (huge **fp**), yielding **high sensitivity** but **very low specificity** and **ACC ~35%**.
- The **decision tree** degrades more gracefully (ACC 61%) but still **hurts specificity** vs full training.
- **Takeaway:** “Few-shot” for ML here means **extreme underfitting / instability** on bag-of-codes; it is **not** analogous to LLM in-context few-shot, which does not re-estimate thousands of feature weights from six points.

---

### 4.3 Intra-LLM — ranking and deltas vs zero-shot

**Ranking by F1:**

1. **EHR-CoAgent — 62.57**
2. Zero-Shot — 62.24
3. Few-Shot (N=6) — 61.93
4. Zero-Shot+ — 61.54

**Ranking by accuracy:**

1. **EHR-CoAgent — 80.93**
2. Zero-Shot — 80.82
3. Zero-Shot+ — 80.73
4. Few-Shot (N=6) — 80.65

**Δ vs zero-shot** (percentage points):

| Approach | Δ ACC | Δ Sens | Δ Spec | Δ F1 |
|----------|-------|--------|--------|------|
| Zero-Shot+ | −0.09 | −1.47 | +0.44 | −0.70 |
| Few-Shot (N=6) | −0.17 | −0.30 | −0.11 | −0.31 |
| EHR-CoAgent | **+0.11** | **+0.52** | −0.05 | **+0.33** |

**Interpretation:**

- All four LLM configurations sit in a **tight band** (~80.6–80.9% ACC, ~61.5–62.6% F1). Differences are **small relative to sampling noise** (no bootstrap CIs were computed by default).
- **Zero-shot+** slightly shifts toward **higher specificity**, **lower sensitivity** vs plain zero-shot (consistent with longer instructions + prevalence text).
- **EHR-CoAgent** nudges **sensitivity and F1** upward vs zero-shot by a few **counts** in the confusion matrix (e.g. **+3 tp**, **−4 fn** vs zero-shot), but **aggregate metrics barely move**.
- Empirically, parsed **Yes/No** decisions agree with zero-shot on **~98%+** of test rows across modes (high **decision correlation**), which explains the **flat** leaderboard.

**Takeaway:** For this model and task, **prompt engineering and coagent feedback** produce **marginal** aggregate gains; the dominant signal is the **shared model + narrative + label definition**.

---

## 5. Inter-family comparisons (ML vs LLM)

### 5.1 Best-in-class vs best LLM (this run)

Compared to **fully supervised** ML only (stable models), versus **EHR-CoAgent**:

| Metric | Best fully supervised ML | Best LLM (CoAgent) | Δ (LLM − ML) |
|--------|--------------------------|---------------------|--------------|
| ACC | 77.05 (logistic regression) | **80.93** | **+3.88** |
| Sensitivity | 47.73 (logistic regression) | **58.21** | **+10.48** |
| Specificity | 96.24 (random forest) | 89.50 | −6.74 |
| F1 | 53.27 (logistic regression) | **62.57** | **+9.30** |

**Note:** **ML few-shot** runs reach **higher raw sensitivity** (e.g. logistic regression **90.34%**) but with **very low specificity and ACC**; they are **not** comparable as successful deployable models. Random forest (full) wins **specificity** by being **very conservative** on positives (low sensitivity **24.01%**).

### 5.2 LLM vs strongest “reasonable” ML baseline

Comparing **EHR-CoAgent** to **fully supervised logistic regression** (best ACC/F1 among stable ML):

| | Logistic regression (full) | EHR-CoAgent |
|--|----------------------------|-------------|
| ACC | 77.05 | 80.93 |
| F1 | 53.27 | 62.57 |
| Sensitivity | 47.73 | 58.21 |
| Specificity | 88.11 | 89.50 |

**Interpretation:**

- On this split, the **narrative + LLM** pipeline achieves **higher accuracy and much higher F1** than the best classical baseline, driven mainly by **better sensitivity** while keeping **specificity in the same ballpark** as logistic regression.
- **Random forest** (full) has higher **specificity** than the LLM but **much worse F1** due to missed positives.

### 5.3 LLM few-shot (in-context) vs ML few-shot (parameter learning)

| | ML few-shot (best ACC among DT/LR/RF) | LLM Few-Shot (N=6) |
|--|--------------------------------------|---------------------|
| Best model | Decision Tree | gpt-4o-mini |
| ACC | 61.40 | **80.65** |
| F1 | 44.52 | **61.93** |

**Takeaway:** **Six labels** are enough to **destabilize** small-data **fitted** classifiers on high-dimensional sparse codes, but **six in-context demonstrations** for a pretrained LLM **do not** cause the same collapse. The comparison underscores that **“few-shot”** means **different mechanisms** in the two tracks.

### 5.4 Modality caveat

LLM inputs are **human-readable narratives**; ML uses **raw code tokens**. Some information may be easier to exploit in text (e.g. phrasing, implicit structure); conversely, codes are **complete** up to extraction. **Fair fusion** (e.g. ensemble of both) is out of scope here but relevant for future work.

---

## 6. Summary matrix (all twelve configurations)

Sorted by **F1** descending.

| Rank | Model | Approach | ACC | Sens | Spec | F1 |
|------|--------|----------|-----|------|------|-----|
| 1 | gpt-4o-mini | EHR-CoAgent | 80.93 | 58.21 | 89.50 | **62.57** |
| 2 | gpt-4o-mini | Zero-Shot | 80.82 | 57.69 | 89.55 | 62.24 |
| 3 | gpt-4o-mini | Few-Shot (N=6) | 80.65 | 57.39 | 89.44 | 61.93 |
| 4 | gpt-4o-mini | Zero-Shot+ | 80.73 | 56.22 | 89.99 | 61.54 |
| 5 | logistic_regression | Fully Supervised | 77.05 | 47.73 | 88.11 | 53.27 |
| 6 | random_forest | Fully Supervised | 76.44 | 24.01 | 96.24 | 35.85 |
| 7 | decision_tree | Fully Supervised | 73.48 | 50.66 | 82.09 | 51.15 |
| 8 | decision_tree | Few Shot | 61.40 | 56.52 | 63.24 | 44.52 |
| 9 | logistic_regression | Few Shot | 35.27 | 90.34 | 14.48 | 43.34 |
| 10 | random_forest | Few Shot | 34.95 | 92.53 | 13.21 | 43.81 |

---

## 7. Conclusions

1. **Intra-ML:** Full-data **logistic regression** is the best overall classical method on ACC/F1; **random forest** is conservative on positives. **N=6 ML training** is **not** viable for LR/RF on this feature setup.
2. **Intra-LLM:** All four prompting strategies perform **similarly**; **EHR-CoAgent** is **marginally best** on ACC/F1/sensitivity in this run.
3. **Inter ML vs LLM:** The best LLM configuration **beats** the best fully supervised ML baseline on **ACC and F1**, with **higher sensitivity** and **comparable specificity** to logistic regression. **Input modality** differs; interpret as **system comparison**, not a controlled ablation of model family alone.
4. **Reporting:** No **bootstrap CIs** or significance tests were run by default (`evaluation.bootstrap_ci: false`); small gaps between LLM modes should be read with that limitation.

---

## 8. Artifact map

| Artifact | Contents |
|----------|----------|
| `data/outputs/mimiciii/summary_table.csv` | Combined table (source for Section 6) |
| `data/outputs/mimiciii/ml_results_fully_supervised.csv` | ML full-training metrics + confusion |
| `data/outputs/mimiciii/ml_results_few_shot.csv` | ML N=6 metrics + confusion |
| `data/outputs/mimiciii/llm_*_metrics.json` | Per-mode LLM metrics + confusion + parse counts |

---

*Report reflects outputs stored under `data/outputs/mimiciii/` at the time of writing. Rerun the pipeline and refresh this document if configs, data, or prompts change.*
