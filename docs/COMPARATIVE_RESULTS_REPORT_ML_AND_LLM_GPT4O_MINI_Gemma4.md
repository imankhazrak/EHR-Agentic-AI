# Comparative Results Report — Classical ML vs LLM Approaches (GPT-4o-mini & Gemma 4)

This report summarizes the completed MIMIC-III lipid-disorder next-visit prediction experiments, covering classical ML and **two** local/API LLM tracks: **OpenAI `gpt-4o-mini`** and **local `google/gemma-4-e4b-it`**. It presents full metric tables, confusion-level outcomes, and structured intra-family and inter-family comparisons.

**MIMIC-III next-visit prediction:** Disorders of Lipid Metabolism (binary), held-out test evaluation.

**Data sources:**

- **ML and GPT-4o-mini LLM:** `data/outputs/mimiciii/` (`summary_table.csv`, `ml_results_*.csv`, `llm_*_metrics.json`).
- **Gemma 4 LLM (prompt experiments):** `data/outputs/mimiciii_gemma/` (`llm_*_metrics.json`); Co-Agent completed in Slurm job **46615354** (see `docs/MIMICIII_GEMMA_JOB_46510786_RESULTS_REPORT.md`).

> **Naming convention:** Model-scoped output roots (`mimiciii` vs `mimiciii_gemma`); tables label **model + approach** explicitly—metrics from different models are not merged into unlabeled rows.

**Related:** `docs/LLM_EXPERIMENT_REPORT_GPT4O_MINI.md`, `docs/MIMICIII_GEMMA_JOB_46510786_RESULTS_REPORT.md`, `METHODOLOGY.md`.

*HTML export:* `docs/COMPARATIVE_RESULTS_REPORT_ML_AND_LLM_GPT4O_MINI_Gemma4.html` (regenerate with `pandoc COMPARATIVE_RESULTS_REPORT_ML_AND_LLM_GPT4O_MINI_Gemma4.md -o COMPARATIVE_RESULTS_REPORT_ML_AND_LLM_GPT4O_MINI_Gemma4.html --standalone --toc --toc-depth=3 -M title="Comparative Results — Classical ML vs LLM (GPT-4o-mini & Gemma 4)" --self-contained` from `docs/`).

---

## 1. Task and evaluation (shared)

| Item | Detail |
|------|--------|
| **Target** | Presence of lipid-metabolism ICD-9 codes (`272.x` / CCS-53 family) on the **next** hospital admission |
| **Test unit** | Visit-pair rows; **stratified** hold-out (**20%** test, seed **42**) |
| **Metrics** | Accuracy (ACC), **Sensitivity** (recall on positives), **Specificity**, **F1** (positive class); all in **%** |
| **Positive prevalence (test)** | Positives ≈ **tp + fn** on fully parsed rows → **683 / 2492 ≈ 27.4%** (consistent across modes after accounting for dropped unparseables) |

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

**Source:** `data/outputs/mimiciii/ml_results_fully_supervised.csv` / `summary_table.csv`.

### 2.2 Few-shot ML (N = 6 labeled training examples)

Same models and features, but each model is **fit** on only **6** labeled rows (`ml.few_shot_n: 6`, balanced), then evaluated on the full test set.

| Model | ACC | Sensitivity | Specificity | F1 | tp | fp | tn | fn |
|--------|-----|-------------|-------------|-----|----|----|----|-----|
| Decision Tree | **61.40** | 56.52 | 63.24 | 44.52 | 386 | 665 | 1144 | 297 |
| Logistic Regression | 35.27 | **90.34** | 14.48 | 43.34 | 617 | 1547 | 262 | 66 |
| Random Forest | 34.95 | **92.53** | 13.21 | 43.81 | 632 | 1570 | 239 | 51 |

**Source:** `data/outputs/mimiciii/ml_results_few_shot.csv` / `summary_table.csv`.

---

## 3. Complete LLM results — `gpt-4o-mini-2024-07-18`

Single underlying chat model; four **prompting** strategies. One completion per test row (Co-Agent adds extra calls on calibration/critic only).

| Approach | ACC | Sensitivity | Specificity | F1 | tp | fp | tn | fn | n_valid | n_unparseable |
|----------|-----|-------------|-------------|-----|----|----|----|-----|---------|---------------|
| Zero-Shot | 80.82 | 57.69 | 89.55 | 62.24 | 394 | 189 | 1620 | 289 | 2492 | **0** |
| Zero-Shot+ | 80.73 | 56.22 | **89.99** | 61.54 | 384 | 181 | 1627 | 299 | 2491 | 1 |
| Few-Shot (N=6) | 80.65 | 57.39 | 89.44 | 61.93 | 392 | 191 | 1617 | 291 | 2491 | 1 |
| **EHR-CoAgent** | **80.93** | **58.21** | 89.50 | **62.57** | **397** | 190 | 1619 | 285 | 2491 | 1 |

**Source:** `data/outputs/mimiciii/llm_*_metrics.json` / `summary_table.csv`.

**Note:** Rows with **unparseable** assistant text are **excluded** from tp/fp/tn/fn but counted in `n_unparseable`; three modes show **one** such test example in this run.

---

## 4. Complete LLM results — `google/gemma-4-e4b-it` (local snapshot)

Same four **prompt** modes and evaluation protocol as the GPT run; local inference via `models/hf_snapshots/google--gemma-4-e4b-it`. Baseline modes from Slurm **46510786**; **EHR-CoAgent** test pass completed in Slurm **46615354**.

| Approach | ACC | Sensitivity | Specificity | F1 | tp | fp | tn | fn | n_valid | n_unparseable |
|----------|-----|-------------|-------------|-----|----|----|----|-----|---------|---------------|
| Zero-Shot | 80.26 | 55.93 | 89.44 | 60.83 | 382 | 191 | 1618 | 301 | 2492 | **0** |
| **Zero-Shot+** | **80.97** | **60.61** | 88.66 | **63.59** | 414 | 205 | 1603 | 269 | 2491 | 1 |
| Few-Shot (N=6) | 80.10 | 57.83 | 88.50 | 61.43 | 395 | 208 | 1601 | 288 | 2492 | 0 |
| EHR-CoAgent | 79.78 | **62.96** | 86.12 | 63.05 | 430 | 251 | 1558 | 253 | 2492 | **0** |

**Source:** `data/outputs/mimiciii_gemma/llm_*_metrics.json`; Co-Agent authoritative in `llm_coagent_metrics.json` if `summary_table.csv` omits that row.

**Note:** Zero-Shot+ has **one** unparseable response (`n_valid = 2491`).

---

## 5. Cross-LLM comparison (same prompts, different models)

**Δ = Gemma 4 − GPT-4o-mini** (percentage points), matched by prompt approach.

| Approach | Δ ACC | Δ Sens | Δ Spec | Δ F1 |
|----------|-------|--------|--------|------|
| Zero-Shot | −0.56 | −1.76 | −0.11 | −1.41 |
| Zero-Shot+ | **+0.24** | **+4.39** | −1.33 | **+2.05** |
| Few-Shot (N=6) | −0.55 | +0.44 | −0.94 | −0.50 |
| EHR-CoAgent | −1.15 | **+4.75** | −3.38 | +0.48 |

**Readout:**

- **Aggregate accuracy** remains near **~80%** for both models across modes; differences are **small** for zero-shot and few-shot, with Gemma **Zero-Shot+** and **Co-Agent** trading **higher sensitivity** (and for Co-Agent, **lower specificity**) vs GPT.
- **Best F1 (this split):** Gemma **Zero-Shot+ (63.59)** vs GPT **EHR-CoAgent (62.57)**—not directly comparable as the same *named* strategy (different prompt realizations still share templates).
- **Best GPT accuracy:** EHR-CoAgent **80.93**; **best Gemma accuracy:** Zero-Shot+ **80.97**.
- No **bootstrap CIs** by default; treat sub–percentage-point swings between models as **indicative**, not definitive.

---

## 6. Intra-family comparisons

### 6.1 Intra-ML — fully supervised

**Ranking by F1 (primary balance of precision/recall on minority positive class):**

1. Logistic Regression — **53.27**
2. Decision Tree — 51.15
3. Random Forest — 35.85

**Ranking by accuracy:**

1. **Logistic Regression — 77.05**
2. Random Forest — 76.44
3. Decision Tree — 73.48

**Sensitivity vs specificity trade-off:**

- **Random Forest** maximizes **specificity (96.24%)** at the cost of **very low sensitivity (24.01%)** and weak F1.
- **Logistic regression** offers the **best overall** mix among the three under full supervision.

**Takeaway:** Under full supervision, **logistic regression** on bag-of-codes is the strongest classical baseline in both ACC and F1.

---

### 6.2 Intra-ML — few-shot (N=6) vs fully supervised

**Δ = few_shot − fully_supervised** (same model):

| Model | Δ ACC | Δ Sens | Δ Spec | Δ F1 |
|--------|-------|--------|--------|------|
| Decision Tree | −12.08 | +5.86 | −18.85 | −6.63 |
| Logistic Regression | **−41.78** | +42.61 | −73.63 | −9.93 |
| Random Forest | **−41.49** | +68.52 | −83.03 | +7.96 |

**Takeaway:** Six labeled rows **destabilize** fitted sparse classifiers; this is **not** analogous to LLM in-context few-shot.

---

### 6.3 Intra-LLM — GPT-4o-mini: ranking and deltas vs zero-shot

**Ranking by F1:** EHR-CoAgent (62.57) → Zero-Shot (62.24) → Few-Shot (61.93) → Zero-Shot+ (61.54).

**Ranking by ACC:** EHR-CoAgent (80.93) → Zero-Shot (80.82) → Zero-Shot+ (80.73) → Few-Shot (80.65).

**Δ vs zero-shot (percentage points):**

| Approach | Δ ACC | Δ Sens | Δ Spec | Δ F1 |
|----------|-------|--------|--------|------|
| Zero-Shot+ | −0.09 | −1.47 | +0.44 | −0.70 |
| Few-Shot (N=6) | −0.17 | −0.30 | −0.11 | −0.31 |
| EHR-CoAgent | **+0.11** | **+0.52** | −0.05 | **+0.33** |

**Takeaway:** GPT-4o-mini modes sit in a **tight band**; Co-Agent is **marginally** best on ACC/F1/sensitivity in this run.

---

### 6.4 Intra-LLM — Gemma 4: ranking and deltas vs zero-shot

**Ranking by F1:** Zero-Shot+ (**63.59**) → EHR-CoAgent (63.05) → Few-Shot (61.43) → Zero-Shot (60.83).

**Ranking by ACC:** Zero-Shot+ (**80.97**) → Zero-Shot (80.26) → Few-Shot (80.10) → EHR-CoAgent (79.78).

**Δ vs zero-shot (percentage points):**

| Approach | Δ ACC | Δ Sens | Δ Spec | Δ F1 |
|----------|-------|--------|--------|------|
| Zero-Shot+ | **+0.71** | **+4.68** | −0.78 | **+2.76** |
| Few-Shot (N=6) | −0.16 | +1.90 | −0.94 | +0.60 |
| EHR-CoAgent | −0.48 | **+7.03** | −3.32 | +2.22 |

**Takeaway:** For Gemma 4, **Zero-Shot+** clearly **lifts sensitivity and F1** vs plain zero-shot; **Co-Agent** maximizes **sensitivity** but **reduces specificity** vs the three baselines. Single calibration/critic realization (job **46615354**).

---

## 7. Inter-family comparisons (ML vs LLM)

### 7.1 Best fully supervised ML vs best LLM per model

Compared to **fully supervised logistic regression** (best stable ML on ACC/F1):

| Metric | Logistic regression (full) | Best GPT-4o-mini | Best Gemma 4 |
|--------|----------------------------|------------------|--------------|
| **Configuration** | — | EHR-CoAgent | Zero-Shot+ (ACC/F1); Co-Agent for max Sens |
| ACC | 77.05 | **80.93** | **80.97** |
| Sensitivity | 47.73 | 58.21 | **60.61** (ZS+); **62.96** (Co-Agent) |
| Specificity | 88.11 | 89.50 | 88.66 (ZS+); 86.12 (Co-Agent) |
| F1 | 53.27 | 62.57 | **63.59** (ZS+) |

**Δ (LLM − ML)** for the listed “best” cells above:

| | GPT (CoAgent) − LR | Gemma (ZS+) − LR |
|--|-------------------|------------------|
| ACC | +3.88 | +3.92 |
| F1 | +9.30 | +10.32 |

**Note:** ML few-shot runs can show **very high sensitivity** at unusable specificity/ACC; **random forest (full)** wins specificity by rarely predicting positive.

### 7.2 LLM vs strongest “reasonable” ML baseline

Both LLM families **beat** fully supervised logistic regression on **ACC and F1** on this split, with **higher sensitivity** and **similar or slightly higher specificity** (Gemma Co-Agent excepted on specificity).

### 7.3 LLM few-shot (in-context) vs ML few-shot (parameter learning)

| | ML few-shot (best ACC: DT) | GPT Few-Shot | Gemma Few-Shot |
|--|---------------------------|--------------|----------------|
| ACC | 61.40 | **80.65** | **80.10** |
| F1 | 44.52 | **61.93** | **61.43** |

**Takeaway:** Six **in-context** examples do not collapse the LLM track the way six **training** points collapse LR/RF on bag-of-codes.

### 7.4 Modality caveat

LLM inputs are **narratives**; ML uses **code tokens**. Interpret as **system** comparison, not a controlled ablation of model family alone.

---

## 8. Summary matrix (all fourteen configurations)

Sorted by **F1** descending. **Model** identifies the LLM snapshot; ML rows unchanged from `mimiciii`.

| Rank | Model | Approach | ACC | Sens | Spec | F1 |
|------|--------|----------|-----|------|------|-----|
| 1 | gemma-4-e4b-it | Zero-Shot+ | 80.97 | 60.61 | 88.66 | **63.59** |
| 2 | gemma-4-e4b-it | EHR-CoAgent | 79.78 | 62.96 | 86.12 | 63.05 |
| 3 | gpt-4o-mini | EHR-CoAgent | 80.93 | 58.21 | 89.50 | 62.57 |
| 4 | gpt-4o-mini | Zero-Shot | 80.82 | 57.69 | 89.55 | 62.24 |
| 5 | gpt-4o-mini | Few-Shot (N=6) | 80.65 | 57.39 | 89.44 | 61.93 |
| 6 | gpt-4o-mini | Zero-Shot+ | 80.73 | 56.22 | 89.99 | 61.54 |
| 7 | gemma-4-e4b-it | Few-Shot (N=6) | 80.10 | 57.83 | 88.50 | 61.43 |
| 8 | gemma-4-e4b-it | Zero-Shot | 80.26 | 55.93 | 89.44 | 60.83 |
| 9 | logistic_regression | Fully Supervised | 77.05 | 47.73 | 88.11 | 53.27 |
| 10 | random_forest | Fully Supervised | 76.44 | 24.01 | 96.24 | 35.85 |
| 11 | decision_tree | Fully Supervised | 73.48 | 50.66 | 82.09 | 51.15 |
| 12 | decision_tree | Few Shot | 61.40 | 56.52 | 63.24 | 44.52 |
| 13 | logistic_regression | Few Shot | 35.27 | 90.34 | 14.48 | 43.34 |
| 14 | random_forest | Few Shot | 34.95 | 92.53 | 13.21 | 43.81 |

---

## 9. Conclusions

1. **Intra-ML:** Full-data **logistic regression** is the best overall classical method on ACC/F1; **N=6 ML training** is not viable for LR/RF on this feature setup.
2. **Intra-LLM (GPT):** Four prompting strategies are **similar**; **EHR-CoAgent** is marginally best on ACC/F1/sensitivity.
3. **Intra-LLM (Gemma):** **Zero-Shot+** and **Co-Agent** move **sensitivity/F1** more than GPT’s zero-shot baseline; **Co-Agent** trades **specificity** for recall.
4. **Cross-LLM:** On this split, Gemma **Zero-Shot+** achieves the **highest F1** among all listed LLM configurations; GPT **Co-Agent** leads **GPT** family on ACC/F1. Differences are **modest** and **not significance-tested**.
5. **Inter ML vs LLM:** Both LLM families **beat** the best fully supervised ML baseline on **ACC and F1** with **higher sensitivity**; modality differs from bag-of-codes ML.
6. **Reporting:** Default **no bootstrap CIs** (`evaluation.bootstrap_ci: false`).

---

## 10. Artifact map

| Artifact | Contents |
|----------|----------|
| `data/outputs/mimiciii/summary_table.csv` | Combined ML + GPT LLM table (Section 8 partial) |
| `data/outputs/mimiciii/ml_results_fully_supervised.csv` | ML full-training metrics + confusion |
| `data/outputs/mimiciii/ml_results_few_shot.csv` | ML N=6 metrics + confusion |
| `data/outputs/mimiciii/llm_*_metrics.json` | GPT-4o-mini per-mode metrics |
| `data/outputs/mimiciii_gemma/llm_*_metrics.json` | Gemma 4 per-mode metrics (incl. `llm_coagent_metrics.json`) |
| `docs/MIMICIII_GEMMA_JOB_46510786_RESULTS_REPORT.md` | Slurm jobs **46510786** / **46615354** notes |

---

*Report reflects `data/outputs/mimiciii/` and `data/outputs/mimiciii_gemma/` at the time of writing. Rerun pipelines and refresh if configs, data, or prompts change.*
