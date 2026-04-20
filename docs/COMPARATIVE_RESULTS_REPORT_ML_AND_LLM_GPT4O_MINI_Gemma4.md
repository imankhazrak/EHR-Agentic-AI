# Comparative Results Report — ML vs LLM (GPT-4o-mini + Gemma 4, Prompt V1/V2)

This report compares classical ML baselines with LLM prompt strategies for MIMIC-III lipid-disorder next-visit prediction.

It now includes:

- GPT-4o-mini **Prompt V1**
- GPT-4o-mini **Prompt V2**
- Gemma 4 **Prompt V1**
- Gemma 4 **Prompt V2** (added from `docs/LLM_EXPERIMENT_REPORT_Gemma4_PROMPT_V2.md`)

---

## 1) Scope and Evaluation Setup

| Item | Detail |
|------|--------|
| Task | Binary prediction of lipid-metabolism disorder (`272.x` / CCS-53) on next admission |
| Test split | Stratified hold-out, seed 42 |
| Core metrics | Accuracy, Sensitivity (Recall+), Specificity, F1 |
| LLM eval size | `n_total = 2492` across all compared LLM prompt modes |
| Input caveat | ML uses bag-of-codes; LLM uses narrative text |

Because features differ (codes vs narrative), treat this as an **end-to-end system comparison**, not a strict model-only ablation.

---

## 2) Classical ML Baselines

### 2.1 Fully Supervised

| Model | ACC | Sens | Spec | F1 |
|------|-----:|-----:|-----:|---:|
| Decision Tree | 73.48 | 50.66 | 82.09 | 51.15 |
| Logistic Regression | **77.05** | 47.73 | 88.11 | **53.27** |
| Random Forest | 76.44 | 24.01 | **96.24** | 35.85 |

### 2.2 ML Few-Shot (6 training examples)

| Model | ACC | Sens | Spec | F1 |
|------|-----:|-----:|-----:|---:|
| Decision Tree | **61.40** | 56.52 | 63.24 | **44.52** |
| Logistic Regression | 35.27 | 90.34 | 14.48 | 43.34 |
| Random Forest | 34.95 | 92.53 | 13.21 | 43.81 |

**ML takeaway:** Fully supervised logistic regression is the strongest stable baseline; ML few-shot training is unstable at `N=6`.

---

## 3) LLM Results by Model + Prompt Version

### 3.1 GPT-4o-mini — Prompt V1

| Approach | ACC | Sens | Spec | F1 |
|----------|----:|-----:|-----:|---:|
| Zero-Shot | 80.82 | 57.69 | 89.55 | 62.24 |
| Zero-Shot+ | 80.73 | 56.22 | **89.99** | 61.54 |
| Few-Shot | 80.65 | 57.39 | 89.44 | 61.93 |
| EHR-CoAgent | **80.93** | **58.21** | 89.50 | **62.57** |

### 3.2 GPT-4o-mini — Prompt V2

| Approach | ACC | Sens | Spec | F1 |
|----------|----:|-----:|-----:|---:|
| Zero-Shot | 73.07 | **83.60** | 69.10 | 62.99 |
| Zero-Shot+ | 73.27 | 83.31 | 69.49 | 63.08 |
| Few-Shot | 75.56 | 77.75 | 74.74 | 63.55 |
| EHR-CoAgent | **76.00** | 77.89 | **75.29** | **64.02** |

### 3.3 Gemma 4 — Prompt V1

| Approach | ACC | Sens | Spec | F1 |
|----------|----:|-----:|-----:|---:|
| Zero-Shot | 80.26 | 55.93 | 89.44 | 60.83 |
| Zero-Shot+ | **80.97** | 60.61 | 88.66 | **63.59** |
| Few-Shot | 80.10 | 57.83 | 88.50 | 61.43 |
| EHR-CoAgent | 79.78 | **62.96** | 86.12 | 63.05 |

### 3.4 Gemma 4 — Prompt V2 (newly added)

| Approach | ACC | Sens | Spec | F1 | AUC | AUPRC |
|----------|----:|-----:|-----:|---:|----:|------:|
| Zero-Shot | 74.92 | 79.36 | 73.24 | 63.43 | **80.78** | **57.41** |
| Zero-Shot+ | 73.80 | 79.94 | 71.48 | 62.58 | 78.31 | 51.12 |
| Few-Shot | 70.95 | **80.82** | 67.22 | 60.39 | 74.11 | 44.37 |
| EHR-CoAgent | **75.08** | 80.67 | **72.97** | **63.96** | 79.54 | 53.25 |

**Prompt V2 Gemma takeaway:** CoAgent is best on fixed-threshold metrics (ACC/F1/balanced tradeoff), while Zero-Shot is best on ranking quality (AUC/AUPRC).

---

## 4) Readable Cross-Track Summary

### 4.1 Best configuration per track

| Track | Best Setup | ACC | Sens | Spec | F1 |
|------|------------|----:|-----:|-----:|---:|
| ML (fully supervised) | Logistic Regression | 77.05 | 47.73 | 88.11 | 53.27 |
| GPT-4o-mini Prompt V1 | EHR-CoAgent | **80.93** | 58.21 | 89.50 | 62.57 |
| GPT-4o-mini Prompt V2 | EHR-CoAgent | 76.00 | **77.89** | 75.29 | **64.02** |
| Gemma 4 Prompt V1 | Zero-Shot+ | **80.97** | 60.61 | 88.66 | 63.59 |
| Gemma 4 Prompt V2 | EHR-CoAgent | 75.08 | 80.67 | 72.97 | 63.96 |

### 4.2 Highest values across all LLM tracks

- **Highest ACC:** Gemma 4 Prompt V1 Zero-Shot+ (**80.97**)
- **Highest Sensitivity:** GPT-4o-mini Prompt V2 Zero-Shot (**83.60**)
- **Highest Specificity:** GPT-4o-mini Prompt V1 Zero-Shot+ (**89.99**)
- **Highest F1:** GPT-4o-mini Prompt V2 EHR-CoAgent (**64.02**)
- **Highest AUC/AUPRC (reported in Gemma V2 report):** Gemma 4 Prompt V2 Zero-Shot (**80.78 / 57.41**)

---

## 5) Key Interpretations

1. **LLMs consistently outperform ML baseline on F1** in this setup.
2. **GPT Prompt V2 shifts strongly toward recall**, increasing sensitivity while reducing specificity and ACC.
3. **Gemma Prompt V1 is the strongest for high ACC/high SPEC operation** (notably Zero-Shot+).
4. **Gemma Prompt V2 CoAgent gives balanced threshold behavior**, while **Gemma Prompt V2 Zero-Shot** gives strongest score-ranking quality (AUC/AUPRC).
5. Choice of prompt family changes operating point significantly; select by clinical objective (miss fewer positives vs control false alarms).

---

<!--
## 6) Artifact Map

| Artifact | Description |
|----------|-------------|
| `data/outputs/mimiciii/ml_results_fully_supervised.csv` | Classical ML full-training results |
| `data/outputs/mimiciii/ml_results_few_shot.csv` | Classical ML few-shot training results |
| `data/outputs/mimiciii/llm_*_metrics.json` | GPT-4o-mini Prompt V1 metrics |
| `data/outputs/mimiciii_llm_gpt4o_mini_promptv2/llm_*_metrics.json` | GPT-4o-mini Prompt V2 metrics |
| `data/outputs/mimiciii_gemma/llm_*_metrics.json` | Gemma 4 Prompt V1 metrics |
| `data/outputs/mimiciii_llm_promptv2_gemma/llm_*_metrics.json` | Gemma 4 Prompt V2 metrics |
| `docs/LLM_EXPERIMENT_REPORT_Gemma4_PROMPT_V2.md` | Detailed Gemma 4 Prompt V2 report |

---
-->

## 7) Final Consolidated Table (All Models, All Experiments)

`NA` means the metric was not available in the experiment artifact used for this report.

| Family | Model | Prompt Version | Approach | ACC | Sens | Spec | F1 | AUC | AUPRC |
|--------|-------|----------------|----------|----:|-----:|-----:|---:|----:|------:|
| ML | Decision Tree | NA | Fully Supervised | 73.48 | 50.66 | 82.09 | 51.15 | NA | NA |
| ML | Logistic Regression | NA | Fully Supervised | 77.05 | 47.73 | 88.11 | 53.27 | NA | NA |
| ML | Random Forest | NA | Fully Supervised | 76.44 | 24.01 | 96.24 | 35.85 | NA | NA |
| ML | Decision Tree | NA | Few-Shot (N=6) | 61.40 | 56.52 | 63.24 | 44.52 | NA | NA |
| ML | Logistic Regression | NA | Few-Shot (N=6) | 35.27 | 90.34 | 14.48 | 43.34 | NA | NA |
| ML | Random Forest | NA | Few-Shot (N=6) | 34.95 | 92.53 | 13.21 | 43.81 | NA | NA |
| LLM | gpt-4o-mini-2024-07-18 | Prompt V1 | Zero-Shot | 80.82 | 57.69 | 89.55 | 62.24 | NA | NA |
| LLM | gpt-4o-mini-2024-07-18 | Prompt V1 | Zero-Shot+ | 80.73 | 56.22 | 89.99 | 61.54 | NA | NA |
| LLM | gpt-4o-mini-2024-07-18 | Prompt V1 | Few-Shot | 80.65 | 57.39 | 89.44 | 61.93 | NA | NA |
| LLM | gpt-4o-mini-2024-07-18 | Prompt V1 | EHR-CoAgent | 80.93 | 58.21 | 89.50 | 62.57 | NA | NA |
| LLM | gpt-4o-mini-2024-07-18 | Prompt V2 | Zero-Shot | 73.07 | 83.60 | 69.10 | 62.99 | 81.07 | 55.67 |
| LLM | gpt-4o-mini-2024-07-18 | Prompt V2 | Zero-Shot+ | 73.27 | 83.31 | 69.49 | 63.08 | 79.58 | 52.97 |
| LLM | gpt-4o-mini-2024-07-18 | Prompt V2 | Few-Shot | 75.56 | 77.75 | 74.74 | 63.55 | 78.25 | 52.58 |
| LLM | gpt-4o-mini-2024-07-18 | Prompt V2 | EHR-CoAgent | 76.00 | 77.89 | 75.29 | 64.02 | 78.38 | 52.46 |
| LLM | google/gemma-4-e4b-it | Prompt V1 | Zero-Shot | 80.26 | 55.93 | 89.44 | 60.83 | NA | NA |
| LLM | google/gemma-4-e4b-it | Prompt V1 | Zero-Shot+ | 80.97 | 60.61 | 88.66 | 63.59 | NA | NA |
| LLM | google/gemma-4-e4b-it | Prompt V1 | Few-Shot | 80.10 | 57.83 | 88.50 | 61.43 | NA | NA |
| LLM | google/gemma-4-e4b-it | Prompt V1 | EHR-CoAgent | 79.78 | 62.96 | 86.12 | 63.05 | NA | NA |
| LLM | google/gemma-4-e4b-it | Prompt V2 | Zero-Shot | 74.92 | 79.36 | 73.24 | 63.43 | 80.78 | 57.41 |
| LLM | google/gemma-4-e4b-it | Prompt V2 | Zero-Shot+ | 73.80 | 79.94 | 71.48 | 62.58 | 78.31 | 51.12 |
| LLM | google/gemma-4-e4b-it | Prompt V2 | Few-Shot | 70.95 | 80.82 | 67.22 | 60.39 | 74.11 | 44.37 |
| LLM | google/gemma-4-e4b-it | Prompt V2 | EHR-CoAgent | 75.08 | 80.67 | 72.97 | 63.96 | 79.54 | 53.25 |

---

# Final Model Selection Based on Balanced Sensitivity and Specificity


To determine the most appropriate model for clinical deployment, we prioritize a balanced trade-off between sensitivity (recall) and specificity, as these metrics represent complementary aspects of diagnostic performance. While several configurations achieved higher accuracy or extreme values in either sensitivity or specificity, such imbalanced behavior is not desirable in clinical settings where both false negatives and false positives carry significant consequences.

Among all evaluated models, the **GPT-4o-mini (Prompt V2, EHR-CoAgent)** configuration demonstrates the most favorable balance between sensitivity and specificity, with values of 77.89% and 75.29%, respectively. This near-symmetric performance (difference ≈ 2.6 percentage points) indicates a well-calibrated decision boundary. In addition, this model achieves the **highest F1-score (64.02%)** among all experiments, reflecting strong overall predictive performance, and maintains competitive accuracy (76.00%).

Other Prompt V2 configurations, such as GPT-4o-mini Few-Shot and Gemma 4 EHR-CoAgent, also exhibit relatively balanced behavior; however, they either show slightly lower F1-scores or greater deviation between sensitivity and specificity. In contrast, Prompt V1 configurations—despite achieving higher accuracy and specificity—consistently underperform in sensitivity, making them less suitable for applications where missing positive cases is critical.

Overall, these results indicate that **Prompt V2-based approaches shift the operating point toward a more clinically meaningful balance**, with the GPT-4o-mini EHR-CoAgent configuration emerging as the most suitable choice for deployment under a balanced sensitivity–specificity criterion.

---

## Best Candidate Models (Balanced Performance)

| Rank | Model | Prompt | Approach | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1 (%) | AUC (%) | AUPRC (%) | \|Sens−Spec\| (%) |
|------|-------|--------|----------|--------------|-----------------|-----------------|--------|---------|-----------|-------------------|
| 🥇 1 | GPT-4o-mini | V2 | EHR-CoAgent | 76.00 | 77.89 | 75.29 | **64.02** | 78.38 | 52.46 | **2.60** |
| 🥈 2 | GPT-4o-mini | V2 | Few-Shot | 75.56 | 77.75 | 74.74 | 63.55 | 78.25 | 52.58 | 3.01 |
| 🥉 3 | Gemma 4 | V2 | EHR-CoAgent | 75.08 | 80.67 | 72.97 | 63.96 | 79.54 | 53.25 | 7.70 |

---

## Interpretation

- The **top-ranked model (GPT-4o-mini, Prompt V2, EHR-CoAgent)** provides the best compromise between identifying true positive cases and avoiding false positives.
- The **small gap between sensitivity and specificity** is a key indicator of stability and reliability in decision-making.
- Higher accuracy models (e.g., Prompt V1) are excluded due to **significant imbalance**, which can bias predictions toward one class.
- Prompt V2 configurations consistently move models toward a **more balanced operating regime**, making them more appropriate for real-world clinical prediction tasks.

---

## Key Takeaway

The selection of the best model should not rely solely on accuracy or recall in isolation. Instead, a balanced evaluation reveals that **GPT-4o-mini (Prompt V2, EHR-CoAgent)** achieves the most clinically meaningful performance, making it the preferred configuration for downstream use.

