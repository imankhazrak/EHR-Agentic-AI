# MIMIC-III Next-Visit Prediction Report

## 1. Overview

This report summarizes the work completed on the MIMIC-III next-visit prediction project, including methodology, experiments, and key results. The primary objective was to predict whether a patient will be diagnosed with Disorders of Lipid Metabolism in their next hospital admission using only information from the current visit.

The project compares two main approaches:
- Classical Machine Learning (ML)
- Large Language Models (LLMs)

> **LLM results attribution (current status):**
> - All LLM results in this document currently correspond to **OpenAI `gpt-4o-mini`** runs.
> - **Gemma4** experiments are currently running and their results are **not included yet**.
> - When Gemma4 outputs are ready, they should be added as a separately labeled section/table.

> **Naming convention (best practice):**
> - Use model-scoped output roots: `data/outputs/mimiciii_gpt4o_mini/` and `data/outputs/mimiciii_gemma4/`.
> - Use model-tagged result filenames in shared folders, for example: `summary_table_gpt4o_mini.csv`, `summary_table_gemma4.csv`.
> - In every table/header, include both **model family** and **version/snapshot** (for example `gpt-4o-mini-2024-07-18`, `gemma-4-<version>`).
> - Never merge metrics from different models into one unlabeled table row; keep one row per model+mode.

![](image/image.png)

---

## 2. Problem Definition

- **Task:** Binary classification
- **Target:** Presence of lipid metabolism disorder (ICD-9 272.x) in the next visit
- **Input:** Current visit only (no leakage from future visit)
- **Dataset:** MIMIC-III hospital admissions (visit pairs)

### What Exactly Is Predicted (and What It Means)

- The model predicts whether the **immediate next hospital admission** for the same patient will contain a lipid-disorder diagnosis code (`272.x`, stored without dot in MIMIC-III).
- This is a prediction of **coded diagnosis occurrence on the next admission**, not a direct estimate of true biological dyslipidemia burden or long-term cardiovascular risk.
- The label is built from `diagnoses_codes_next` in each adjacent visit pair (current visit -> next visit).

### Is the Horizon Next Day / Week / Month?

- The current pipeline does **not** define a fixed temporal horizon (for example, 7-day, 30-day, or 90-day prediction).
- Instead, horizon is **event-based**: "next recorded admission" in chronological order for that patient.
- Therefore, the gap between current and next visit can vary substantially across patients (short for some, long for others).
- If a patient has only one recorded visit, no pair is created and that case is excluded from this next-visit task.

In short: this is a **next-admission prediction task**, not a fixed-time-window prediction task.

### Observed Data Distribution and Class Balance

Based on output artifacts in `data/outputs/mimiciii_gpt4o_mini/`:

- `label_distribution.csv`: negative = 9,040, positive = 3,416 (total = 12,456 pairs)
- `target_label_report.json`: prevalence = 0.2742 (~27.4% positive)
- `processed_dataset_summary.json`: train prevalence = 0.2743, test prevalence = 0.2741

Interpretation:

- The dataset is **imbalanced** (roughly **72.6% negative** vs **27.4% positive**).
- This is a **moderate class imbalance** (not extreme), and stratified splitting preserved a nearly identical positive rate in train and test.

---

## 3. Methodology

### 3.1 Data Preparation

- Constructed **visit pairs** (current visit → next visit)
- Created binary labels based on next visit diagnosis codes
- Split dataset into:
  - 80% training
  - 20% testing (stratified)

### 3.2 Input Representations

#### Classical ML
- Bag-of-codes representation
- Features extracted from:
  - Diagnoses
  - Procedures
  - Medications

#### LLM
- Natural language narrative per visit
- Includes:
  - Diagnoses
  - Medications
  - Procedures

---

## 4. Models and Approaches

### 4.1 Classical ML Models

- Decision Tree
- Logistic Regression
- Random Forest

Two training strategies:
- Fully supervised (full dataset)
- Few-shot ML (trained on only 6 samples)

### 4.2 LLM Approaches

Using GPT-based model (`gpt-4o-mini` for current reported runs):

- Zero-shot
- Zero-shot+
- Few-shot (in-context examples)
- EHR-CoAgent (few-shot + critic feedback)

---

## 5. Evaluation Metrics

- Accuracy (ACC)
- Sensitivity (Recall)
- Specificity
- F1-score

---

## 6. Results Summary

### 6.1 Classical ML (Fully Supervised)

- Best model: Logistic Regression
- Accuracy: ~77%
- F1-score: ~53%

Observations:
- Balanced performance
- Random Forest had high specificity but very low sensitivity

### 6.2 Classical ML (Few-Shot)

- Severe performance degradation
- Accuracy dropped to ~35–61%

Observations:
- Models overfit or became unstable
- Extremely high sensitivity but very low specificity

---

### 6.3 LLM Results (`gpt-4o-mini`)

All reported LLM approaches (for `gpt-4o-mini`) performed similarly:

| Approach | Accuracy | F1 |
|----------|--------|----|
| Zero-shot | ~80.8% | ~62.2% |
| Zero-shot+ | ~80.7% | ~61.5% |
| Few-shot | ~80.6% | ~61.9% |
| CoAgent | ~80.9% | ~62.6% |

Observations:
- Very stable across prompt variations
- Minimal differences between modes
- Near-zero parsing errors

---

## 7. Key Findings

### 7.1 ML vs LLM

- LLM outperformed classical ML in:
  - Accuracy (+3–4%)
  - F1-score (+9%)
  - Sensitivity (significantly higher)

- ML (Random Forest) achieved higher specificity but missed many positives

### 7.2 Few-Shot Behavior

- **ML few-shot:** Failed due to insufficient data
- **LLM few-shot:** Remained stable due to in-context learning

### 7.3 Prompt Engineering Impact

- Zero-shot, few-shot, and CoAgent produced very similar results
- CoAgent showed only marginal improvement

---

## 8. Interpretation

- LLMs effectively leverage structured clinical narratives
- Narrative representation may capture richer context than bag-of-codes
- Prompt engineering has limited impact compared to model capability

---

## 9. Limitations

- Data split at visit level (possible patient leakage)
- No statistical significance testing
- LLM inputs differ from ML inputs (not strictly controlled comparison)
- Label reflects coding behavior, not true clinical diagnosis

---

## 10. Conclusion

This project demonstrates that LLM-based approaches provide strong and stable performance for next-visit clinical prediction tasks. Compared to classical ML baselines, LLMs achieved better overall predictive performance, especially in detecting positive cases.

However, improvements from advanced prompting strategies (few-shot, CoAgent) were marginal, suggesting that model capability and input representation are the dominant factors.

---

## 11. Future Work

- Instead of relying only on prompt engineering and repeated LLM API calls, a stronger next step is to fine-tune a new open-source **Gemma 4** model.
- This direction can reduce inference latency and long-term cost compared with repeated prompt-based calls to proprietary APIs.
- Because **Gemma 4** is multimodal, it is a promising option for extending this work to **MIMIC-IV**, which contains richer and more multidimensional data sources.
- In a future version of the project, we can design a unified framework that leverages structured clinical variables, text, and other available modalities from MIMIC-IV within one fine-tuned open-source model.
- This would make the system more scalable, more customizable, and potentially better suited for real-world clinical prediction tasks.

---

