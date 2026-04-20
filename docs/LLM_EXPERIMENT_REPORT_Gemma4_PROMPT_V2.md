# LLM Experiment Report: Gemma 4 Prompt V2

## Experiment Summary

- **Model:** `models/hf_snapshots/google--gemma-4-e4b-it`
- **Dataset:** MIMIC-III evaluation split
- **Evaluation size:** `n_total = 2492` (for all four approaches)
- **Compared prompt approaches:** `zero_shot`, `zero_shot_plus`, `few_shot`, `coagent`

This report compares the four prompt strategies under the same model and evaluation size to identify the strongest approach for next-step prediction performance.

## Results by Prompt Approach

### 1) Zero-Shot

- Accuracy: **74.92**
- Precision: **52.83**
- Recall (Sensitivity): **79.36**
- Specificity: **73.24**
- F1: **63.43**
- Balanced Accuracy: **76.30**
- AUC: **80.78**
- AUPRC: **57.41**
- Confusion Matrix: TP=542, FP=484, TN=1325, FN=141
- Notes: `n_invalid_probability = 6`, all predictions parseable.

### 2) Zero-Shot-Plus

- Accuracy: **73.80**
- Precision: **51.41**
- Recall (Sensitivity): **79.94**
- Specificity: **71.48**
- F1: **62.58**
- Balanced Accuracy: **75.71**
- AUC: **78.31**
- AUPRC: **51.12**
- Confusion Matrix: TP=546, FP=516, TN=1293, FN=137
- Notes: No parse/probability issues.

### 3) Few-Shot

- Accuracy: **70.95**
- Precision: **48.21**
- Recall (Sensitivity): **80.82**
- Specificity: **67.22**
- F1: **60.39**
- Balanced Accuracy: **74.02**
- AUC: **74.11**
- AUPRC: **44.37**
- Confusion Matrix: TP=552, FP=593, TN=1216, FN=131
- Notes: Highest recall, but many false positives.

### 4) CoAgent

- Accuracy: **75.08**
- Precision: **52.98**
- Recall (Sensitivity): **80.67**
- Specificity: **72.97**
- F1: **63.96**
- Balanced Accuracy: **76.82**
- AUC: **79.54**
- AUPRC: **53.25**
- Confusion Matrix: TP=551, FP=489, TN=1320, FN=132
- Notes: Strong balance between sensitivity and specificity.

## Comparative Ranking

### By threshold-dependent classification quality (Accuracy/F1/Balanced Accuracy)

1. **CoAgent** (best overall threshold metrics)
2. **Zero-Shot**
3. **Zero-Shot-Plus**
4. **Few-Shot**

### By ranking/discrimination quality (AUC/AUPRC)

1. **Zero-Shot** (best AUC and AUPRC)
2. **CoAgent**
3. **Zero-Shot-Plus**
4. **Few-Shot**

## Key Takeaways

- **Best operating-point performance:** `coagent` (top Accuracy, F1, Balanced Accuracy).
- **Best probability ranking quality:** `zero_shot` (top AUC, AUPRC).
- `few_shot` emphasizes recall but introduces substantial false positives (lowest precision/specificity).
- `zero_shot_plus` improves over `few_shot` but is consistently below `zero_shot` and `coagent`.

## Recommendation

- Use **`coagent`** as the default production-style prompt mode when decisions are made at a fixed threshold.
- Use **`zero_shot`** when score ranking/calibration quality is the primary objective (e.g., triage pipelines with downstream threshold tuning).

## Source Metrics Files

- `data/outputs/mimiciii_llm_promptv2_gemma/llm_zero_shot_metrics.json`
- `data/outputs/mimiciii_llm_promptv2_gemma/llm_zero_shot_plus_metrics.json`
- `data/outputs/mimiciii_llm_promptv2_gemma/llm_few_shot_metrics.json`
- `data/outputs/mimiciii_llm_promptv2_gemma/llm_coagent_metrics.json`
