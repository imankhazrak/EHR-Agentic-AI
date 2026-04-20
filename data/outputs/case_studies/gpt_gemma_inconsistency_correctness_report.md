# GPT-Gemma Inconsistency and Correctness Report

## Objective

This report evaluates label consistency between GPT-4o-mini (Few-Shot) and Gemma4 (CoAgent), and checks correctness against `true_label`.

Input file:

- `data/outputs/case_studies/model_comparison_gemma_coagent_vs_gpt4omini_fewshot.csv`

Analysis output file:

- `data/outputs/case_studies/gpt_gemma_inconsistency_analysis.csv`

## Metric Definitions

- **Inconsistent to each other**: `gpt_label != gemma_label`
- **Consistent**: `gpt_label == gemma_label`
- **Corrected and same as true label**: both models are correct  
  (`gpt_label == true_label` and `gemma_label == true_label`)
- **Consistent and equal to true label**: models are consistent and shared label equals `true_label`

## Summary Counts

Total cases analyzed: **2492**

- Inconsistent to each other: **186** (7.46%)
- Consistent: **2306** (92.54%)
- Corrected and same as true label (both correct): **1784** (71.59%)
- Consistent and equal to true label: **1784** (71.59%)

## Additional Context

- GPT correct vs true label: **1883** (75.56%)
- Gemma correct vs true label: **1871** (75.08%)

Because “both correct” requires both models to match truth, that count equals the
“consistent and equal to true label” count in this dataset.

## Interpretation

- The two models agree on most cases (over 92%).
- About 7.5% of cases are direct model disagreements.
- Roughly 71.6% of all cases are jointly correct (both models match ground truth).
- GPT and Gemma have very similar individual correctness, with GPT slightly higher by 12 cases.

## Per-Case CSV Schema

`data/outputs/case_studies/gpt_gemma_inconsistency_analysis.csv` includes:

- `case_id`
- `true_label`
- `gpt_label`
- `gemma_label`
- `models_inconsistent` (0/1)
- `models_consistent` (0/1)
- `gpt_correct` (0/1)
- `gemma_correct` (0/1)
- `both_correct` (0/1)
- `consistent_and_true` (0/1)
