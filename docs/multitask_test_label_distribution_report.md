# Multitask Test Dataset Label Distribution Report

This report summarizes label distribution for the dataset used in the multitask job, based on:

- `data/processed/mimiciii_multitask/test.csv`
- `data/processed/mimiciii_multitask/label_stats.csv`

## Dataset Scope

- **Split:** `test`
- **Total rows:** `2492`
- **Missing labels:** `0` for all multitask targets

## Label Distribution (Test Split)

| Label | Positive (n) | Negative (n) | Positive Rate |
|---|---:|---:|---:|
| `label_lipid_next` | 683 | 1809 | 0.274077 |
| `label_diabetes_current` | 825 | 1667 | 0.331059 |
| `label_hypertension_current` | 1354 | 1138 | 0.543339 |
| `label_obesity_current` | 139 | 2353 | 0.055778 |
| `label_cardio_next` | 1246 | 1246 | 0.500000 |
| `label_kidney_next` | 1088 | 1404 | 0.436597 |
| `label_stroke_next` | 147 | 2345 | 0.058989 |

## Notes for Clinical/Model Interpretation

- `label_cardio_next` is perfectly balanced in the test split (`50%` positive).
- `label_hypertension_current` is moderately positive-skewed (`54.33%` positive).
- `label_obesity_current` and `label_stroke_next` are highly imbalanced (both around `~5-6%` positive).
- The remaining tasks (`lipid`, `diabetes`, `kidney`) are imbalanced toward negatives to varying degrees.

## Source Traceability

Values in this report were taken directly from `label_stats.csv` rows where `split == test`.
