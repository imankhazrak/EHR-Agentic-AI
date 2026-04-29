# MIMIC-III Multitask Test Label Distribution Report

This report summarizes the distribution of all multitask labels for the test split.

Source files:
- `data/processed/mimiciii_multitask/test.csv`
- `data/processed/mimiciii_multitask/label_stats.csv`

## Dataset Scope

- Split: `test`
- Total rows: `2492`
- Missing labels: `0` for every multitask label

## Label Distribution (All Multitask Labels, Test Split)

| Label | Positive (n) | Negative (n) | Positive Rate |
|---|---:|---:|---:|
| `label_lipid_disorder` | 683 | 1809 | 0.274077 |
| `label_lipid_next` | 683 | 1809 | 0.274077 |
| `label_diabetes_current` | 825 | 1667 | 0.331059 |
| `label_hypertension_current` | 1354 | 1138 | 0.543339 |
| `label_obesity_current` | 139 | 2353 | 0.055778 |
| `label_cardio_next` | 1246 | 1246 | 0.500000 |
| `label_kidney_next` | 1088 | 1404 | 0.436597 |
| `label_stroke_next` | 147 | 2345 | 0.058989 |

## Quick Notes

- `label_hypertension_current` has the highest positive rate (`54.33%`).
- `label_cardio_next` is exactly balanced (`50.00%`).
- `label_obesity_current` and `label_stroke_next` are strongly imbalanced (about `5-6%` positive).
- `label_lipid_disorder` and `label_lipid_next` have identical distributions in this split.
