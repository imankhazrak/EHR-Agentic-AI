# Gemma-4 Prompt_v3 Multitask Experiment Report

Date: 2026-05-05  
Model: `models/hf_snapshots/google--gemma-4-e4b-it`  
Dataset split: `data/processed/mimiciii_multitask/test.csv`  
Output dir: `data/outputs/mimiciii_llm_gemma4_promptv3_multitask_test`

## Scope

This report summarizes multitask results for completed prompt approaches:

- Zero-shot (`m3_g4_p3_zs`, job `47281874`)
- Zero-shot+ (`m3_g4_p3_zsp`, job `47281875`)
- Few-shot (`m3_g4_p3_fs`, job `47281876`)

CoAgent (`m3_g4_p3_ca`, job `47281877`) was still running at report generation time, so it is not included in the finalized metric tables below.

## Metrics Definition

Per task, metrics are computed on rows with a valid parsed prediction (`n_valid`) using:

- Accuracy (ACC)
- Precision (P)
- Recall / Sensitivity (R)
- F1
- Specificity (Spec)
- AUC
- AUPRC

## Zero-shot Results

| Task | n_valid | ACC | P | R | F1 | Spec | AUC | AUPRC | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lipid | 2453 | 76.60 | 55.30 | 69.98 | 61.78 | 79.05 | 74.40 | 46.77 | 464 | 375 | 1415 | 199 |
| diabetes | 2453 | 94.13 | 84.91 | 100.00 | 91.84 | 91.24 | 94.80 | 91.77 | 810 | 144 | 1499 | 0 |
| hypertension | 2453 | 95.96 | 93.09 | 100.00 | 96.42 | 91.16 | 93.60 | 93.88 | 1333 | 99 | 1021 | 0 |
| obesity | 2453 | 99.47 | 91.33 | 100.00 | 95.47 | 99.44 | 96.10 | 86.71 | 137 | 13 | 2303 | 0 |
| cardio | 2453 | 75.87 | 74.53 | 78.20 | 76.32 | 73.56 | 77.00 | 72.57 | 954 | 326 | 907 | 266 |
| kidney | 2453 | 73.22 | 71.02 | 65.11 | 67.94 | 79.48 | 71.24 | 61.66 | 696 | 284 | 1100 | 373 |
| stroke | 2453 | 88.83 | 12.57 | 15.38 | 13.84 | 93.38 | 53.90 | 6.67 | 22 | 153 | 2157 | 121 |

## Zero-shot+ Results

| Task | n_valid | ACC | P | R | F1 | Spec | AUC | AUPRC | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lipid | 2401 | 79.13 | 59.95 | 67.65 | 63.56 | 83.36 | 75.71 | 49.70 | 437 | 292 | 1463 | 209 |
| diabetes | 2401 | 90.55 | 77.57 | 100.00 | 87.37 | 85.95 | 97.96 | 95.27 | 785 | 227 | 1389 | 0 |
| hypertension | 2401 | 93.84 | 89.76 | 100.00 | 94.61 | 86.58 | 96.24 | 96.13 | 1298 | 148 | 955 | 0 |
| obesity | 2401 | 99.54 | 92.62 | 100.00 | 96.17 | 99.51 | 99.00 | 90.68 | 138 | 11 | 2252 | 0 |
| cardio | 2401 | 72.55 | 76.95 | 64.16 | 69.98 | 80.90 | 73.34 | 69.78 | 768 | 230 | 974 | 429 |
| kidney | 2401 | 72.05 | 72.37 | 58.38 | 64.63 | 82.68 | 70.62 | 62.06 | 613 | 234 | 1117 | 437 |
| stroke | 2401 | 89.50 | 9.93 | 10.07 | 10.00 | 94.39 | 52.48 | 6.19 | 14 | 127 | 2135 | 125 |

## Few-shot Results

| Task | n_valid | ACC | P | R | F1 | Spec | AUC | AUPRC | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lipid | 2490 | 68.31 | 45.52 | 78.92 | 57.74 | 64.31 | 72.47 | 43.00 | 539 | 645 | 1162 | 144 |
| diabetes | 2490 | 99.16 | 97.52 | 100.00 | 98.74 | 98.74 | 92.97 | 89.93 | 825 | 21 | 1644 | 0 |
| hypertension | 2490 | 98.11 | 97.11 | 99.48 | 98.28 | 96.48 | 96.52 | 96.16 | 1346 | 40 | 1097 | 7 |
| obesity | 2490 | 99.40 | 90.26 | 100.00 | 94.88 | 99.36 | 98.22 | 86.94 | 139 | 15 | 2336 | 0 |
| cardio | 2490 | 74.38 | 74.28 | 74.64 | 74.46 | 74.12 | 75.14 | 70.37 | 930 | 322 | 922 | 316 |
| kidney | 2490 | 72.33 | 68.80 | 67.10 | 67.94 | 76.39 | 72.47 | 63.75 | 730 | 331 | 1071 | 358 |
| stroke | 2490 | 89.00 | 11.52 | 12.93 | 12.18 | 93.77 | 52.53 | 6.56 | 19 | 146 | 2197 | 128 |

## Comparative Table (All Completed Approaches, All Tasks)

| Approach | Task | n_valid | ACC | P | R | F1 | Spec | AUC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_shot | lipid | 2453 | 76.60 | 55.30 | 69.98 | 61.78 | 79.05 | 74.40 | 46.77 |
| zero_shot | diabetes | 2453 | 94.13 | 84.91 | 100.00 | 91.84 | 91.24 | 94.80 | 91.77 |
| zero_shot | hypertension | 2453 | 95.96 | 93.09 | 100.00 | 96.42 | 91.16 | 93.60 | 93.88 |
| zero_shot | obesity | 2453 | 99.47 | 91.33 | 100.00 | 95.47 | 99.44 | 96.10 | 86.71 |
| zero_shot | cardio | 2453 | 75.87 | 74.53 | 78.20 | 76.32 | 73.56 | 77.00 | 72.57 |
| zero_shot | kidney | 2453 | 73.22 | 71.02 | 65.11 | 67.94 | 79.48 | 71.24 | 61.66 |
| zero_shot | stroke | 2453 | 88.83 | 12.57 | 15.38 | 13.84 | 93.38 | 53.90 | 6.67 |
| zero_shot_plus | lipid | 2401 | 79.13 | 59.95 | 67.65 | 63.56 | 83.36 | 75.71 | 49.70 |
| zero_shot_plus | diabetes | 2401 | 90.55 | 77.57 | 100.00 | 87.37 | 85.95 | 97.96 | 95.27 |
| zero_shot_plus | hypertension | 2401 | 93.84 | 89.76 | 100.00 | 94.61 | 86.58 | 96.24 | 96.13 |
| zero_shot_plus | obesity | 2401 | 99.54 | 92.62 | 100.00 | 96.17 | 99.51 | 99.00 | 90.68 |
| zero_shot_plus | cardio | 2401 | 72.55 | 76.95 | 64.16 | 69.98 | 80.90 | 73.34 | 69.78 |
| zero_shot_plus | kidney | 2401 | 72.05 | 72.37 | 58.38 | 64.63 | 82.68 | 70.62 | 62.06 |
| zero_shot_plus | stroke | 2401 | 89.50 | 9.93 | 10.07 | 10.00 | 94.39 | 52.48 | 6.19 |
| few_shot | lipid | 2490 | 68.31 | 45.52 | 78.92 | 57.74 | 64.31 | 72.47 | 43.00 |
| few_shot | diabetes | 2490 | 99.16 | 97.52 | 100.00 | 98.74 | 98.74 | 92.97 | 89.93 |
| few_shot | hypertension | 2490 | 98.11 | 97.11 | 99.48 | 98.28 | 96.48 | 96.52 | 96.16 |
| few_shot | obesity | 2490 | 99.40 | 90.26 | 100.00 | 94.88 | 99.36 | 98.22 | 86.94 |
| few_shot | cardio | 2490 | 74.38 | 74.28 | 74.64 | 74.46 | 74.12 | 75.14 | 70.37 |
| few_shot | kidney | 2490 | 72.33 | 68.80 | 67.10 | 67.94 | 76.39 | 72.47 | 63.75 |
| few_shot | stroke | 2490 | 89.00 | 11.52 | 12.93 | 12.18 | 93.77 | 52.53 | 6.56 |

## Comparative Table (Sorted by Task)

| Task | Approach | n_valid | ACC | P | R | F1 | Spec | AUC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cardio | few_shot | 2490 | 74.38 | 74.28 | 74.64 | 74.46 | 74.12 | 75.14 | 70.37 |
| cardio | zero_shot | 2453 | 75.87 | 74.53 | 78.20 | 76.32 | 73.56 | 77.00 | 72.57 |
| cardio | zero_shot_plus | 2401 | 72.55 | 76.95 | 64.16 | 69.98 | 80.90 | 73.34 | 69.78 |
| diabetes | few_shot | 2490 | 99.16 | 97.52 | 100.00 | 98.74 | 98.74 | 92.97 | 89.93 |
| diabetes | zero_shot | 2453 | 94.13 | 84.91 | 100.00 | 91.84 | 91.24 | 94.80 | 91.77 |
| diabetes | zero_shot_plus | 2401 | 90.55 | 77.57 | 100.00 | 87.37 | 85.95 | 97.96 | 95.27 |
| hypertension | few_shot | 2490 | 98.11 | 97.11 | 99.48 | 98.28 | 96.48 | 96.52 | 96.16 |
| hypertension | zero_shot | 2453 | 95.96 | 93.09 | 100.00 | 96.42 | 91.16 | 93.60 | 93.88 |
| hypertension | zero_shot_plus | 2401 | 93.84 | 89.76 | 100.00 | 94.61 | 86.58 | 96.24 | 96.13 |
| kidney | few_shot | 2490 | 72.33 | 68.80 | 67.10 | 67.94 | 76.39 | 72.47 | 63.75 |
| kidney | zero_shot | 2453 | 73.22 | 71.02 | 65.11 | 67.94 | 79.48 | 71.24 | 61.66 |
| kidney | zero_shot_plus | 2401 | 72.05 | 72.37 | 58.38 | 64.63 | 82.68 | 70.62 | 62.06 |
| lipid | few_shot | 2490 | 68.31 | 45.52 | 78.92 | 57.74 | 64.31 | 72.47 | 43.00 |
| lipid | zero_shot | 2453 | 76.60 | 55.30 | 69.98 | 61.78 | 79.05 | 74.40 | 46.77 |
| lipid | zero_shot_plus | 2401 | 79.13 | 59.95 | 67.65 | 63.56 | 83.36 | 75.71 | 49.70 |
| obesity | few_shot | 2490 | 99.40 | 90.26 | 100.00 | 94.88 | 99.36 | 98.22 | 86.94 |
| obesity | zero_shot | 2453 | 99.47 | 91.33 | 100.00 | 95.47 | 99.44 | 96.10 | 86.71 |
| obesity | zero_shot_plus | 2401 | 99.54 | 92.62 | 100.00 | 96.17 | 99.51 | 99.00 | 90.68 |
| stroke | few_shot | 2490 | 89.00 | 11.52 | 12.93 | 12.18 | 93.77 | 52.53 | 6.56 |
| stroke | zero_shot | 2453 | 88.83 | 12.57 | 15.38 | 13.84 | 93.38 | 53.90 | 6.67 |
| stroke | zero_shot_plus | 2401 | 89.50 | 9.93 | 10.07 | 10.00 | 94.39 | 52.48 | 6.19 |

## Key Takeaways

- Lipid next-visit performance is best in this snapshot with **zero_shot_plus** (ACC 79.13, F1 63.56).
- **Few_shot** has the highest recall for lipid (78.92) but with substantially lower precision (45.52), indicating more false positives.
- Diabetes/hypertension/obesity tasks are very strong across approaches, especially in few-shot.
- Stroke remains difficult for all approaches (very low precision/recall despite high specificity due to class imbalance).
- CoAgent results will need to be appended once job `47281877` completes and writes final outputs.
