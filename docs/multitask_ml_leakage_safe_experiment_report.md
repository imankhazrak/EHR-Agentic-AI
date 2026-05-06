# Leakage-Safe Multitask Machine Learning Pipeline: Encoding, Dataset Preparation, and Results

## Abstract
This report documents the end-to-end preparation of tabular inputs for multitask clinical prediction, with emphasis on leakage-aware design. We describe (i) feature encoding, (ii) split construction and validation, (iii) task-specific leakage controls, and (iv) model performance after implementing these controls. The final pipeline uses subject-grouped splitting (`SUBJECT_ID`), train-only vocabulary fitting, and per-task feature masking based on diagnosis-code prefixes used in label definitions. The resulting metrics are more conservative and clinically plausible than prior over-optimistic estimates.

## 1. Objective and Scope
The objective was to build and evaluate a multitask ML pipeline over MIMIC-III-derived visit-pair data while minimizing optimistic bias from data leakage. The prediction targets included seven tasks:

- `label_lipid_disorder`
- `label_diabetes_current`
- `label_hypertension_current`
- `label_obesity_current`
- `label_cardio_next`
- `label_kidney_next`
- `label_stroke_next`

Three model families were evaluated: logistic regression, decision tree, and random forest.

## 2. Dataset Preparation Workflow
The pipeline executed the following stages:

1. Preprocessing and visit-pair construction from MIMIC-III.
2. Multitask label augmentation.
3. Leakage-safe train/test split by patient identity.
4. Encoding into sparse ML matrices.
5. Per-task mask generation.
6. Model training and evaluation across train/validation/test.

### 2.1 Split Construction (Leakage-Aware)
The split was performed using grouped splitting with `SUBJECT_ID` as the grouping key to prevent patient overlap across partitions. The processed split summary was:

- Total pairs: `12,456`
- Train pairs: `10,031`
- Test pairs: `2,425`
- Split strategy: `grouped_by_subject_id`
- Overlap diagnostics:
  - `subject_overlap = 0`
  - `hadm_current_overlap = 0`
  - `hadm_next_overlap = 0`

This directly addresses patient-level leakage from repeated admissions.

### 2.2 Inner Validation Split and Encoded Dataset Sizes
Within the training set, an inner validation split was created:

- Train inner: `8,526`
- Validation inner: `1,505`
- Test: `2,425`

## 3. Feature Encoding for ML Models
The tabular representation used a sparse bag-of-codes encoding:

- Input code sources:
  - `diagnoses_codes_current`
  - `procedures_codes_current`
  - `medications_current`
- Encoder: binary `CountVectorizer` over semicolon-tokenized code strings.
- Vocabulary fit policy: **fit on `train_inner` only**, then transformed for validation and test.

From the latest run:

- Encoded feature count: `6,927`
- Feature artifacts:
  - `train_inner_encoded.npz`
  - `val_inner_encoded.npz`
  - `test_encoded.npz`
  - `feature_vocab.json`
  - `task_feature_masks.json`

## 4. Data Leakage Risk and Mitigation Strategy
Two leakage mechanisms were explicitly addressed:

1. **Patient overlap leakage**: same patient in train and test can inflate performance through repeated patient-specific patterns.
2. **Target-definition leakage**: some labels are defined directly from diagnosis-code prefixes that can also appear in input features.

### 4.1 Task-Specific Prefix Exclusion
For each task, features with diagnosis-token prefixes used to define that label were excluded from that task's model input. The mask specification included:

- Lipid disorder: `272*`
- Diabetes (current): `250*`
- Hypertension (current): `401*` to `405*`
- Obesity (current): `2780*`
- Cardio (next): `410*`, `411*`, `412*`, `413*`, `414*`, `428*`
- Kidney (next): `584*`, `585*`, `586*`
- Stroke (next): `430*` to `436*`

The pipeline generated deterministic keep-indices per label in `task_feature_masks.json`, and asserted forbidden prefixes were absent from each task's active feature set.

### 4.2 Modeling Implication
The training loop used per-label masked feature matrices, meaning each label is predicted from an input space tailored to prevent direct target reconstruction from leaked tokens.

## 5. Experimental Results (Leakage-Safe Rerun)

### 5.1 Test Macro-Average Performance
From `multitask_metrics_all_models.csv` (test split, `macro_avg` rows):

- Logistic Regression: Accuracy `79.34`, F1 `51.84`, AUC `76.45`, AUPRC `55.40`
- Decision Tree: Accuracy `76.19`, F1 `49.38`, AUC `64.38`, AUPRC `40.70`
- Random Forest: Accuracy `81.62`, F1 `44.11`, AUC `78.86`, AUPRC `57.17`

### 5.2 Per-Task Observations (Test Split)
- Random forest achieved the highest AUC for all seven tasks in this run.
- Logistic regression provided a stronger macro-F1 balance than random forest, indicating a precision/recall trade-off across tasks.
- Minority or highly imbalanced tasks (notably obesity and stroke) remained challenging under strict leakage controls, with low sensitivity/F1 for tree ensembles despite high specificity.

### 5.3 Interpretation
The new results should be interpreted as more reliable than earlier near-perfect scores for current-condition tasks, because:

- patient overlap was eliminated,
- direct label-defining diagnosis prefixes were task-wise masked,
- encoding was fit strictly on training data.

Therefore, performance values are expectedly lower for previously leaked tasks and better aligned with realistic generalization.

## 6. Reproducibility
The experiment can be reproduced by running, in order:

1. `python -m src.scripts.run_preprocessing --config configs/default.yaml`
2. `python -m src.scripts.build_mimiciii_multitask_dataset --source-processed data/processed/mimiciii --output-processed data/processed/mimiciii_multitask`
3. `python -m src.scripts.prepare_multitask_ml_inputs --input-dir data/processed/mimiciii_multitask --output-dir data/processed/mimiciii_multitask_ml`
4. `python -m src.scripts.run_multitask_ml_models --input-dir data/processed/mimiciii_multitask_ml --output-dir data/outputs/mimiciii_multitask_ml_models`

Primary output artifacts:

- `data/processed/mimiciii_multitask_ml/encoding_summary.json`
- `data/processed/mimiciii_multitask_ml/task_feature_masks.json`
- `data/outputs/mimiciii_multitask_ml_models/multitask_metrics_all_models.csv`
- `data/outputs/mimiciii_multitask_ml_models/multitask_run_summary.json`

## 7. Limitations and Next Steps
- Class imbalance remains substantial for some outcomes (e.g., obesity and stroke), motivating calibrated thresholds, reweighting, or cost-sensitive training.
- Further robustness checks should include confidence intervals (bootstrap), temporal external validation, and subgroup/fairness analyses.
- Reporting both threshold-dependent (F1, sensitivity, specificity) and threshold-independent (AUC, AUPRC) metrics should remain standard for clinical interpretability.

