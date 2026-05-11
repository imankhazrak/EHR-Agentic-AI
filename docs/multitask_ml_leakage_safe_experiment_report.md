# Leakage-Safe and Imbalance-Robust Multitask ML Pipeline: Data Encoding, Leakage Control, and Updated Results

## Abstract
This report documents a leakage-safe and imbalance-robust multitask machine-learning pipeline for MIMIC-III visit-pair prediction. The workflow combines patient-grouped data splitting, train-only feature encoding, task-specific leakage masks, class-weighted learners, train-only positive-class upsampling, and validation-based threshold optimization. We report the updated results after rerunning all classical ML models (logistic regression, decision tree, random forest) and summarize methodological implications for highly imbalanced labels.

## 1. Study Objective
The objective was to build a robust classical ML baseline for seven binary tasks while minimizing two major validity threats:

1. data leakage due to shared patients between train and test;
2. optimistic bias and poor minority recall caused by severe class imbalance.

Predicted labels:

- `label_lipid_disorder`
- `label_diabetes_current`
- `label_hypertension_current`
- `label_obesity_current`
- `label_cardio_next`
- `label_kidney_next`
- `label_stroke_next`

## 2. Dataset Preparation and Encoding

### 2.1 Leakage-safe split
Data were split by grouped patient identity (`SUBJECT_ID`) rather than row-wise random split.

- Total visit pairs: `12,456`
- Train: `10,031`
- Test: `2,425`
- Overlap diagnostics:
  - `subject_overlap = 0`
  - `hadm_current_overlap = 0`
  - `hadm_next_overlap = 0`

This removes patient-level contamination between model development and final evaluation.

### 2.2 Inner development split
Within train, an inner split was used for model development:

- `train_inner = 8,526`
- `val_inner = 1,505`
- `test = 2,425`

### 2.3 Feature encoding
A sparse bag-of-codes representation was used:

- Inputs: `diagnoses_codes_current`, `procedures_codes_current`, `medications_current`
- Encoder: binary `CountVectorizer`
- Vocabulary fit policy: **train_inner only** (then transform val/test)
- Final feature space: `6,927` features

Generated artifacts:

- `train_inner_encoded.npz`, `val_inner_encoded.npz`, `test_encoded.npz`
- `feature_vocab.json`
- `task_feature_masks.json`

## 3. Leakage Handling Strategy

### 3.1 Task-specific feature masking
For each target, diagnosis-code prefixes used in target construction were removed from that task's input feature space, preventing direct target reconstruction. Examples:

- diabetes: `250*`
- hypertension: `401*`-`405*`
- obesity: `2780*`
- lipid: `272*`
- cardio: `410*`, `411*`, `412*`, `413*`, `414*`, `428*`
- kidney: `584*`, `585*`, `586*`
- stroke: `430*`-`436*`

Masking was deterministic and persisted (`task_feature_masks.json`), with guardrail checks ensuring forbidden prefixes are absent from each active task feature set.

## 4. Imbalance-Robust Modeling Strategy
To address strong skew in positive/negative prevalence, we implemented a hybrid strategy:

1. **Class-weighted learners**:
   - logistic regression (`class_weight='balanced'`)
   - decision tree (`class_weight='balanced'`)
   - random forest (`class_weight='balanced'`)
2. **Train-only positive upsampling** to class parity (performed separately per task).
3. **Threshold tuning on validation** using a fixed grid (`0.05` to `0.95`):
   - primary criterion: macro-AUPRC policy
   - tie-breaker: macro-F1

### 4.1 Exact protocol used to handle imbalance
The imbalance procedure used in this experiment is strictly as follows:

1. Start from leakage-safe masked features for each task.
2. For that task, use only `train_inner` rows to rebalance classes by random upsampling of the minority class until class parity.
3. Fit a class-weighted classifier (`class_weight='balanced'`) on the rebalanced training subset.
4. Compute probabilities on `val_inner` (without any resampling).
5. Select the task-specific decision threshold from `0.05` to `0.95` based on the chosen policy (primary AUPRC, tie-break F1).
6. Apply the selected threshold to both `val_inner` and `test`.

### 4.2 What was explicitly avoided
- **No** upsampling or synthetic sampling on validation data.
- **No** upsampling or synthetic sampling on test data.
- **No** threshold tuning on test data.
- **No** change to label prevalence in evaluation splits.

This design ensures imbalance correction is learned strictly from the training partition, while validation/test remain unbiased for model selection and final reporting.

All selected thresholds and upsampling diagnostics are saved in:

- `data/outputs/mimiciii_multitask_ml_models/multitask_run_summary.json`

## 5. Updated Results After Imbalance-Robust Rerun

### 5.1 Test macro-average performance
From `multitask_metrics_all_models.csv` (`split=test`, `label=macro_avg`):

- Logistic Regression: Accuracy `75.08`, Sensitivity `62.70`, Specificity `69.01`, F1 `52.84`, AUC `75.63`, AUPRC `54.51`
- Decision Tree: Accuracy `74.43`, Sensitivity `51.72`, Specificity `76.62`, F1 `48.70`, AUC `64.21`, AUPRC `40.17`
- Random Forest: Accuracy `77.38`, Sensitivity `69.50`, Specificity `71.56`, F1 `58.15`, AUC `79.81`, AUPRC `58.49`

### 5.2 Main interpretation
- Random forest is now the strongest overall model by macro AUC/AUPRC and macro F1 on test.
- Compared with pre-imbalance handling runs, sensitivity improved notably for several difficult labels (especially rare outcomes), with an expected specificity trade-off.
- Extremely imbalanced tasks (e.g., stroke, obesity) remain challenging, but the updated pipeline provides a more clinically useful recall-oriented operating point.

## 6. Reproducibility
Pipeline commands:

1. `python -m src.scripts.run_preprocessing --config configs/default.yaml`
2. `python -m src.scripts.build_mimiciii_multitask_dataset --source-processed data/processed/mimiciii --output-processed data/processed/mimiciii_multitask`
3. `python -m src.scripts.prepare_multitask_ml_inputs --input-dir data/processed/mimiciii_multitask --output-dir data/processed/mimiciii_multitask_ml`
4. `python -m src.scripts.run_multitask_ml_models --input-dir data/processed/mimiciii_multitask_ml --output-dir data/outputs/mimiciii_multitask_ml_models`

Primary output files:

- `data/processed/mimiciii_multitask_ml/encoding_summary.json`
- `data/processed/mimiciii_multitask_ml/task_feature_masks.json`
- `data/outputs/mimiciii_multitask_ml_models/multitask_metrics_all_models.csv`
- `data/outputs/mimiciii_multitask_ml_models/multitask_run_summary.json`

## 7. Limitations and Future Work
- Threshold tuning is validation-set dependent and may require external recalibration.
- Rare-event tasks remain difficult; additional methods (focal loss alternatives via gradient boosting frameworks, calibrated ensembles, or temporal modeling) may improve robustness.
- Future reporting should include uncertainty intervals (bootstrap CIs), subgroup analyses, and external validation.

