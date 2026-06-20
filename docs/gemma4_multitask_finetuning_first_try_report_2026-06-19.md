# Gemma-4 Multitask Fine-Tuning: First-Try Experiment Report

**Date:** 2026-06-19  
**Repository:** EHR-Agentic-AI  
**Experiment line:** Unsloth LoRA SFT on natural class distribution (`natural_dist`)  
**Related audit:** [gemma4_finetuning_multitask_first_try.md](gemma4_finetuning_multitask_first_try.md)  
**Dashboard metrics:** [dashboard_with_llms_finetuining.html](dashboard_with_llms_finetuining.html)

---

## 1. Executive summary

This report documents the **first end-to-end attempt** at supervised multitask fine-tuning of **Gemma 4** using **Unsloth**, **4-bit quantization**, and **LoRA**. A single model reads one current hospital-visit narrative and emits **one strict JSON object** covering **seven binary clinical tasks** (each with `prediction`: Yes/No and `probability` in [0, 1]) plus a short `reasoning` string.

**Headline findings:**

- **Strong:** `diabetes_current`, `hypertension_current`, and `obesity_current` reached near-perfect hard-label agreement on parseable test rows.
- **Moderate:** `lipid_next`, `cardio_next`, and `kidney_next` showed usable but imperfect performance.
- **Failed:** `stroke_next` — the model **never predicted Yes** on usable rows (0% recall, ROC-AUC ≈ 0.50) despite 94% accuracy driven by class prevalence.
- **Imbalance:** No rebalancing was applied; training used the **natural label distribution**.

**Artifacts:**

| Artifact | Path |
|----------|------|
| LoRA adapter + tokenizer | `outputs/gemma4_router_lora/final_lora/` |
| First-pass test predictions | `outputs/gemma4_router_lora/test_predictions_first_try.jsonl` |
| Training JSONL | `dataset_for_unsloth/train.jsonl` (10,031 rows), `test.jsonl` (2,425 rows) |

---

## 2. Clinical tasks (seven labels)

Each example predicts seven binary outcomes from one **current-visit narrative** (`narrative_current`). Labels are derived from **ICD-coded visit pairs** in MIMIC-III multitask preprocessing.

| JSON key | CSV column | Clinical intent | Visit timing |
|----------|------------|-----------------|--------------|
| `lipid_next` | `label_lipid_next` | Disorders of lipid metabolism | **Next** visit |
| `diabetes_current` | `label_diabetes_current` | Diabetes mellitus | **Current** visit |
| `hypertension_current` | `label_hypertension_current` | Hypertension | **Current** visit |
| `obesity_current` | `label_obesity_current` | Obesity | **Current** visit |
| `cardio_next` | `label_cardio_next` | Major cardiovascular-related outcome | **Next** visit |
| `kidney_next` | `label_kidney_next` | Kidney-related outcome | **Next** visit |
| `stroke_next` | `label_stroke_next` | Stroke-related outcome | **Next** visit |

Schema enforcement: `MULTITASK_JSON_TASK_KEYS` in `src/llm/output_parser.py`. Gold JSON is exported by `src/scripts/export_multitask_unsloth_jsonl.py` with binary labels mapped to `"Yes"`/`"No"` and hard probabilities `1.0`/`0.0`.

---

## 3. Dataset splits and label distribution

**Source CSVs:** `data/processed/mimiciii_multitask/train.csv`, `test.csv`  
**Exported JSONL:** `dataset_for_unsloth/` (manifest: `dataset_for_unsloth/manifest.json`)

Positive class = **Yes** (label value 1). **Neg:Pos** = negative count ÷ positive count.

### 3.1 Training set (n = 10,031)

| Task | Positives | Negatives | Prev. Yes | Neg:Pos ratio |
|------|----------:|----------:|----------:|--------------:|
| `lipid_next` | 2,807 | 7,224 | 27.98% | 2.57:1 |
| `diabetes_current` | 3,469 | 6,562 | 34.58% | 1.89:1 |
| `hypertension_current` | 5,624 | 4,407 | 56.07% | 0.78:1 (Yes majority) |
| `obesity_current` | 580 | 9,451 | 5.78% | **16.29:1** |
| `cardio_next` | 5,170 | 4,861 | 51.54% | 0.94:1 |
| `kidney_next` | 4,374 | 5,657 | 43.60% | 1.29:1 |
| `stroke_next` | 581 | 9,450 | 5.79% | **16.27:1** |

### 3.2 Test set (n = 2,425)

| Task | Positives | Negatives | Prev. Yes | Neg:Pos ratio |
|------|----------:|----------:|----------:|--------------:|
| `lipid_next` | 609 | 1,816 | 25.11% | 2.98:1 |
| `diabetes_current` | 780 | 1,645 | 32.16% | 2.11:1 |
| `hypertension_current` | 1,324 | 1,101 | 54.60% | 0.83:1 (Yes majority) |
| `obesity_current` | 125 | 2,300 | 5.15% | **18.40:1** |
| `cardio_next` | 1,231 | 1,194 | 50.76% | 0.97:1 |
| `kidney_next` | 1,017 | 1,408 | 41.94% | 1.38:1 |
| `stroke_next` | 140 | 2,285 | 5.77% | **16.32:1** |

### 3.3 Class imbalance handling

**None was applied in this experiment.**

| Technique | Used? |
|-----------|-------|
| Oversampling / undersampling | No |
| Per-class or per-task loss weights | No |
| Focal loss | No |
| Stratified batch sampling | No |
| Validation threshold tuning | No |

Training explicitly preserved **natural prevalence** (Slurm run name `natural_dist`; see `slurm/train_unsloth_mt_natural_dist.slurm`). Loss is uniform cross-entropy on supervised JSON tokens only. Rare tasks (`obesity_current`, `stroke_next`) face roughly **16–18 negatives per positive** in both splits; the model still learned obesity well but **collapsed to all-negative predictions for stroke**.

---

## 4. Fine-tuning method

### 4.1 Pipeline overview

```
train.csv / test.csv
    → export_multitask_unsloth_jsonl
    → dataset_for_unsloth/*.jsonl
    → train_unsloth_router.py (Unsloth + TRL SFTTrainer)
    → outputs/gemma4_router_lora/final_lora/
    → test_unsloth_router.py (greedy generation)
    → parse_multitask_output
    → score_unsloth_multitask_jsonl.py (per-task metrics)
```

### 4.2 Stack and mechanism

| Component | Role |
|-----------|------|
| **Unsloth** `FastLanguageModel.from_pretrained` | Loads Gemma 4 with **NF4 4-bit** weights (`load_in_4bit=True`) |
| **PEFT LoRA** via `FastLanguageModel.get_peft_model` | Trainable adapters on attention + MLP projections; base weights frozen |
| **TRL** `SFTTrainer` | Supervised fine-tuning on Alpaca-formatted `text` column |
| **Response-only loss** | `completion_only_loss=True`; instruction/input masked; loss on JSON + EOS only (`src/utils/unsloth_multitask_labels.py`) |
| **Alpaca layout** | `### Instruction` / `### Input` / `### Response` — matches inference prompt prefix |

### 4.3 Key source files

| File | Purpose |
|------|---------|
| `src/scripts/export_multitask_unsloth_jsonl.py` | CSV → JSONL export |
| `scripts/train_unsloth_router.py` | Training entrypoint |
| `slurm/train_unsloth_mt_natural_dist.slurm` | Full Slurm driver |
| `scripts/test_unsloth_router.py` | Greedy test generation |
| `scripts/score_unsloth_multitask_jsonl.py` | Metric computation |
| `src/llm/output_parser.py` | Strict JSON parser (`parse_multitask_output`) |

---

## 5. Hyperparameters and training configuration

Recorded from `scripts/train_unsloth_router.py` defaults and `slurm/train_unsloth_mt_natural_dist.slurm`.

### 5.1 Base model and PEFT

| Parameter | Value |
|-----------|-------|
| Hugging Face / Unsloth ID | `unsloth/gemma-4-e2b-it-unsloth-bnb-4bit` |
| Quantization | 4-bit NF4 at load |
| PEFT method | LoRA |
| Rank `r` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `lora_bias` | none |

### 5.2 Optimization (SFTTrainer)

| Parameter | Value |
|-----------|-------|
| `max_seq_length` | 2048 |
| `max_steps` | 1500 (~1.2 epochs over 10,031 rows) |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| **Effective batch size** | **8** |
| `learning_rate` | `1e-4` |
| `warmup_ratio` | 0.03 |
| `weight_decay` | 0.0 |
| Optimizer | `adamw_8bit` |
| Mixed precision | bf16 if GPU supports it, else fp16 |
| `logging_steps` | 5 |
| `save_steps` | 50 |
| `save_total_limit` | 3 |
| `seed` | 42 |
| Final training loss (step 1500) | ≈ 0.433 |

### 5.3 Compute environment

| Resource | Value |
|----------|-------|
| GPU | 1 × NVIDIA V100-32G |
| Partition | `gpu-exp` |
| Conda env | `unsloth_env` |
| CUDA | 11.8 (module load when available) |

### 5.4 Framework versions (training snapshot)

| Package | Version |
|---------|---------|
| TRL | 0.24.0 |
| Transformers | 5.5.0 |
| PyTorch | 2.10.0 |
| PEFT | 0.19.1 |

---

## 6. Input format — real complete sample

**Source:** `dataset_for_unsloth/train.jsonl`, row 1 (`pair_id = 1`).

Each JSONL record has fields `instruction`, `input` (narrative only), and `output` (gold JSON). At **inference**, the model receives the Alpaca prefix ending at `### Response:` with no completion. At **training**, the gold JSON (and EOS) is appended to form the full supervised sequence.

### 6.1 Fixed instruction (v1 template)

```
You are given one current hospital visit narrative for a patient. Output exactly one JSON object with keys: lipid_next, diabetes_current, hypertension_current, obesity_current, cardio_next, kidney_next, stroke_next. Each key maps to an object with fields prediction (Yes or No) and probability (a number from 0 to 1, use two decimal places). Also include key reasoning with a short string. Do not add text outside the JSON object.
```

### 6.2 Inference prompt (complete, `pair_id = 1`)

```
### Instruction:
You are given one current hospital visit narrative for a patient. Output exactly one JSON object with keys: lipid_next, diabetes_current, hypertension_current, obesity_current, cardio_next, kidney_next, stroke_next. Each key maps to an object with fields prediction (Yes or No) and probability (a number from 0 to 1, use two decimal places). Also include key reasoning with a short string. Do not add text outside the JSON object.

### Input:
- Diagnoses made: Subendocardial infarction, initial episode of care; Cardiogenic shock; Blood in stool; Acute kidney failure, unspecified; Hypertensive chronic kidney disease, unspecified, with chronic kidney disease stage V or end stage renal disease; Congestive heart failure, unspecified; Compression of vein; Pneumonitis due to inhalation of food or vomitus; Atrial fibrillation; Paroxysmal ventricular tachycardia; Coronary atherosclerosis of native coronary artery; Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; Anemia in chronic kidney disease; Candidiasis of other urogenital sites; Pure hypercholesterolemia; Gout, unspecified; Personal history of malignant neoplasm of prostate; Other late effects of cerebrovascular disease
- Medications prescribed: Vial; Calcium Gluconate; Normocarb; Sterile Water; Potassium Chloride; Epoetin Alfa; Levofloxacin; D5W; Heparin Sodium; HydrALAZINE HCl; Isosorbide Mononitrate (Extended Release); Metoprolol; Morphine Sulfate; Warfarin; Amiodarone HCl; D5W (EXCEL BAG); Metolazone; SW; Furosemide; Docusate Sodium; Pantoprazole; Insulin; Artificial Tears; Captopril; Glycerin Supps; Magnesium Sulfate; Zolpidem Tartrate; NS; Lisinopril; Pneumococcal Vac Polyvalent; Fluconazole; Simvastatin; Nitroglycerin; Sodium Chloride 0.9%  Flush; Calcium Carbonate; Clonidine HCl; Sodium Bicarbonate; Acetylcysteine 20%; Calcitriol; Sodium Chloride; Chlorothiazide Sodium; Atropine Sulfate; Heparin Flush CRRT (5000 Units/mL); Polysaccharide Iron Complex; Clopidogrel Bisulfate; Sertraline HCl; Aspirin; Atorvastatin; Eucerin; Senna
- Procedures performed: Percutaneous transluminal coronary angioplasty [PTCA]; Implant of pulsation balloon; Angioplasty of other non-coronary vessel(s); Insertion of non-drug-eluting coronary artery stent(s); Procedure on three vessels; Insertion of three vascular stents; Venous catheterization for renal dialysis; Hemodialysis; Combined right and left heart cardiac catheterization; Coronary arteriography using two catheters; Transfusion of packed cells; Nonoperative removal of heart assist system

### Response:

```

Long narratives are shortened at tokenization time when needed (`src/utils/alpaca_multitask_fit.py`) so the JSON completion fits within `max_seq_length = 2048`.

### 6.3 Training `text` field

Training concatenates the inference prompt above with the gold JSON completion and tokenizer EOS. Only tokens from the first `{` of the JSON onward receive gradient (response-only masking).

---

## 7. Output format — real complete sample

### 7.1 Gold supervised target (`pair_id = 1`)

Gold labels are binary; exporter sets probability to **1.0** for Yes and **0.0** for No (not calibrated model scores):

```json
{
  "lipid_next": {"prediction": "No", "probability": 0.0},
  "diabetes_current": {"prediction": "Yes", "probability": 1.0},
  "hypertension_current": {"prediction": "Yes", "probability": 1.0},
  "obesity_current": {"prediction": "No", "probability": 0.0},
  "cardio_next": {"prediction": "Yes", "probability": 1.0},
  "kidney_next": {"prediction": "No", "probability": 0.0},
  "stroke_next": {"prediction": "No", "probability": 0.0},
  "reasoning": "Supervised gold labels from ICD-derived visit-pair coding."
}
```

### 7.2 Model-generated output (expected schema)

At inference the model must emit the **same JSON structure** with no text outside the object. The first-try artifact stores raw generations in `outputs/gemma4_router_lora/test_predictions_first_try.jsonl` (nested `gold_json` / `pred_json` when parsing succeeded). Parse failures (~56 of 2,425 rows per task) excluded some rows from per-task metrics.

---

## 8. Classifier and decision function

There is **no separate sklearn or classical classifier**. The fine-tuned language model **is** the multitask classifier: it generates JSON, which is parsed into seven binary decisions.

### 8.1 Generation (`scripts/test_unsloth_router.py`)

| Parameter | Value |
|-----------|-------|
| `temperature` | 0.0 |
| `do_sample` | False (greedy decoding) |
| `max_new_tokens` | 768 (current script default; first try may have used a shorter budget) |
| `max_seq_length` | 2048 |
| `load_in_4bit` | True |
| Model mode | `FastLanguageModel.for_inference(model)` |

### 8.2 Parsing (`parse_multitask_output` in `src/llm/output_parser.py`)

1. Strip optional Markdown JSON fences.
2. Parse a **single top-level JSON object** with no trailing text.
3. Require all seven task keys plus `reasoning`.
4. For each task block `{"prediction": "Yes"|"No", "probability": float}`:
   - Map `"Yes"` → `{prefix}_pred = 1`, `"No"` → `0`
   - Map valid probability → `{prefix}_prob` (e.g. `lipid_prob`, `diabetes_prob`, …)
5. Return `None` on any validation failure.

### 8.3 Hard classification rule

- **Positive class:** Yes = 1  
- **Decision:** use parsed `prediction` string directly — **no probability threshold** (e.g. no 0.5 cutoff on `probability`)  
- **Rank metrics:** ROC-AUC and AUPRC use the reported `probability` field when finite; these measure discrimination, not calibration

### 8.4 Metric function (`src/evaluation/metrics.py` → `compute_metrics`)

Called by `scripts/score_unsloth_multitask_jsonl.py` per task with:

| Argument | Meaning |
|----------|---------|
| `y_true` | Gold binary labels (0/1) |
| `y_pred` | Parsed hard predictions (0/1) |
| `y_score` | Parsed probabilities (optional, for AUC/AUPRC) |
| `positive_label` | 1 (Yes) |

**Outputs (0–100 scale):** accuracy, precision, recall/sensitivity, specificity, F1, balanced accuracy; optionally AUC and AUPRC. Confusion matrix TP/FP/TN/FN. Matthews correlation (MCC) added in the scoring script.

**Parse modes for scoring:** `strict` (default), `salvage`, `lenient_flat` — first-try dashboard metrics used strict parsing on nested `pred_json` from the first-try artifact.

---

## 9. First-try test results

**Source:** `outputs/gemma4_router_lora/test_predictions_first_try.jsonl`  
**Metrics date:** 2026-05-13 (see audit doc)  
**Coverage:** 2,425 test lines; per-task usable N ≈ 2,369 (stroke: 2,368)

Positive class = **Yes**. All percentages on [0, 100].

| Task | N | Prev Yes | Acc | Sens | Spec | F1 | ROC-AUC | AUPRC | TN | FP | FN | TP |
|------|--:|---------:|----:|-----:|-----:|---:|--------:|------:|---:|---:|---:|---:|
| `lipid_next` | 2,369 | 25.08% | 82.40 | 61.28 | 89.46 | 63.58 | 75.37 | 50.19 | 1588 | 187 | 230 | 364 |
| `diabetes_current` | 2,369 | 32.04% | 99.83 | 100.00 | 99.75 | 99.74 | 99.88 | 99.48 | 1606 | 4 | 0 | 759 |
| `hypertension_current` | 2,369 | 54.96% | 99.79 | 100.00 | 99.53 | 99.81 | 99.77 | 99.62 | 1062 | 5 | 0 | 1302 |
| `obesity_current` | 2,369 | 5.23% | 99.92 | 99.19 | 99.96 | 99.19 | 99.57 | 98.44 | 2244 | 1 | 1 | 123 |
| `cardio_next` | 2,369 | 50.95% | 82.69 | 83.51 | 81.84 | 83.10 | 82.68 | 77.46 | 951 | 211 | 199 | 1008 |
| `kidney_next` | 2,369 | 42.00% | 74.34 | 57.69 | 86.39 | 65.38 | 72.04 | 61.28 | 1187 | 187 | 421 | 574 |
| `stroke_next` | 2,368 | 5.79% | 94.21 | **0.00** | 100.00 | **0.00** | 50.00 | 5.79 | 2231 | 0 | 137 | 0 |

### 9.1 Pooled summaries

| Summary | Value |
|---------|------:|
| Pooled binary instances | 16,582 |
| Micro-accuracy | 90.45% |
| Micro-F1 (Yes) | 83.92% |
| Micro-ROC-AUC | 87.75% |
| Micro-AUPRC | 76.49% |
| Macro-F1 (Yes, unweighted over 7 tasks) | 72.97% |

These metrics appear in [dashboard_with_llms_finetuining.html](dashboard_with_llms_finetuining.html) under model **Gemma-4 finetuning**.

---

## 10. Reproduction commands

From repository root with `unsloth_env` activated:

```bash
export PYTHONPATH="$(pwd)"

# 1) Export CSV splits to JSONL (once)
python -m src.scripts.export_multitask_unsloth_jsonl

# 2) Train LoRA adapter
python scripts/train_unsloth_router.py \
  --run-name natural_dist \
  --train-jsonl dataset_for_unsloth/train.jsonl \
  --eval-jsonl dataset_for_unsloth/test.jsonl \
  --max-steps 1500 \
  --max-seq-length 2048

# Or submit Slurm job
sbatch slurm/train_unsloth_mt_natural_dist.slurm

# 3) Generate test predictions
python scripts/test_unsloth_router.py \
  --model-path outputs/gemma4_router_lora/final_lora \
  --test-jsonl dataset_for_unsloth/test.jsonl \
  --output-jsonl outputs/gemma4_router_lora/test_predictions_first_try.jsonl

# 4) Score predictions (CPU)
python scripts/score_unsloth_multitask_jsonl.py \
  outputs/gemma4_router_lora/test_predictions_first_try.jsonl \
  --parse-mode strict
```

---

## 11. Limitations and recommended next steps

1. **Stroke task unusable** — all-negative predictions; needs class-weighted loss, oversampling, focal loss, or a dedicated binary head.
2. **No imbalance correction** — natural distribution preserved; rare next-visit outcomes underrepresented in gradient signal.
3. **JSON validity** — partial parse failures reduce usable rows; constrained decoding or grammar-based JSON generation could help.
4. **Probability calibration** — model-reported probabilities used for AUC without reliability analysis; not clinically calibrated.
5. **Train–test parity** — ensure `max_seq_length` and `max_new_tokens` match between training audit and inference reproduction.

---

## 12. File index

| Path | Role |
|------|------|
| `docs/gemma4_multitask_finetuning_first_try_report_2026-06-19.md` | This report |
| `docs/gemma4_finetuning_multitask_first_try.md` | Detailed technical audit (metrics definitions, MCC, NPV) |
| `docs/dashboard_with_llms_finetuining.html` | ML vs LLM comparison dashboard |
| `dataset_for_unsloth/train.jsonl` | Training JSONL |
| `dataset_for_unsloth/test.jsonl` | Test JSONL |
| `scripts/train_unsloth_router.py` | Training script |
| `scripts/test_unsloth_router.py` | Inference script |
| `scripts/score_unsloth_multitask_jsonl.py` | Metric scoring |
| `outputs/gemma4_router_lora/final_lora/` | Saved adapter |
| `outputs/gemma4_router_lora/test_predictions_first_try.jsonl` | First-pass predictions |

---

*Report generated 2026-06-19 from repository artifacts and `data/processed/mimiciii_multitask/*.csv` label counts.*
