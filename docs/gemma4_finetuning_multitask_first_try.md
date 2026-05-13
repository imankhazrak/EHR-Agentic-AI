# Gemma 4 Multitask Clinical Router: Unsloth LoRA Fine-Tuning and First-Pass Test Evaluation

**Document:** `gemma4_finetuning_multitask_first_try.md`  
**Repository:** EHR-Agentic-AI  
**Last updated:** 2026-05-13  

This report documents the **first end-to-end attempt** at supervised multitask fine-tuning of a **Gemma 4** instruction model using **Unsloth**, **4-bit quantization**, and **LoRA**, followed by **held-out test-set generation and per-label classification metrics**. It is intended as an auditable technical record for collaborators and for future iterations (hyperparameters, class imbalance, and decoding settings).

---

## 1. Executive summary

- **Objective:** Train a single model to read one current hospital-visit narrative and emit **one strict JSON object** with **seven binary clinical tasks** (each with `prediction`: Yes/No and `probability` in \([0,1]\)) plus a short `reasoning` string, aligned with the multitask schema enforced in `src/llm/output_parser.py`.
- **Method:** Causal **supervised fine-tuning (SFT)** on Alpaca-style triples (`### Instruction` / `### Input` / `### Response`) using **Hugging Face TRL** `SFTTrainer` and **Unsloth** `FastLanguageModel`, with **NF4 4-bit** base weights and **LoRA** adapters on attention and MLP projections.
- **Artifacts:** Adapter weights and tokenizer under `outputs/gemma4_router_lora/final_lora/` (see `adapter_config.json`). First-pass greedy predictions on the multitask test JSONL are archived as `outputs/gemma4_router_lora/test_predictions_first_try.jsonl`.
- **Headline results:** **Diabetes, hypertension, and obesity** tasks reach **near-perfect** hard-label agreement on valid parses. **Lipid (next visit)** and **kidney (next visit)** show **moderate** performance with meaningful error mass on the minority positive class. **Stroke (next visit)** shows **complete failure to predict the positive class** in this run (all negatives at the hard threshold), yielding **ROC-AUC ≈ 0.50** and **F1 = 0** for Yes despite **94% accuracy** from prevalence.

---

## 2. Clinical prediction tasks (multitask schema)

Each training and test example uses the same instruction template: the model must output **exactly one JSON object** with the following keys (see `MULTITASK_JSON_TASK_KEYS` in `src/llm/output_parser.py`):

| JSON key | Clinical intent (high level) |
|----------|--------------------------------|
| `lipid_next` | Disorders of lipid metabolism **in a subsequent visit** (visit-pair label). |
| `diabetes_current` | Diabetes **present in the current visit** narrative. |
| `hypertension_current` | Hypertension **present in the current visit**. |
| `obesity_current` | Obesity **present in the current visit**. |
| `cardio_next` | Major cardiovascular-related outcome **in a subsequent visit**. |
| `kidney_next` | Kidney-related outcome **in a subsequent visit**. |
| `stroke_next` | Stroke-related outcome **in a subsequent visit**. |

Each task value is an object `{"prediction": "Yes"|"No", "probability": float}`. Gold labels in `dataset_for_unsloth/*.jsonl` were produced from **ICD-derived visit-pair coding** (as noted in gold `reasoning` strings in the dataset).

---

## 3. Data pipeline and splits

| Split | Path | Line count (examples) |
|-------|------|-------------------------|
| Train | `dataset_for_unsloth/train.jsonl` | 10,031 |
| Test | `dataset_for_unsloth/test.jsonl` | 2,425 |

- **Exporter:** `python -m src.scripts.export_multitask_unsloth_jsonl` (see docstring in `scripts/train_unsloth_router.py`). Outputs are validated in training (unless disabled) with **`parse_multitask_output`**, the same strict parser used at evaluation time in the current `scripts/test_unsloth_router.py`.
- **Formatting:** Training uses `build_multitask_sft_text` so the **tokenized `text` field** matches the **generation prefix** up to `### Response:` (instruction + input + assistant start), ensuring train–test alignment.

---

## 4. Base model and PEFT configuration

### 4.1 Base checkpoint

| Field | Value |
|-------|--------|
| Hugging Face / Unsloth ID | `unsloth/gemma-4-e2b-it-unsloth-bnb-4bit` |
| Model class (from adapter metadata) | `Gemma4ForConditionalGeneration` |
| Quantization at load | **4-bit** (`load_in_4bit=True` in training and inference scripts) |
| Task type | Causal language modeling (`CAUSAL_LM`) |

### 4.2 LoRA (PEFT) hyperparameters

Recorded from `outputs/gemma4_router_lora/final_lora/adapter_config.json` and defaults in `scripts/train_unsloth_router.py`:

| Parameter | Value |
|-----------|--------|
| PEFT method | LoRA (`peft_type`: `LORA`) |
| Rank `r` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| `lora_bias` | `false` (none) |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `bias` | `none` |
| `use_rslora` | `false` |
| `use_dora` | `false` |
| PEFT version (environment) | 0.19.1 (adapter file) |

**Trainable scope:** Only LoRA adapter weights on the listed linear projections; base Gemma weights remain frozen in 4-bit form during training.

---

## 5. Unsloth-specific training mechanisms

Unsloth is used for **memory- and speed-optimized** Gemma fine-tuning. Mechanisms relevant to this run:

| Mechanism | Role |
|-----------|------|
| `FastLanguageModel.from_pretrained` | Loads the **Unsloth-patched** Gemma 4 build with **bitsandbytes** NF4 weights. |
| `FastLanguageModel.get_peft_model` | Injects LoRA layers with `use_gradient_checkpointing="unsloth"` for **activation checkpointing** tuned to Unsloth kernels. |
| Alpaca SFT text field | TRL `SFTTrainer` trains on the `text` column built from instruction + input + **full multitask JSON** completion. |
| EOS handling | Training appends the tokenizer EOS after the JSON completion unless `--no-append-eos` is set (default: append). |

These choices prioritize **single-GPU feasibility** on mid-size institutional GPUs (e.g., V100-class in the provided Slurm template).

---

## 6. Optimization and TRL `SFTTrainer` settings

Defaults below are taken from **`scripts/train_unsloth_router.py`** and the **recommended full-training** invocation in **`slurm/train_unsloth_router.slurm`**. Your on-disk `checkpoint-1500/trainer_state.json` under `outputs/gemma4_router_lora/` reflects a completed **1,500 optimizer steps** run at **≈1.20 epochs** over 10,031 training rows (effective batch 8 ⇒ ~1.25 steps per epoch).

### 6.1 Core hyperparameters

| Setting | Value |
|---------|--------|
| `max_seq_length` | 2,048 |
| `max_steps` | 1,500 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| **Effective batch size** | **8** |
| `learning_rate` | `1e-4` |
| `warmup_ratio` | 0.03 |
| `weight_decay` | 0.0 |
| Optimizer | `adamw_8bit` |
| Mixed precision | `bf16` if supported by GPU, else `fp16` (auto in script) |
| `logging_steps` | 5 |
| `save_steps` | 50 |
| `save_total_limit` | 3 |
| `seed` | 42 |
| `report_to` | `none` |

### 6.2 Evaluation during training (when enabled)

If `--eval-jsonl dataset_for_unsloth/test.jsonl` is passed (as in `slurm/train_unsloth_router.slurm`):

| Setting | Value |
|---------|--------|
| `eval_strategy` | `steps` |
| `eval_steps` | Same as `save_steps` (50) in the Slurm driver |
| `load_best_model_at_end` | `True` |
| `metric_for_best_model` | `eval_loss` |
| `greater_is_better` | `False` |

**Note:** The bundled `checkpoint-1500/trainer_state.json` in `outputs/gemma4_router_lora/` does **not** contain `eval_loss` entries in `log_history` (final training loss at step 1,500 is **≈0.433**). That may indicate this specific checkpoint tree was produced with **evaluation disabled** or a different driver revision. For auditing, prefer the **`train_manifest.json`** in the run directory that actually produced these weights, if present.

### 6.3 Typical Slurm resource envelope

From `slurm/train_unsloth_router.slurm`:

| Resource | Value |
|----------|--------|
| Partition | `gpu-exp` |
| GPU | `v100-32g:1` |
| CPUs | 8 |
| Memory | 64 GB |
| Wall time | 24 h |
| Modules / env | CUDA 11.8 (if available), Conda env **`unsloth_env`** |

---

## 7. Framework versions (training environment snapshot)

From `outputs/gemma4_router_lora/README.md` (written by the training stack):

| Package | Version |
|---------|---------|
| TRL | 0.24.0 |
| Transformers | 5.5.0 |
| PyTorch | 2.10.0 |
| Datasets | 4.3.0 |
| Tokenizers | 0.22.2 |

---

## 8. Test-time inference protocol (first-try artifact)

**Prediction file:** `outputs/gemma4_router_lora/test_predictions_first_try.jsonl`  

Each line is a JSON object including at least: `gold_json`, `pred_json` (nested multitask dicts when parsing succeeded), `prediction_text`, and narrative fields. The **first-try** file format uses **nested** `gold_json` / `pred_json` objects (legacy evaluation dump).

**Recommended current driver** (for reproducibility going forward): `scripts/test_unsloth_router.py` with:

- `FastLanguageModel.from_pretrained(..., load_in_4bit=True)`
- `FastLanguageModel.for_inference(model)`
- Greedy decoding: `temperature=0.0`, `do_sample=False`
- Defaults in the script as of 2026-05: `--max-seq-length 2048`, `--max-new-tokens 512`

If the first-try run used an **older revision** (shorter `max_new_tokens` or context), decoding truncation could contribute to **JSON parse failures**; the metrics below therefore report **empirical coverage** (rows usable per task).

---

## 9. Evaluation methodology (metrics definitions)

### 9.1 Unit of analysis

For each **test example** \(i\) and each **task** \(t \in \{\text{seven keys}\}\):

- **Gold label** \(y_{i,t} \in \{0,1\}\): 1 if gold `prediction` is **Yes**, 0 if **No** (from `gold_json`).
- **Predicted label** \(\hat y_{i,t}\): same mapping from `pred_json` when the nested field exists and `prediction` is exactly `Yes` or `No` (case-insensitive).

**Rows excluded per task:** Examples where gold or predicted `prediction` is missing or non-binary for that task are dropped **for that task’s metrics** (e.g., malformed JSON such as wrong nesting of `prediction`).

**Coverage (this artifact):**

- **2,425** test lines total.
- **Per-task usable N:** **2,369** for six tasks; **2,368** for `stroke_next` (one additional row unusable for that task).
- **Fully missing `pred_json`:** **9** lines; additional exclusions come from **partially malformed** multitask JSON where individual tasks cannot be read.

### 9.2 Hard-label metrics (per task, positive = Yes = 1)

All percentages are on \([0,100]\).

| Metric | Definition |
|--------|------------|
| **Accuracy** | \(\mathrm{Acc} = \frac{TP+TN}{TP+TN+FP+FN}\) |
| **Balanced accuracy** | \(\frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)\) (equivalent sklearn formulation on 0/1 labels). |
| **Precision (PPV for Yes)** | \(TP/(TP+FP)\); **zero** if no positive predictions. |
| **Recall (sensitivity for Yes)** | \(TP/(TP+FN)\). |
| **Specificity (for No)** | \(TN/(TN+FP)\). |
| **F1 (Yes)** | Harmonic mean of precision and recall for the positive class. |
| **NPV (for No)** | \(TN/(TN+FN)\) — how trustworthy **No** predictions are when the disease is absent in gold. |
| **MCC** | Matthews correlation coefficient between \(y\) and \(\hat y\) (balanced measure for binary confusion tables). |

### 9.3 Rank-based metrics (probability head)

When `pred_json[task].probability` parses as a finite float, **ROC-AUC** and **AUPRC** are computed **versus gold Yes/No** using `sklearn.metrics.roc_auc_score` and `average_precision_score`. **Important:** these scores measure **discrimination** of the reported probability head; they **do not** assert clinical calibration without a separate reliability analysis.

### 9.4 Pooled (multi-label instance) summaries

Across **all tasks and all usable rows**, we pool \((y_{i,t}, \hat y_{i,t})\) into one long vector of **\(N_{\text{inst}} = 16{,}582\)** binary decisions (seven per example for most rows; stroke uses one fewer usable row in the sum—see micro counts below). Reported **micro** accuracy / F1 / AUC / AUPRC are **global** measures on this pooled set.

**Macro F1 (Yes):** unweighted mean of the seven per-task F1 (Yes) values (**≈72.97%**), which treats each task equally regardless of support.

---

## 10. Per-label detailed results (first try)

Positive class = **Yes**. **NPV** = \(100 \times TN/(TN+FN)\). Confusion matrix layout: \([[TN, FP], [FN, TP]]\).

### 10.1 `lipid_next`

| Quantity | Value |
|----------|------:|
| Usable examples \(N\) | 2,369 |
| Gold prevalence Yes | 25.08% (594 / 2,369) |
| Accuracy (%) | 82.40 |
| Balanced accuracy (%) | 75.37 |
| Precision / PPV Yes (%) | 66.06 |
| Recall / sensitivity Yes (%) | 61.28 |
| Specificity (%) | 89.46 |
| NPV No (%) | 87.35 |
| F1 Yes (%) | 63.58 |
| MCC | 0.5206 |
| ROC-AUC (%) | 75.37 |
| AUPRC (%) | 50.19 |
| Confusion \([[TN,FP],[FN,TP]]\) | \([[1588, 187], [230, 364]]\) |

### 10.2 `diabetes_current`

| Quantity | Value |
|----------|------:|
| Usable examples \(N\) | 2,369 |
| Gold prevalence Yes | 32.04% (759 / 2,369) |
| Accuracy (%) | 99.83 |
| Balanced accuracy (%) | 99.88 |
| Precision / PPV Yes (%) | 99.48 |
| Recall / sensitivity Yes (%) | 100.00 |
| Specificity (%) | 99.75 |
| NPV No (%) | 100.00 |
| F1 Yes (%) | 99.74 |
| MCC | 0.9961 |
| ROC-AUC (%) | 99.88 |
| AUPRC (%) | 99.48 |
| Confusion \([[TN,FP],[FN,TP]]\) | \([[1606, 4], [0, 759]]\) |

### 10.3 `hypertension_current`

| Quantity | Value |
|----------|------:|
| Usable examples \(N\) | 2,369 |
| Gold prevalence Yes | 54.96% (1,302 / 2,369) |
| Accuracy (%) | 99.79 |
| Balanced accuracy (%) | 99.77 |
| Precision / PPV Yes (%) | 99.62 |
| Recall / sensitivity Yes (%) | 100.00 |
| Specificity (%) | 99.53 |
| NPV No (%) | 100.00 |
| F1 Yes (%) | 99.81 |
| MCC | 0.9957 |
| ROC-AUC (%) | 99.77 |
| AUPRC (%) | 99.62 |
| Confusion \([[TN,FP],[FN,TP]]\) | \([[1062, 5], [0, 1302]]\) |

### 10.4 `obesity_current`

| Quantity | Value |
|----------|------:|
| Usable examples \(N\) | 2,369 |
| Gold prevalence Yes | 5.23% (124 / 2,369) |
| Accuracy (%) | 99.92 |
| Balanced accuracy (%) | 99.57 |
| Precision / PPV Yes (%) | 99.19 |
| Recall / sensitivity Yes (%) | 99.19 |
| Specificity (%) | 99.96 |
| NPV No (%) | 99.96 |
| F1 Yes (%) | 99.19 |
| MCC | 0.9915 |
| ROC-AUC (%) | 99.57 |
| AUPRC (%) | 98.44 |
| Confusion \([[TN,FP],[FN,TP]]\) | \([[2244, 1], [1, 123]]\) |

### 10.5 `cardio_next`

| Quantity | Value |
|----------|------:|
| Usable examples \(N\) | 2,369 |
| Gold prevalence Yes | 50.95% (1,207 / 2,369) |
| Accuracy (%) | 82.69 |
| Balanced accuracy (%) | 82.69 |
| Precision / PPV Yes (%) | 82.69 |
| Recall / sensitivity Yes (%) | 83.51 |
| Specificity (%) | 81.84 |
| NPV No (%) | 82.69 |
| F1 Yes (%) | 83.10 |
| MCC | 0.6537 |
| ROC-AUC (%) | 82.68 |
| AUPRC (%) | 77.46 |
| Confusion \([[TN,FP],[FN,TP]]\) | \([[951, 211], [199, 1008]]\) |

### 10.6 `kidney_next`

| Quantity | Value |
|----------|------:|
| Usable examples \(N\) | 2,369 |
| Gold prevalence Yes | 42.00% (995 / 2,369) |
| Accuracy (%) | 74.34 |
| Balanced accuracy (%) | 72.04 |
| Precision / PPV Yes (%) | 75.43 |
| Recall / sensitivity Yes (%) | 57.69 |
| Specificity (%) | 86.39 |
| NPV No (%) | 73.82 |
| F1 Yes (%) | 65.38 |
| MCC | 0.4659 |
| ROC-AUC (%) | 72.04 |
| AUPRC (%) | 61.28 |
| Confusion \([[TN,FP],[FN,TP]]\) | \([[1187, 187], [421, 574]]\) |

### 10.7 `stroke_next`

| Quantity | Value |
|----------|------:|
| Usable examples \(N\) | 2,368 |
| Gold prevalence Yes | 5.79% (137 / 2,368) |
| Accuracy (%) | 94.21 |
| Balanced accuracy (%) | 50.00 |
| Precision / PPV Yes (%) | 0.00 |
| Recall / sensitivity Yes (%) | 0.00 |
| Specificity (%) | 100.00 |
| NPV No (%) | 94.21 |
| F1 Yes (%) | 0.00 |
| MCC | 0.00 |
| ROC-AUC (%) | 50.00 |
| AUPRC (%) | 5.79 |
| Confusion \([[TN,FP],[FN,TP]]\) | \([[2231, 0], [137, 0]]\) |

**Interpretation:** The model **never predicts Yes** for `stroke_next` on usable rows, so **recall for stroke is zero**, ROC-AUC is **uninformative (chance)**, and accuracy is dominated by the negative majority. This task requires **rebalancing, focal loss, task-specific heads, or a dedicated second-stage model** in future work.

---

## 11. Combined results table (all labels)

Single-row-per-task summary for the **first-try** predictions file. **Prev Yes** = gold prevalence of Yes (%). **PPV** = precision for Yes. **Sens** = recall for Yes. **Spec** = specificity for No.

| Task | \(N\) | Prev Yes % | Acc % | Bal acc % | PPV % | Sens % | Spec % | NPV % | F1 % | MCC | ROC-AUC % | AUPRC % | TN | FP | FN | TP |
|------|------:|-----------:|------:|----------:|------:|-------:|-------:|------:|-----:|----:|----------:|--------:|---:|---:|---:|---:|
| lipid_next | 2,369 | 25.08 | 82.40 | 75.37 | 66.06 | 61.28 | 89.46 | 87.35 | 63.58 | 0.5206 | 75.37 | 50.19 | 1588 | 187 | 230 | 364 |
| diabetes_current | 2,369 | 32.04 | 99.83 | 99.88 | 99.48 | 100.00 | 99.75 | 100.00 | 99.74 | 0.9961 | 99.88 | 99.48 | 1606 | 4 | 0 | 759 |
| hypertension_current | 2,369 | 54.96 | 99.79 | 99.77 | 99.62 | 100.00 | 99.53 | 100.00 | 99.81 | 0.9957 | 99.77 | 99.62 | 1062 | 5 | 0 | 1302 |
| obesity_current | 2,369 | 5.23 | 99.92 | 99.57 | 99.19 | 99.19 | 99.96 | 99.96 | 99.19 | 0.9915 | 99.57 | 98.44 | 2244 | 1 | 1 | 123 |
| cardio_next | 2,369 | 50.95 | 82.69 | 82.69 | 82.69 | 83.51 | 81.84 | 82.69 | 83.10 | 0.6537 | 82.68 | 77.46 | 951 | 211 | 199 | 1008 |
| kidney_next | 2,369 | 42.00 | 74.34 | 72.04 | 75.43 | 57.69 | 86.39 | 73.82 | 65.38 | 0.4659 | 72.04 | 61.28 | 1187 | 187 | 421 | 574 |
| stroke_next | 2,368 | 5.79 | 94.21 | 50.00 | 0.00 | 0.00 | 100.00 | 94.21 | 0.00 | 0.00 | 50.00 | 5.79 | 2231 | 0 | 137 | 0 |

### 11.1 Pooled (micro) and macro summaries

| Summary | Value |
|---------|------:|
| Pooled binary instances \(N_{\text{inst}}\) | 16,582 |
| Micro-accuracy (%) | 90.45 |
| Micro–F1 for Yes (%) | 83.92 |
| Micro–ROC-AUC (%) | 87.75 |
| Micro–AUPRC (%) | 76.49 |
| Macro–F1 for Yes (%, unweighted over 7 tasks) | 72.97 |

---

## 12. Known limitations and recommended next steps

1. **Rare next-visit stroke:** collapse to all-negative predictions ⇒ **do not use** this checkpoint for stroke risk as deployed. Mitigations: class-weighted loss, oversampling, task-specific threshold tuning on validation, or **separate binary heads** instead of pure JSON generation.
2. **JSON validity:** ~56/2,425 rows per task lose hard metrics due to malformed structure in `pred_json` (nested `prediction` missing or wrong type). Tighter decoding (`max_new_tokens`), constrained decoding (JSON schema / grammar), or post-hoc repair could improve coverage.
3. **Probability calibration:** AUC/AUPRC use model-reported `probability` values **without calibration**; clinical use would require reliability curves and external validation.
4. **Train vs. inference parity:** Ensure `max_seq_length` and decoding budgets match training when reproducing metrics; document exact Git commit and CLI flags per Slurm job in `train_manifest.json` for each artifact directory.

---

## 13. File index (quick reference)

| Path | Role |
|------|------|
| `scripts/train_unsloth_router.py` | Unsloth + TRL LoRA SFT entrypoint and hyperparameter CLI. |
| `slurm/train_unsloth_router.slurm` | Example full-training Slurm driver. |
| `scripts/test_unsloth_router.py` | Greedy multitask JSON evaluation (current schema-aware parser). |
| `src/llm/output_parser.py` | Strict multitask JSON schema (`MULTITASK_JSON_TASK_KEYS`). |
| `outputs/gemma4_router_lora/final_lora/` | Saved LoRA adapter + tokenizer for this experiment line. |
| `outputs/gemma4_router_lora/test_predictions_first_try.jsonl` | Archived first-pass test generations and nested parses. |
| `docs/gemma4_finetuning_multitask_first_try.md` | This report. |

---

*Metrics in Sections 9–11 were computed in Python with `scikit-learn` on 2026-05-13 from `test_predictions_first_try.jsonl` using the definitions above.*
