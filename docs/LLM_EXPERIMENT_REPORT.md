# LLM Experiment Report — MIMIC-III Next-Visit Lipid Disorder Prediction

This report provides a complete technical record of the repository’s LLM experiments for next-visit lipid-disorder prediction, including task framing, dataset and label setup, prompt modes, inference and parsing flow, evaluation rules, reproducibility commands, output artifacts, and archived full-run results.

This document is a **standalone technical report** of the large language model (LLM) experiments implemented in this repository: task definition, data, prompts, pipelines, evaluation protocol, outputs, and **numerical results** as recorded in `data/outputs/mimiciii/`. It is intended for your own archive or for sharing with collaborators.

**Companion notes:** `docs/llm_prompt_modes_explained.md` (pedagogical walkthrough of zero-shot vs zero-shot+ vs few-shot), `METHODOLOGY.md` (whole-pipeline reference).

---

## Table of Contents
- [1. Objective](#1-objective)
- [2. Ground Truth (Label Construction)](#2-ground-truth-label-construction)
- [3. Data and Split](#3-data-and-split)
  - [3.1 Evaluation set size for the reported LLM metrics](#31-evaluation-set-size-for-the-reported-llm-metrics)
- [4. Narrative Construction (LLM Input)](#4-narrative-construction-llm-input)
- [5. Model and API Settings](#5-model-and-api-settings)
- [6. Prompt Modes (Four LLM Experiments)](#6-prompt-modes-four-llm-experiments)
  - [6.1 EHR-CoAgent pipeline (operational detail)](#61-ehr-coagent-pipeline-operational-detail)
- [7. Inference Loop and Artifacts](#7-inference-loop-and-artifacts)
- [8. Output Parsing](#8-output-parsing)
- [9. Evaluation Protocol](#9-evaluation-protocol)
- [10. Results (Archived Full Run)](#10-results-archived-full-run)
  - [10.1 Summary table (LLM rows)](#101-summary-table-llm-rows)
  - [10.2 Confusion matrices and parse health](#102-confusion-matrices-and-parse-health)
- [11. How to Reproduce](#11-how-to-reproduce)
- [12. Key Artifacts (Checklist)](#12-key-artifacts-checklist)
- [13. Limitations and Caveats](#13-limitations-and-caveats)
- [14. Related Documentation](#14-related-documentation)

## 1. Objective

**Binary prediction:** Given structured information from a patient’s **current** hospital admission, predict whether **Disorders of Lipid Metabolism** (CCS-style phenotype) will appear among **diagnoses coded on the patient’s next hospital admission**.

- **Input to the LLM:** A natural-language **narrative** built from the current visit (bulleted diagnoses, medications, procedures), *not* the raw next-visit record.
- **Positive class (label = 1):** At least one qualifying ICD-9-CM code appears in the **next** visit’s diagnosis list.
- **Negative class (label = 0):** Otherwise.

The LLM **does not** see next-visit codes; those are used only to construct `label_lipid_disorder` and for evaluation.

---

## 2. Ground Truth (Label Construction)

Implementation: `src/data/build_target_labels.py`.

- Phenotype aligned with **HCUP CCS single-level category 53** (Disorders of lipoid metabolism), implemented via ICD-9-CM codes **`272.x`**, stored in MIMIC-III **without** the decimal (e.g. `2724`).
- Code list and prefix logic are documented in the module docstring; exact matching and prefix rules are defined in `is_lipid_disorder()`.

**Important cohort note (from `METHODOLOGY.md`):** The pipeline uses **hospital admissions** from MIMIC-III v1.4 as visits; it is **not** restricted to an ICU-only cohort unless you add that filter separately.

---

## 3. Data and Split

| Item | Setting (default `configs/default.yaml`) |
|------|-------------------------------------------|
| Visit pairs | Consecutive admissions per patient with ≥ 2 visits |
| Train/test split | Random, **80% / 20%**, **stratified** on `label_lipid_disorder` (`test_size: 0.2`, `stratify_test: true`) |
| Split unit | **Visit-pair rows** (not patient-level blocking) |
| Reproducibility | `seed: 42` |

**Processed artifacts:** `data/processed/mimiciii/train.csv`, `test.csv`, plus interim tables under `data/interim/mimiciii/`.

### 3.1 Evaluation set size for the reported LLM metrics

The metrics JSON files in `data/outputs/mimiciii/` (`llm_*_metrics.json`) record:

| Quantity | Value |
|----------|--------|
| **Test pairs evaluated** (`n_total`) | **2,492** |

Counts **tp / fp / tn / fn** in those files sum to 2,492 for modes with no unparseable exclusions on the merged set.

> **Note:** If you rerun preprocessing with different `max_pairs`, `test_size`, or data extracts, row counts in `train.csv` / `test.csv` may change. Always use `n_total` and `llm_*_results.csv` for the run you are reporting.

---

## 4. Narrative Construction (LLM Input)

Configuration: `configs/default.yaml` → `narrative`.

| Field | Value |
|-------|--------|
| Style | Bullet lists |
| Max diagnoses | 50 |
| Max medications | 50 |
| Max procedures | 50 |
| Empty sections | `"None recorded"` |

Column used at inference time: **`narrative_current`**.

---

## 5. Model and API Settings

| Parameter | Value |
|-----------|--------|
| Provider | OpenAI Chat Completions (`configs/default.yaml` → `llm.provider`) |
| Configured model name | `gpt-4o-mini` |
| Resolved model ID (observed in metrics) | `gpt-4o-mini-2024-07-18` |
| `base_url` | `null` (default `api.openai.com` unless overridden) |
| `temperature` | **0.0** |
| `max_tokens` | **1024** |
| `seed` | 42 (passed where supported) |
| Response cache | **Enabled** (`cache_responses: true`, `cache_dir: data/outputs/mimiciii/llm_cache`) |
| Rate limit | **30** requests/minute (`rate_limit_rpm`) |
| Logprobs | **Off** (`request_logprobs: false`) |

Authentication: repository-root **`.env`** with `OPENAI_API_KEY` (loaded locally or via `scripts/slurm_llm_env.sh` on the cluster). Do not commit secrets.

---

## 6. Prompt Modes (Four LLM Experiments)

All modes use Jinja2 templates under `prompts/` rendered by `src/llm/prompt_builder.py` into chat messages (`system` + `user`), then sent through `src/llm/predictor.py`.

| Mode (code) | Summary table label | Template file |
|-------------|---------------------|---------------|
| `zero_shot` | Zero-Shot | `prompts/predictor_zero_shot.txt` |
| `zero_shot_plus` | Zero-Shot+ | `prompts/predictor_zero_shot_plus.txt` |
| `few_shot` | Few-Shot (N=6) | `prompts/predictor_few_shot.txt` |
| `coagent` | EHR-CoAgent | `prompts/predictor_coagent_base.txt` |

Shared content across templates (evolving with your edits) typically includes:

- Definition of the prediction target (lipid metabolism disorders / CCS 53).
- **Prediction clarification:** predict whether the diagnosis will be **coded on the next visit**, not generic clinical “risk”; instructions to avoid treating **statin-only** or **generic cardiovascular history** as sufficient for **Yes** unless explicit lipid-related evidence appears (see current template text).

**Zero-shot:** Instructions + current narrative + compact output format.

**Zero-shot+:** Adds structured analysis instructions and an explicit **population prevalence** hint (~27.6% of subsequent visits with the target in MIMIC-III, as stated in the template).

**Few-shot (in-context, not weight training):**

- **Exemplars:** `n_positive: 3`, `n_negative: 3`, **`random_balanced`**, `seed: 42` (`configs/default.yaml` → `few_shot`).
- Sampled from **`train.csv` only**; formatted as `Case k:` + narrative + `Outcome: Yes/No` (`src/data/exemplar_selector.py` → `format_exemplar_block`).
- Saved artifact: `data/outputs/mimiciii/llm_few_shot_exemplars.csv` (for traceability).

**EHR-CoAgent:** Few-shot-style prompt **plus** a block of **consolidated critic feedback** derived from errors on a **calibration** subset of the training data.

### 6.1 EHR-CoAgent pipeline (operational detail)

Source: `src/llm/coagent.py`. Default hyperparameters from `configs/default.yaml` → `coagent`:

| Parameter | Default |
|-----------|---------|
| `calibration_size` | 200 (rows sampled from `train_df`) |
| `n_wrong_samples` | 30 (cap on mispredictions fed to critic) |
| `critic_batch_size` | 10 |
| `n_critic_rounds` | 3 |
| `consolidation_method` | `llm` |

**Steps:**

1. Sample up to **200** training pairs; run **`few_shot`** predictions on them.
2. Select **incorrect** predictions; sample up to **30** for critique.
3. **Critic** (`prompts/critic_batch_reflection.txt`) processes batches; outputs saved under `data/outputs/mimiciii/critic_feedback/`.
4. **Consolidation** (`prompts/feedback_consolidation.txt`) merges rounds into one text block.
5. Run **`coagent`** mode on the **full test set** with the same `demonstration_cases` as few-shot **and** `critic_feedback` set to the consolidated text.

Edge case: if no errors on the calibration run, the pipeline falls back to plain **few-shot** on the test set (see `run_coagent_pipeline`).

Reference dump of exemplars + feedback: `data/outputs/mimiciii/prompts_used/coagent_augmented_context.txt` (when the critic path runs).

---

## 7. Inference Loop and Artifacts

For each test row:

1. Build messages from the template + `narrative` (+ `demonstration_cases`, + `critic_feedback` when applicable).
2. Call the LLM; store **raw** payload under `data/outputs/mimiciii/raw_llm_responses/<mode>/<pair_id>.json`.
3. Parse assistant text; append rows to a results table.
4. Save **rendered prompts** per row: `data/outputs/mimiciii/prompts_used/<mode>/prompt_<pair_id>.txt` (as implemented in `src/llm/predictor.py`).

Per-mode prediction tables: `data/outputs/mimiciii/llm_<mode>_results.csv`.

---

## 8. Output Parsing

Module: `src/llm/output_parser.py` → `parse_prediction`.

**Order of rules:**

1. **First line:** `Prediction: Yes` / `Prediction: No` (or leading `Yes` / `No` alone).
2. Else search for `Prediction: Yes` / `Prediction: No` anywhere (**fallback**).
3. Else use first dominant standalone `Yes` vs `No` by position.
4. Else **`unparseable`**.

Optional: if logprobs were enabled, `extract_logprob_confidence` could populate `prob_yes` / `prob_no`; default config keeps this off.

---

## 9. Evaluation Protocol

Module: `src/evaluation/evaluate_llm_runs.py` after each mode.

1. Merge predictions with `test_df` on **`pair_id`**.
2. Map **`Yes` → 1**, **`No` → 0**; **`unparseable`** → missing binary prediction.
3. **Metrics** computed on rows with valid predictions only (`src/evaluation/metrics.py` → `compute_metrics`):
   - **Accuracy**, **sensitivity (recall on positive)**, **specificity**, **F1** (positive class = 1), all reported as **percentages** in JSON/CSV.
   - Confusion counts: **tp, fp, tn, fn**.
4. `n_total` = all merged rows; `n_valid` = rows used for metrics; `n_unparseable` = remainder.

**Bootstrap CIs:** Disabled by default (`evaluation.bootstrap_ci: false`).

---

## 10. Results (Archived Full Run)

The following values are taken from **`data/outputs/mimiciii/summary_table.csv`** and **`llm_*_metrics.json`** as they exist in the repository after a completed full pipeline run. They summarize performance on **n = 2,492** test pairs unless noted.

### 10.1 Summary table (LLM rows)

| Model | Approach | ACC (%) | Sensitivity (%) | Specificity (%) | F1 (%) |
|--------|-----------|---------|-----------------|-----------------|--------|
| gpt-4o-mini-2024-07-18 | Zero-Shot | 80.82 | 57.69 | 89.55 | 62.24 |
| gpt-4o-mini-2024-07-18 | Zero-Shot+ | 80.73 | 56.22 | 89.99 | 61.54 |
| gpt-4o-mini-2024-07-18 | Few-Shot (N=6) | 80.65 | 57.39 | 89.44 | 61.93 |
| gpt-4o-mini-2024-07-18 | EHR-CoAgent | **80.93** | **58.21** | 89.50 | **62.57** |

**Interpretation:** All four modes perform **very similarly** on aggregate metrics (differences on the order of **tenths of a percent** accuracy / F1). That is consistent with **high pairwise agreement** between modes on the same `pair_id` (most prompts yield the same Yes/No decision; only a small fraction of rows flip when adding instructions, exemplars, or critic text).

### 10.2 Confusion matrices and parse health

| Mode | tp | fp | tn | fn | n_total | n_valid | n_unparseable |
|------|----|----|----|----|---------|---------|---------------|
| zero_shot | 394 | 189 | 1620 | 289 | 2492 | 2492 | 0 |
| zero_shot_plus | 384 | 181 | 1627 | 299 | 2492 | 2491 | 1 |
| few_shot | 392 | 191 | 1617 | 291 | 2492 | 2491 | 1 |
| coagent | 397 | 190 | 1619 | 285 | 2492 | 2491 | 1 |

**Unparseable rows** are excluded from tp/fp/tn/fn but counted in `n_unparseable`; three modes show **one** such row in this run.

---

## 11. How to Reproduce

**Full pipeline (preprocess + ML + all LLM modes + summary):**

```bash
export PYTHONPATH=/path/to/EHR-Agentic-AI
STAGE=full bash scripts/run_mimic_pipeline.sh
```

**LLM-only** (requires existing `train.csv` / `test.csv`):

```bash
export PYTHONPATH=/path/to/EHR-Agentic-AI
bash scripts/run_llm_only.sh
```

**Smoke / debug** (small data and test cap via `configs/debug_small.yaml`):

```bash
STAGE=smoke bash scripts/run_mimic_pipeline.sh
# or
STAGE=smoke bash scripts/run_llm_only.sh
```

**Cluster (example):** `sbatch slurm/mimic_iii_full.slurm` — see `slurm/` and `scripts/slurm_llm_env.sh`.

**Post-hoc audit markdown** (narrative + model reply + verdict per row):

```bash
PYTHONPATH=. python scripts/export_llm_audit.py
```

---

## 12. Key Artifacts (Checklist)

| Path | Description |
|------|-------------|
| `data/outputs/mimiciii/llm_*_results.csv` | Per–test-row predictions and raw text |
| `data/outputs/mimiciii/llm_*_metrics.json` | Metrics + confusion + parse counts |
| `data/outputs/mimiciii/summary_table.csv` | Paper-style combined table |
| `data/outputs/mimiciii/raw_llm_responses/<mode>/` | Raw API-style JSON per `pair_id` |
| `data/outputs/mimiciii/prompts_used/<mode>/` | Rendered prompts per `pair_id` |
| `data/outputs/mimiciii/llm_few_shot_exemplars.csv` | Few-shot / coagent exemplar IDs |
| `data/outputs/mimiciii/critic_feedback/` | Critic inputs/outputs, consolidated feedback |
| `data/outputs/mimiciii/llm_cache/` | Hashed response cache (reuse across runs) |

---

## 13. Limitations and Caveats

1. **Pair-level split:** The same patient can contribute pairs to both train and test; metrics are **not** patient-clustered unless you change the split.
2. **Label vs clinical intuition:** The target is **coded diagnoses on the next admission**, not “true” dyslipidemia or cardiovascular risk; models may disagree with labels on clinically plausible grounds.
3. **Template drift:** Prompt text in `prompts/*.txt` may change after this report; always cite **git revision** or archive templates with results.
4. **API and model versioning:** OpenAI may route `gpt-4o-mini` to a **dated snapshot** (e.g. `gpt-4o-mini-2024-07-18`); replication should record the resolved model string from `llm_*_metrics.json`.
5. **Cost and time:** Full runs issue thousands of API calls; caching and `rate_limit_rpm` materially affect duration and repeatability when prompts change.

---

## 14. Related Documentation

| Document | Role |
|----------|------|
| `METHODOLOGY.md` | End-to-end pipeline and configuration index |
| `docs/llm_prompt_modes_explained.md` | Conceptual comparison of the three base prompt styles |
| `ML_AND_LLM_INPUTS.md` | Inputs and artifact paths (if present in repo) |

---

*Report generated to describe the implemented LLM experiment and the results files currently under `data/outputs/mimiciii/`. Update this file when you change configs, prompts, or reporting conventions.*
