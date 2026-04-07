# MIMIC-III Next-Visit Prediction — Methodology and Project Steps

This methodology note summarizes the implemented end-to-end pipeline for MIMIC-III next-visit lipid-disorder prediction, including cohort construction, ML and LLM modeling tracks, evaluation setup, run configuration, and primary artifacts for reproducibility.

Concise reference for advising meetings. Code lives under `src/`; configuration under `configs/`.

## Table of Contents
- [1. Objective](#1-objective)
- [2. Data and cohort](#2-data-and-cohort)
- [3. Train / test split](#3-train--test-split)
- [4. Classical ML baselines](#4-classical-ml-baselines)
- [5. LLM experiments](#5-llm-experiments)
- [6. Evaluation metrics](#6-evaluation-metrics)
- [7. What we configured and ran](#7-what-we-configured-and-ran)
- [8. Primary output locations](#8-primary-output-locations)

## 1. Objective

Predict whether **Disorders of Lipid Metabolism** appear in a patient’s **next hospital admission**, given structured information from the **current** admission. Task is framed as **adjacent visit pairs** (current visit → next visit), binary label.

## 2. Data and cohort

| Item | Definition |
|------|------------|
| **Source** | MIMIC-III v1.4 CSVs (PhysioNet); raw path set in `configs/default.yaml` → `paths.mimic_raw`. |
| **Visits** | One row per **hospital admission** (`ADMISSIONS` + diagnoses, procedures, prescriptions). |
| **Pairs** | For each patient with ≥2 admissions, consecutive admissions form `(visit_t, visit_{t+1})` pairs. |
| **Label** | Positive if **any ICD-9 code for lipid disorder** appears in the **next** visit (CCS-53 style: codes `272.x`, stored without dot in MIMIC, e.g. `2720`, `2724`). |
| **Input text** | Natural-language **narratives** built from current visit: bullet lists of diagnoses (with titles), medications, procedures; capped per section in config. |

**Note:** The implementation uses **all hospital admissions**, not an ICU-only cohort. If the paper specifies ICU-linked current visits, that is a **documented difference** from this codebase unless ICU filtering is added later.

## 3. Train / test split

- **Method:** Random split with **stratification** on the binary label (`stratify_test: true`).
- **Fraction:** `test_size: 0.2` (default).
- **Unit:** **Visit-pair rows**, not patients — the same patient can contribute pairs to both train and test (possible **patient-level leakage**; no `split_by_patient` flag in current code).

Outputs: `data/processed/mimiciii/train.csv`, `test.csv`, plus `processed_dataset_summary.json` and `label_distribution.csv` under outputs.

## 4. Classical ML baselines

- **Features:** Bag-of-codes style encoding from diagnosis / procedure / medication strings (`feature_type: bag_of_codes`).
- **Models:** Decision Tree, Logistic Regression, Random Forest (see `configs/default.yaml` → `ml.models`).
- **Settings:** (1) **Fully supervised** on full training set; (2) **Few-shot ML** trained on **N = 6** labeled exemplars (balanced positive/negative by default).

Metrics saved: `ml_results_fully_supervised.csv`, `ml_results_few_shot.csv`, `ml_results.csv`.

## 5. LLM experiments

- **Provider / model:** OpenAI Chat Completions; default model **`gpt-4o-mini`** (`temperature: 0`, `max_tokens: 1024`).
- **Prompts:** Jinja2 templates in `prompts/` (zero-shot, zero-shot+, few-shot, coagent base; critic and consolidation prompts for EHR-CoAgent).
- **Parsing:** Model outputs are parsed for `Prediction: Yes` / `Prediction: No` (with fallbacks); unparseable responses are counted and excluded from metric denominators where implemented.
- **Modes (order of execution):**  
  1. Zero-shot  
  2. Zero-shot+  
  3. Few-shot (exemplars sampled from **train** only; IDs saved to `llm_few_shot_exemplars.csv`)  
  4. EHR-CoAgent (calibration few-shot → wrong predictions → critic batches → consolidated feedback → test with feedback); artifacts under `critic_feedback/`.

**Rate limiting:** `rate_limit_rpm` in config to avoid bursting the API on long runs.

**Authentication:** API key via repo-root **`.env`** (`OPENAI_API_KEY`); never committed (see `.gitignore`). SLURM jobs source `.env` through `scripts/slurm_llm_env.sh`.

## 6. Evaluation metrics

**Accuracy, sensitivity, specificity, F1** for both ML and LLM runs (see `src/evaluation/metrics.py`). Optional bootstrap CI is off by default.

Aggregated table: `data/outputs/mimiciii/summary_table.csv` (from `collect_results`).

## 7. What we configured and ran

| Artifact | Role |
|----------|------|
| `configs/default.yaml` | Main experiment: paths, split, narrative caps, ML, LLM, few-shot, coagent, evaluation. |
| `configs/debug_small.yaml` | Smoke: subsample pairs (`max_pairs`), smaller test fraction override, **fewer LLM test rows** (`max_test_samples`), smaller coagent calibration. |
| `requirements.txt` | Python dependencies for the pipeline. |
| `scripts/run_mimic_pipeline.sh` | Single shell driver: preprocess → ML → four LLM modes → summary (`STAGE=smoke` merges debug overrides). |
| `scripts/slurm_llm_env.sh` | Loads `.env` and checks that a provider API key is set before cluster jobs run LLM steps. |
| `slurm/mimic_iii_smoke.slurm` | Short wall time; smoke configuration; requires real API key in `.env`. |
| `slurm/mimic_iii_full.slurm` | Full pipeline; long wall time (e.g. 48h); same key requirement. |

**Steps completed in this project:**

1. **Repository setup:** YAML configs, paths to local MIMIC-III extract, `pip install -r requirements.txt`, virtual environment.
2. **Smoke test (optional):** `run_preprocessing` + `run_ml_baselines` + LLM scripts with `debug_small.yaml` to validate end-to-end behavior on a small subset.
3. **Full experiment:** Full preprocessing (no `max_pairs`), full ML, full test set for each LLM mode — run **on the cluster** via `sbatch slurm/mimic_iii_full.slurm` (or equivalent), with `OPENAI_API_KEY` in `.env` on the cluster filesystem.

**Operational note:** Long LLM jobs are **API-bound** (thousands of calls). Failures such as `PermissionDeniedError` from OpenAI indicate account/billing/model-access issues on the provider side and should be resolved in the OpenAI dashboard before interpreting LLM metrics.

## 8. Primary output locations

| Path | Contents |
|------|----------|
| `data/processed/mimiciii/` | `train.csv`, `test.csv`, `full_dataset.csv` |
| `data/outputs/mimiciii/` | Summaries, ML CSVs, `llm_*_metrics.json`, `llm_*_results.csv`, `raw_llm_responses/`, `prompts_used/`, `summary_table.csv`, coagent `critic_feedback/` |
| `logs/slurm-*.out` / `.err` | Cluster stdout/stderr for SLURM jobs |

---

*This document describes the implemented pipeline and configuration; alignment with every detail of the published paper should be checked against the paper text (e.g. ICU definition, exact split policy, and model choice).*
