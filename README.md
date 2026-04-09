# EHR-Agentic-AI

![Workflow](image/image.png)

This README summarizes the repository’s MIMIC-III next-visit prediction pipeline, including ML and LLM input pathways, implemented approaches, archived results, documentation map, and reproducible run commands.

Predict whether **Disorders of Lipid Metabolism** (ICD-9 `272.x` / CCS-53) appear on a patient’s **next** hospital admission, using the **current** admission only — implemented on **MIMIC-III** visit pairs with classical ML baselines and **OpenAI** LLM prompts (zero-shot, zero-shot+, few-shot, EHR-CoAgent).

---

## Table of Contents
- [1. How inputs are fed: ML vs LLM](#1-how-inputs-are-fed-ml-vs-llm)
- [2. Approaches](#2-approaches)
- [3. Results (full pipeline run)](#3-results-full-pipeline-run)
- [4. Documentation](#4-documentation)
- [5. Repository layout (main)](#5-repository-layout-main)
- [6. Quick start](#6-quick-start)
- [7. License and data use](#7-license-and-data-use)

## 1. How inputs are fed: ML vs LLM

| Track | What the model sees | Where it comes from |
|--------|---------------------|---------------------|
| **Classical ML** | **Bag-of-codes** features: diagnosis, procedure, and medication strings from the **current** visit are tokenized into a sparse binary representation (`feature_type: bag_of_codes` in `configs/default.yaml`). | Built in `src/ml/` from `train.csv` / `test.csv`. |
| **LLM** | A single **natural-language narrative** per visit pair: bulleted **diagnoses**, **medications**, and **procedures** for the **current** visit (`narrative_current`). | Built during preprocessing; templates in `prompts/` fill `{{ narrative }}` (and optional few-shot `{{ demonstration_cases }}`, co-agent `{{ critic_feedback }}`). |

The **label** is the same for both tracks: derived from **next-visit** diagnosis codes only (the model never sees the next visit text). See `METHODOLOGY.md` and `src/data/build_target_labels.py`.

---

## 2. Approaches

**Classical ML** (`src/scripts/run_ml_baselines.py`)

- **Fully supervised** — Decision tree, logistic regression, random forest trained on the **full** training split.
- **Few-shot ML** — The same three models each trained on only **N = 6** labeled rows (`ml.few_shot_n`), same bag-of-features (not the same mechanism as LLM few-shot).

**LLM (current default)** (`local_gemma`, `src/scripts/run_*.py`)

- **Zero-Shot** — Instructions + current narrative only.
- **Zero-Shot+** — Adds prevalence context and longer analysis instructions (still no labeled examples).
- **Few-Shot (N=6)** — Six **in-context** exemplars (3 positive / 3 negative) sampled from **train** only; **no weight updates**.
- **EHR-CoAgent** — Few-shot-style prompt plus **critic** feedback from mispredictions on a **calibration** subset of train, then consolidated text injected into `predictor_coagent_base.txt`.

Details: `docs/llm_prompt_modes_explained.md`, `docs/LLM_EXPERIMENT_REPORT_GPT4O_MINI.md`.

---

## 3. Results (full pipeline run)

Aggregated metrics (**ACC**, **Sensitivity**, **Specificity**, **F1**, %) on the held-out test set. Historical GPT-4o-mini results are preserved for comparison; new local Gemma runs default to `data/outputs/mimiciii_gemma/summary_table.csv`.

| Model | Approach | ACC | Sensitivity | Specificity | F1 |
|--------|----------|-----|-------------|-------------|-----|
| decision_tree | Fully Supervised | 73.48 | 50.66 | 82.09 | 51.15 |
| logistic_regression | Fully Supervised | 77.05 | 47.73 | 88.11 | 53.27 |
| random_forest | Fully Supervised | 76.44 | 24.01 | 96.24 | 35.85 |
| decision_tree | Few Shot | 61.40 | 56.52 | 63.24 | 44.52 |
| logistic_regression | Few Shot | 35.27 | 90.34 | 14.48 | 43.34 |
| random_forest | Few Shot | 34.95 | 92.53 | 13.21 | 43.81 |
| gpt-4o-mini-2024-07-18 | Zero-Shot | 80.82 | 57.69 | 89.55 | 62.24 |
| gpt-4o-mini-2024-07-18 | Zero-Shot+ | 80.73 | 56.22 | 89.99 | 61.54 |
| gpt-4o-mini-2024-07-18 | Few-Shot (N=6) | 80.65 | 57.39 | 89.44 | 61.93 |
| gpt-4o-mini-2024-07-18 | EHR-CoAgent | 80.93 | 58.21 | 89.50 | 62.57 |

**Intra- and inter-approach discussion** (rankings, confusion matrices, ML vs LLM): `docs/COMPARATIVE_RESULTS_REPORT_ML_AND_LLM_GPT4O_MINI.md`.

---

## 4. Documentation

| Document | Purpose |
|----------|---------|
| `METHODOLOGY.md` | Data, split, configs, scripts, outputs |
| `docs/LLM_EXPERIMENT_REPORT_GPT4O_MINI.md` | LLM experiment details, parsing, artifacts |
| `docs/llm_prompt_modes_explained.md` | Zero-shot vs zero-shot+ vs few-shot (LLM) |
| `docs/COMPARATIVE_RESULTS_REPORT_ML_AND_LLM_GPT4O_MINI.md` | Full ML + LLM tables and comparisons |

---

## 5. Repository layout (main)

| Path | Role |
|------|------|
| `configs/default.yaml` | Main experiment configuration |
| `configs/debug_small.yaml` | Smoke-test overrides |
| `prompts/` | Jinja2 LLM templates |
| `src/` | Preprocessing, ML, LLM, evaluation |
| `scripts/` | Shell drivers (`run_mimic_pipeline.sh`, `run_llm_only.sh`, smoke tests) |
| `slurm/` | Cluster job scripts |
| `data/outputs/mimiciii_gemma/` | Metrics, `summary_table.csv`, LLM raw responses for new Gemma runs |
| `gpt_4o_mini_results/` | Archived historical GPT-4o-mini outputs/logs (gitignored for privacy) |

---

## 6. Quick start

1. **Python 3.10+**, virtual environment recommended.

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment variables**
   - For local Gemma (default), first authenticate with Hugging Face if model access is gated:
     ```bash
     huggingface-cli login
     ```
   - Optional overrides:
     ```bash
     export LLM_MODEL_NAME="google/gemma-4-e4b-it"  # switch later to e2b/26b/31b ids
     export HF_HOME="$HOME/.cache/huggingface"
     ```
   - For optional vLLM endpoint routing, set:
     ```bash
     export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
     ```

4. **MIMIC-III data** — Point `configs/default.yaml` → `paths.mimic_raw` at your local PhysioNet extract.

5. **Preserve historical GPT outputs (one-time)**:

   ```bash
   export PYTHONPATH="$(pwd)"
   python -m src.scripts.archive_gpt4o_results
   ```

6. **Run from repo root** with `PYTHONPATH` set:

   ```bash
   export PYTHONPATH="$(pwd)"
   STAGE=full bash scripts/run_mimic_pipeline.sh
   ```

   LLM prompt-only (after `train.csv` / `test.csv` exist):

   ```bash
   export PYTHONPATH="$(pwd)"
   bash scripts/run_llm_only.sh
   ```

   Fine-tuning (LoRA) with Gemma (`gemma-4-e4b-it`):

   ```bash
   export PYTHONPATH="$(pwd)"
   # Optional first run if processed/train.csv and processed/test.csv are not present yet:
   # python -m src.scripts.run_preprocessing --config configs/default.yaml

   # 1) Smoke test first: tiny train/val split + a few optimizer steps + validation logging
   python -m src.scripts.run_finetune_gemma --config configs/default.yaml --overrides configs/debug_small.yaml --smoke

   # 2) Full fine-tuning experiment (reuses generated train_ft.csv / val_ft.csv by default)
   python -m src.scripts.run_finetune_gemma --config configs/default.yaml
   ```

   Smoke test (small subset, `STAGE=smoke`):

   ```bash
   STAGE=smoke bash scripts/run_mimic_pipeline.sh
   ```

7. **Cluster** — Example: `sbatch slurm/mimic_iii_full.slurm` (see script headers and `scripts/slurm_llm_env.sh`).

---

## 7. License and data use

MIMIC-III requires **PhysioNet credentialing**; use of this code does not grant data access. Follow your institution’s rules for PHI and API keys.
