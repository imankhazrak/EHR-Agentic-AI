# MIMIC-III local Gemma (Gemma 4) — LLM prompt experiments (Slurm 46510786) and Co-Agent completion (Slurm 46615354)

**Report generated:** 2026-04-10  

This document aggregates **prompt-based evaluation** from OSC Slurm job **`46510786`** (`mimic3_llm_gemma`), and the **completed EHR-CoAgent** test pass from follow-up job **`46615354`** (`mimic3_llm_coagent`). Both use **`google/gemma-4-e4b-it`** via the local snapshot in `configs/default.yaml` (`models/hf_snapshots/google--gemma-4-e4b-it`). Job **46510786** hit its wall-time limit before Co-Agent finished; job **46615354** ran the dedicated Co-Agent script to completion.

---

## 1. Job and environment summary

### 1.1 Slurm job 46510786 — Zero-Shot / Zero-Shot+ / Few-Shot

| Field | Value |
|--------|--------|
| **Slurm job ID** | `46510786` |
| **Job name** | `mimic3_llm_gemma` |
| **Submission script** | `slurm/mimic_iii_llm_gemma.slurm` |
| **Pipeline driver** | `scripts/run_llm_only.sh` (`START_FROM=zero_shot`, `STAGE=full`) |
| **Job start (log)** | 2026-04-07, 21:48:52 EDT |
| **Job end** | 2026-04-09, ~21:48:57 EDT — **wall-time limit reached** (Slurm `TIMEOUT`) |
| **GPUs (node)** | 2× NVIDIA V100-PCIE-16GB (`p0242`, per `logs/slurm-llm-gemma-46510786.out`) |
| **PyTorch (log)** | `2.4.1+cu118` |
| **Primary config** | `configs/default.yaml` → `paths.outputs: data/outputs/mimiciii_gemma` |
| **Model artifact** | `models/hf_snapshots/google--gemma-4-e4b-it` |

### 1.2 Slurm job 46615354 — EHR-CoAgent (full test pass)

| Field | Value |
|--------|--------|
| **Slurm job ID** | `46615354` |
| **Job name** | `mimic3_llm_coagent` |
| **Submission script** | `slurm/mimic_iii_llm_gemma_coagent_only.slurm` |
| **Pipeline** | `python -m src.scripts.run_coagent --config configs/default.yaml` (`START_FROM=coagent`, `STAGE=full`, per `logs/slurm-llm-gemma-coagent-46615354.out`) |
| **Job start → end** | 2026-04-09 22:40:01 → 2026-04-10 15:27:10 — **COMPLETED** (`ExitCode=0:0`) |
| **Elapsed** | **16:47:09** (time limit 2 days) |
| **Partition / node** | `gpu-exp` / `p0321` |
| **GPUs** | 2× **Tesla V100S-PCIE-32GB** |
| **Memory** | 128G requested; batch MaxRSS ~24.3 GB |
| **PyTorch (log)** | `2.10.0+cu128` |
| **Primary config** | `configs/default.yaml` → `data/outputs/mimiciii_gemma` |
| **Model artifact** | `models/hf_snapshots/google--gemma-4-e4b-it` |

**Co-Agent pipeline (stderr):** few-shot predictor on calibration **n = 200** → **48** incorrect → critic (**3 rounds**, `batch_size=10`) → LLM consolidation → test co-agent loop **2492** instances (all parseable).

---

## 2. Prediction task (what was evaluated)

- **Outcome:** Binary prediction of **Disorders of Lipid Metabolism** (ICD-9 272.x family in this codebase) at the **next visit**, from the **current-visit narrative** (`narrative_current`).
- **Split:** Held-out **test** rows merged by `pair_id` with model outputs (same protocol as `src.evaluation.evaluate_llm_runs.evaluate_llm_results`).
- **Modes in this report:**  
  **Zero-Shot**, **Zero-Shot+**, **Few-Shot** (job **46510786**); **EHR-CoAgent** (job **46615354**).
- **Metrics:** Accuracy, sensitivity (recall on positives), specificity, macro-oriented **F1** as implemented in `src/evaluation/metrics.py` (positive class = 1).

**Evaluation cohort size:** **`n_total = 2492`** test instances per mode (after merging predictions with labels). Confusion-matrix counts sum to **`n_valid`** rows used to compute metrics (responses mapped to **Yes** → 1, **No** → 0).

---

## 3. Quantitative results — all approaches

### 3.1 Summary table (primary endpoints)

Values are **percent points** (0–100), matching the JSON exports.

| Approach | Accuracy | Sensitivity | Specificity | F1 | Job |
|----------|----------|-------------|-------------|-----|-----|
| **Zero-Shot** | 80.26 | 55.93 | 89.44 | 60.83 | 46510786 |
| **Zero-Shot+** | 80.97 | 60.61 | 88.66 | 63.59 | 46510786 |
| **Few-Shot (N = 6)** | 80.10 | 57.83 | 88.50 | 61.43 | 46510786 |
| **EHR-CoAgent** | **79.78** | **62.96** | **86.12** | **63.05** | **46615354** |

Machine-readable metrics:

- Zero- / Few-Shot: `data/outputs/mimiciii_gemma/llm_*_metrics.json`
- Co-Agent: `data/outputs/mimiciii_gemma/llm_coagent_metrics.json`

**Note on `summary_table.csv`:** `data/outputs/mimiciii_gemma/summary_table.csv` is produced by `src/evaluation/summarize_results.py` and may still list **only the three baseline LLM rows** if aggregation did not pick up Co-Agent (job **46615354** stderr reported `No results found to summarise`). For Co-Agent, treat **`llm_coagent_metrics.json`** as authoritative.

### 3.2 Confusion matrices (test merge, parse-valid subset)

| Approach | TP | FP | TN | FN | `n_valid` | `n_unparseable` |
|----------|----|----|----|----|-----------|-----------------|
| Zero-Shot | 382 | 191 | 1618 | 301 | 2492 | 0 |
| Zero-Shot+ | 414 | 205 | 1603 | 269 | 2491 | 1 |
| Few-Shot (N = 6) | 395 | 208 | 1601 | 288 | 2492 | 0 |
| **EHR-CoAgent** | **430** | **251** | **1558** | **253** | **2492** | **0** |

**Note on Zero-Shot+:** One test instance had a **non-parseable** model response (`n_unparseable = 1`). Metrics were computed on the **2491** rows with valid **Yes/No** parses.

---

## 4. Interpretation (short)

- **Overall accuracy** for the three baseline prompt settings is similar (~**80.1%–81.0%**), with **Zero-Shot+** slightly ahead on **accuracy** and **F1**.
- **Trade-off (Zero-Shot vs Zero-Shot+):** Zero-Shot+ improves **sensitivity** (**60.61%** vs. **55.93%**) at a small **specificity** cost (**88.66%** vs. **89.44%**).
- **Few-Shot (N = 6)** in job **46510786** sits between the two zero-shot variants on sensitivity and underperforms Zero-Shot+ on F1; one realized seed / exemplar draw (config seed **42**).
- **EHR-CoAgent (job 46615354):** **Sensitivity 62.96%** is the **highest** among these four settings; **specificity 86.12%** is **lower** than the three baselines. **Accuracy 79.78%** is slightly below Zero-Shot / Zero-Shot+ / Few-Shot; **F1 63.05** is between Zero-Shot+ (**63.59**) and Few-Shot (**61.43**). This is a **single** calibration + critic realization, not a multi-seed study.

These statements describe **saved artifacts from these Slurm runs** only.

---

## 5. EHR-CoAgent: partial run (46510786) vs completion (46615354)

### 5.1 Job 46510786 — timed out during Co-Agent

The pipeline reached the **Co-Agent** test loop but the job was **terminated by the wall-time limit** before completion.

| Field | Value |
|--------|--------|
| **Partial responses** | **236** saved JSON files under `raw_llm_responses/coagent/` when the earlier report was built |
| **Target test size** | **2492** |
| **Approx. completion** | **236 / 2492 ≈ 9.5%** |
| **Metrics file** | Not produced in this job |

### 5.2 Job 46615354 — full Co-Agent evaluation

| Field | Value |
|--------|--------|
| **State** | **COMPLETED** (`ExitCode=0:0`) |
| **Test predictions** | **2492 / 2492** with **0** unparseable |
| **Metrics** | ACC **79.78**, Sens **62.96**, Spec **86.12**, F1 **63.05** (see §3) |
| **Merged results** | `data/outputs/mimiciii_gemma/llm_coagent_results.csv` |
| **Metrics JSON** | `data/outputs/mimiciii_gemma/llm_coagent_metrics.json` |

---

## 6. Artifact index (reproducibility)

| Content | Path |
|---------|------|
| Zero-Shot metrics | `data/outputs/mimiciii_gemma/llm_zero_shot_metrics.json` |
| Zero-Shot+ metrics | `data/outputs/mimiciii_gemma/llm_zero_shot_plus_metrics.json` |
| Few-Shot metrics | `data/outputs/mimiciii_gemma/llm_few_shot_metrics.json` |
| **Co-Agent metrics** | **`data/outputs/mimiciii_gemma/llm_coagent_metrics.json`** |
| Per-instance merged results | `…/llm_zero_shot_results.csv`, `…_zero_shot_plus_results.csv`, `…_few_shot_results.csv`, **`…/llm_coagent_results.csv`** |
| Raw LLM JSON (per `pair_id`) | `data/outputs/mimiciii_gemma/raw_llm_responses/<mode>/` |
| Saved prompts | `data/outputs/mimiciii_gemma/prompts_used/<mode>/` |
| LLM cache (deduplicated calls) | `data/outputs/mimiciii_gemma/llm_cache/` |
| Slurm logs — job **46510786** | `logs/slurm-llm-gemma-46510786.out`, `logs/slurm-llm-gemma-46510786.err` |
| Slurm logs — job **46615354** | **`logs/slurm-llm-gemma-coagent-46615354.out`**, **`logs/slurm-llm-gemma-coagent-46615354.err`** |

---

## 7. Methods reference (code)

- Prompt execution and saving: `src/llm/predictor.py` (`run_predictions`).
- Co-Agent orchestration (calibration, critic, re-run): `src/llm/coagent.py`.
- Metric definitions: `src/evaluation/metrics.py`.
- Evaluation wrapper: `src/evaluation/evaluate_llm_runs.py` (`evaluate_llm_results`).
- Aggregated table helper: `src/evaluation/summarize_results.py` (`collect_results`).

---

*End of report.*
