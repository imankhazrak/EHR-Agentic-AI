# Fine-tuning diagnosis — flat loss (~15.46) and low recall

**Scope:** Binary next-visit **Disorders of Lipid Metabolism** prediction from `narrative_current`, Gemma + LoRA, answer-only CE after `Prediction:\n`.  
**Method:** Code and configs in this repository at report time; **numeric evidence** from `data/processed/mimiciii/*.csv`, `logs/slurm-gemma4-ft-answer-only-5epochs-46675663.out`, and `data/outputs/mimiciii_gemma/llm_finetuned_test_*` where applicable.

**Important caveat:** `llm_finetuned_test_metrics.json` references adapter path `models/gemma_finetuned_answer_only`, while the long Slurm job `46675663` writes to `models/gemma_finetuned_answer_only_5epochs` per `configs/finetune_answer_only_5epochs.yaml`. **Test metrics in Section 6 are for the on-disk artifact named in that JSON, not automatically the same checkpoint as job 46675663** unless you re-ran eval with matching `--overrides`.

---

## STEP 1 — Data pipeline (verified)

### Files

- `data/processed/mimiciii/train.csv`, `test.csv` (and `train_ft.csv` / `val_ft.csv` produced for fine-tune).

### Label distribution (computed)

| Split | Rows | Positives | Prevalence |
|-------|------|-----------|------------|
| `train.csv` | 9964 | 2733 | **0.2743** |
| `test.csv` | 2492 | 683 | **0.2741** |

- **Duplicate `pair_id`:** 0 in train and test (checked).
- **Trivial empty narratives:** 0 rows with `narrative_current` length &lt; 20 characters (checked).
- **String scan for obvious label leakage in narrative text:** 0 rows containing literal substrings `next visit`, `hadm_id_next`, `admittime_next`, `diagnoses_codes_next` in `narrative_current` on full train (checked). This does **not** rule out all semantic leakage (e.g. current visit already lists lipid-related dx); it rules out **naive paste** of next-visit columns into the narrative.

### Narrative construction (code)

`narrative_current` is built only from **current** visit code columns:

```77:80:src/data/narrative_builder.py
        dx_names = mapper.map_code_list(row.get("diagnoses_codes_current", ""), "diagnosis")
        px_names = mapper.map_code_list(row.get("procedures_codes_current", ""), "procedure")
        med_names = mapper.map_code_list(row.get("medications_current", ""), "medication")
```

**Conclusion (Step 1):** Label prevalence ~**27.4%** on train/test; no duplicate pair ids; narratives are not trivially empty; **by design** the narrative does not concatenate `*_next` fields. **Patient-level split is not enforced** (separate finding under Step 6 / methodology).

---

## STEP 2 — Input construction (verified)

### Code path

```33:38:src/llm/finetune_gemma.py
def _row_to_instruction(narrative: str, label: int) -> str:
    return (
        "Task: Predict whether Disorders of Lipid Metabolism will be present in the next visit.\n\n"
        f"Patient record:\n{narrative}\n\n"
        f"Prediction:\n{_label_to_text(label)}"
    )
```

- **“Next visit”** is stated explicitly (`"in the next visit"`).
- **Output format in training string** is exactly **`Prediction:\n` + `Yes` or `No`** (`_label_to_text`).
- **Generic task text:** The task line is **identical for every patient**; only `narrative_current` changes. That is intentional, not ambiguous, but it is **not** a rich rubric (no prevalence block, no chain-of-thought in the supervised target).

**Conclusion (Step 2):** Prompt matches the intended framing; strict Yes/No tail; weakness is **sparse target text** (no reasoning token supervision), not ambiguous wording.

---

## STEP 3 — Tokenization & masking (critical)

### Answer-only masking (code)

```192:218:src/llm/finetune_gemma.py
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[int]]:
        out = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        labels = [ids.copy() for ids in out["input_ids"]]
        if answer_only_loss:
            prompts: List[str] = []
            for txt in batch["text"]:
                marker = "\nPrediction:\n"
                idx = txt.find(marker)
                if idx >= 0:
                    prompts.append(txt[: idx + len(marker)])
                else:
                    prompts.append(txt)
            prompt_tok = tokenizer(
                prompts,
                max_length=max_length,
                truncation=True,
                padding=False,
            )
            for i, prompt_ids in enumerate(prompt_tok["input_ids"]):
                plen = min(len(prompt_ids), len(labels[i]))
                labels[i][:plen] = [-100] * plen
        out["labels"] = labels
        return out
```

**Verified behavior:**

- Positions aligned with **tokenized prefix through `\\nPrediction:\\n`** are set to **`-100`** (ignored in CE).
- **Only** unmasked label positions (after that prefix, non-padding after collator) contribute to loss.

**Risks (code-level, must be monitored):**

1. **Marker collision:** If `narrative_current` ever contained the substring `\\nPrediction:\\n`, masking would attach to the **wrong** position. The train string scan in Step 1 found **0** such cases on literal phrases like `next visit` / `*_next`; a dedicated grep for `\\nPrediction:\\n` inside `narrative_current` is still advisable in future audits.
2. **Dual tokenization alignment:** Prompt length comes from a **second** `tokenizer()` call on the prompt substring. When **both** full text and prompt hit `max_length` truncation, prefix/suffix alignment can in principle diverge; the repository ships `src/scripts/verify_finetune_feed.py` to detect **truncation eating `Yes`/`No`**. **This report did not successfully execute tokenizer load on the login environment used for automation** (Gemma tokenizer load failed); **run `verify_finetune_feed` on the same conda env / node as training** for a hard pass/fail.
3. **Short supervision:** Supervised region is **at minimum the tokenization of `Yes` or `No`** (often 1–2 tokens). That implies **small gradient variance** per example on CE — consistent with flat logged loss, not proof of correctness alone.

### `max_length` and truncation (config)

Merged config for the 5-epoch job:

- `configs/default.yaml` → `llm.finetune.max_length: **2048**`
- `configs/finetune_answer_only_5epochs.yaml` does not override `max_length` (still **2048**).

**Empirical length check (characters, not tokens):** On a sample of **5000** train rows, full instruction string length via `_row_to_instruction`: **min 229, max 3603, mean ~1399** characters; **0** rows &gt; 8000 chars. This suggests **severe truncation of the answer tail is unlikely at 2048 tokens** for typical rows, but **only token-level verification** is definitive (see `verify_finetune_feed`).

**Collator:** `DataCollatorForLanguageModeling(..., mlm=False)` maps padding label ids to `-100` during training so padding does not contribute to loss (HF default behavior).

**Conclusion (Step 3):** Masking logic matches the intended answer-only recipe; **flat loss is compatible with very few supervised tokens per sequence**; **truncation risk must be closed with `verify_finetune_feed` on a Gemma-capable host**.

---

## STEP 4 — Training configuration (verified from YAML + Slurm)

### Effective merged settings

| Item | Source | Value |
|------|--------|--------|
| `num_train_epochs` | `finetune_answer_only_5epochs.yaml` | **5** |
| `max_length` | `default.yaml` | **2048** |
| `learning_rate` | `default.yaml` | **5e-5** |
| `batch_size` (per device) | `default.yaml` | **2** |
| `lora_r` / `lora_alpha` / `lora_dropout` | `default.yaml` | **8 / 16 / 0.05** |
| `gradient_checkpointing` | `default.yaml` | **true** |
| `force_fp32` | `default.yaml` | **true** |
| `max_grad_norm` | `default.yaml` | **1.0** |
| `evaluation_strategy` | `default.yaml` | **epoch** |
| `warmup_ratio` / `weight_decay` / `grad_accum` | `default.yaml` | **0 / 0 / 1** |
| `oversample_positive_copies` | `default.yaml` | **0** |

### Slurm driver

```39:39:slurm/gemma4_ft_full_answer_only_5epochs.slurm
python -m src.scripts.run_finetune_gemma --config configs/default.yaml --overrides configs/finetune_answer_only_5epochs.yaml
```

### Optimizer (not overridden in repo)

`TrainingArguments` does **not** set a custom optimizer in `finetune_gemma.py`; **Hugging Face `Trainer` defaults** apply (typically **AdamW** with library defaults for betas/eps unless changed by the installed `transformers` version).

### Fine-tune split size (computed)

- `train_ft.csv`: **8469** rows, prevalence **0.2743** (positives **2323**).

### Evaluation

- **LR “too small”:** 5e-5 is **moderate/conservative** for LoRA; not extreme. **Not proven** either way without an LR sweep.
- **Training “too short”:** 5 epochs × ~8469/2 steps per epoch (order-of-magnitude **~21k** steps if one process consumes the dataset at batch 2) is **not** trivially “too short”; flat metrics suggest **diminishing returns**, not necessarily under-training.
- **LoRA capacity:** `r=8` is **low** relative to model size; **limited capacity** is a **plausible** contributor to small loss movement.

**Conclusion (Step 4):** Configuration is internally consistent; **no class rebalancing** in loss; **no warmup**; **LoRA rank is small**.

---

## STEP 5 — Logs and learning behavior (verified from Slurm log)

**Log file:** `logs/slurm-gemma4-ft-answer-only-5epochs-46675663.out`

### Training loss

- Logs show **step-level** `loss` strings around **~15.3–15.6** throughout (e.g. early `epoch` ~0.005 through later `epoch` ~3.57 in the tail sample).
- Prior analysis on this log (602 logged loss rows) gave **mean ~15.465**, **std ~0.084** — **statistically flat** epoch-to-epoch in aggregate.

### Evaluation loss (logged)

From the same log (only **3** `eval_loss` lines present at time of grep — consistent with **per-epoch** eval through epoch 3):

| epoch (log) | eval_loss |
|---------------|-----------|
| 1 | **15.48** |
| 2 | **15.48** |
| 3 | **15.48** |

**Interpretation:** **Eval CE is also flat** to two decimals in this file — not just train noise.

### `grad_norm`

- Logged as **`'0'`** repeatedly in this file.
- **Uncertainty:** This may reflect **logging / mixed precision / aggregation** rather than literal zero gradients. **Do not treat `grad_norm: 0` as proven “no gradients”** without checking `transformers` version and Trainer logging options.

### Loss vs “random baseline”

**Not asserted here.** For a full-vocabulary causal LM, interpreting a single scalar **~15.5** as “random” requires knowing **how much of the loss mass is on supervised positions** after masking and how label smoothing / fp16 behaves. **Step 3 + Step 6** are the right grounds for judgment.

**Conclusion (Step 5):** **Train and eval CE are flat** in the available log; this supports **plateau / weak CE sensitivity**, not a claim that the model is at a mathematically random baseline.

---

## STEP 6 — Model predictions (existing artifact — read caveat)

### Artifact

- `data/outputs/mimiciii_gemma/llm_finetuned_test_metrics.json`
- `data/outputs/mimiciii_gemma/llm_finetuned_test_results.csv`
- JSON field `"model": "models/gemma_finetuned_answer_only"` (**may differ** from `models/gemma_finetuned_answer_only_5epochs`).

### Metrics (from JSON — reproduced)

| Metric | Value |
|--------|-------|
| Accuracy | 69.98% |
| Sensitivity (recall, positive) | **6.0%** |
| Specificity | 94.14% |
| F1 | 9.88% |
| AUC | 53.08% |
| AUPRC | 29.54% |

### Confusion (from CSV — recomputed)

On `llm_finetuned_test_results.csv` (`n = 2492`):

| | Pred 0 | Pred 1 |
|--|--------|--------|
| **True 0** | TN 1703 | FP 106 |
| **True 1** | FN **642** | TP **41** |

### Prediction distribution

- **Predicted positive rate:** **5.9%** `Yes` (147 rows), **94.1%** `No` (2345 rows).
- **True positive prevalence on test:** **27.4%** (683 positives).

**Interpretation (grounded):** The saved evaluation shows **strong majority-class prediction behavior** (low predicted Yes rate vs 27% base rate) and **very low recall** — **consistent with “collapse toward No”** for that checkpoint, **independent** of whether train CE moves.

### Patient overlap train vs test (informational for “leakage” discussion)

Computed on `SUBJECT_ID`:

- **1013** subjects appear in **both** `train.csv` and `test.csv` (6410 unique train subjects, 2140 unique test subjects).

This matches `docs/METHODOLOGY.md`: split is **by pair row**, not by patient. It does **not** explain low recall by itself, but it **does** affect comparability to strict patient-level benchmarks.

**Conclusion (Step 6):** **Low recall is real on the saved finetuned eval artifact**, with **~5.9% predicted Yes** vs **~27%** positives — **majority bias in predictions** is supported by these files. **Re-run eval** after the 5-epoch job completes, with `--overrides` pointing at `finetune_output_dir: models/gemma_finetuned_answer_only_5epochs`, before treating Step 6 as the verdict for job `46675663`.

---

## STEP 7 — Sanity checks

### `verify_finetune_feed`

- Script: `src/scripts/verify_finetune_feed.py` (implements masking parity checks and truncation tail checks).
- **Status for this report:** **Not re-run successfully** in the automated environment (Gemma tokenizer load failure on the node used). **Action:** run on **the same conda env as `gemma4_ft_full_answer_only_5epochs.slurm`**.

Suggested command:

```bash
python -m src.scripts.verify_finetune_feed --config configs/default.yaml --overrides configs/finetune_answer_only_5epochs.yaml --n 200
```

### Manual decode of outputs

- `llm_finetuned_test_results.csv` shows `generated_text` mostly **`No`** with `parser_status` **`first_token_no`** dominating — consistent with Step 6.

**Conclusion (Step 7):** **Automated token alignment proof is pending** a successful `verify_finetune_feed` run; **saved CSV outputs** support consistent **single-token No** generations on the evaluated adapter.

---

## STEP 8 — Root cause ranking (evidence-weighted)

| Rank | Hypothesis | Evidence strength | Notes |
|------|------------|--------------------|-------|
| 1 | **Evaluation / decision behavior biased toward “No”** (low recall, low predicted Yes rate) | **Strong** for `models/gemma_finetuned_answer_only` test CSV/JSON | CE can stay flat while logits favor **No** on most rows. |
| 2 | **Weak supervision signal for CE** (few supervised tokens; answer-only) | **Strong** (code structure + flat train/eval CE) | Explains **flat loss**; does not automatically imply “no learning”. |
| 3 | **Limited LoRA capacity + conservative optimization** (`r=8`, LR 5e-5, no warmup) | **Medium** (config) | Plausible limiter for **large** behavioral change. |
| 4 | **Truncation / mis-alignment of mask vs labels** | **Unproven** here | Must be ruled in/out with `verify_finetune_feed` + token checks. |
| 5 | **Class imbalance (27% pos)** | **Strong as a task property**, **medium as sole cause of flat CE** | Explains **metric** difficulty; **not** a direct proof of flat **CE** without per-class loss curves. |
| 6 | **Patient overlap across train/test** | **Strong** (computed) | Affects **generalization claims**, not a direct explanation of **No**-bias alone. |

---

## STEP 9 — Actionable fixes (prioritized)

1. **Re-evaluate the correct adapter** after the 5-epoch run:  
   `python -m src.scripts.evaluate_finetuned_gemma_on_test --config configs/default.yaml --overrides configs/finetune_answer_only_5epochs.yaml`  
   so metrics match **`models/gemma_finetuned_answer_only_5epochs`**.

2. **Run `verify_finetune_feed`** (200+ rows) on the training node; if failures appear, fix **truncation** (e.g. left-truncate narrative, or raise `max_length`) before further tuning.

3. **Address majority bias for metrics** (not necessarily CE):  
   - `oversample_positive_copies` (already wired in `run_finetune_gemma.py`), and/or  
   - threshold tuning on `val_ft` using `prob_yes` from the eval script, and/or  
   - class-weighted / focal objectives (would require **new code**, not present today).

4. **Increase adaptation capacity / schedule:** repository already includes `configs/finetune_answer_only_5epochs_improved.yaml` (higher LoRA rank, cosine schedule, warmup, optional `load_best_model_at_end`). Treat as **starting point**, not a guarantee.

5. **Richer training target (optional):** supervised reasoning tokens (longer target) can increase CE signal — requires **prompt + label + loss mask** redesign and consistent eval parsing.

6. **Alternative modeling (strongest for pure discrimination):** classification head on pooled hidden states or a small classifier on embeddings — **different architecture**, not a tweak to the current `Trainer` CE path.

---

## Appendix — Commands used for counts in this report

- Pandas: label prevalence, duplicate `pair_id`, narrative length, `SUBJECT_ID` overlap train∩test, prediction rates from `llm_finetuned_test_results.csv`.
- `grep` / `wc` on `logs/slurm-gemma4-ft-answer-only-5epochs-46675663.out` for `eval_loss` and tail `loss` lines.

---

*End of report.*
