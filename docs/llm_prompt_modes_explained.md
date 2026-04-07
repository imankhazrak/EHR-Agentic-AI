# LLM prediction modes: zero-shot, zero-shot+, and few-shot

This guide explains, in end-to-end practical terms, how the three base LLM prompting modes differ in this repository and what is actually passed to the model at inference time.

This note explains **what actually happens** in this repository when you run the three LLM approaches on the MIMIC-III lipid-disorder **next-visit** task. Use it for your own review or to brief someone else.

**Scope:** LLM modes only (not EHR-CoAgent, not classical ML). Implementation details live in `prompts/predictor_*.txt`, `src/llm/predictor.py`, `src/llm/prompt_builder.py`, and `src/data/exemplar_selector.py`.

---

## Table of Contents
- [1. The task (identical for all three modes)](#1-the-task-identical-for-all-three-modes)
- [2. Example test patient (shortened)](#2-example-test-patient-shortened)
- [3. Zero-shot — end-to-end](#3-zero-shot--end-to-end)
- [4. Zero-shot+ — what changes](#4-zero-shot--what-changes)
- [5. Few-shot (LLM) — what changes](#5-few-shot-llm--what-changes)
- [6. Summary table](#6-summary-table)
- [7. Contrast: “Few Shot” for **classical ML** in `summary_table.csv`](#7-contrast-few-shot-for-classical-ml-in-summary_tablecsv)
- [8. Related files](#8-related-files)

## 1. The task (identical for all three modes)

- **Input:** A text **narrative** for the patient’s **current** hospital visit (`narrative_current` in `test.csv`): bullet lists of diagnoses, medications, procedures.
- **Question:** Will **“Disorders of Lipid Metabolism”** appear as a diagnosis on the patient’s **next** hospital visit?
- **Model output:** Natural language; the pipeline expects a first line like `Prediction: Yes` or `Prediction: No`, then optional reasoning (`src/llm/output_parser.py`).
- **Evaluation:** Parsed Yes/No is compared to the binary label `label_lipid_disorder` (derived from **next** visit ICD codes, not shown to the model).

The **only** thing that differs across zero-shot, zero-shot+, and few-shot is **the text of the prompt** sent to the API. There is **no** fine-tuning or weight updates for these modes.

---

## 2. Example test patient (shortened)

Same fictional row for all three modes below:

```text
- Diagnoses: Congestive heart failure; Atrial fibrillation
- Medications: Furosemide; Warfarin; Atorvastatin
- Procedures: None recorded
```

In a real run, this string comes from one row of `data/processed/mimiciii/test.csv` (one `pair_id`). The true label is computed from the **next** admission’s diagnoses; the model does **not** see the next visit.

---

## 3. Zero-shot — end-to-end

1. Load the test row from `test.csv`.
2. Render `prompts/predictor_zero_shot.txt` with Jinja: task rules, target definition, prediction clarification, then **`{{ narrative }}`** for this patient only.
3. Build chat messages (`system` + `user`) via `build_messages` in `src/llm/prompt_builder.py`.
4. Call **`gpt-4o-mini`** once per row (`temperature: 0` in `configs/default.yaml`).
5. Parse the reply for Yes/No; merge with labels in `src/evaluation/evaluate_llm_runs.py`.

**What the user message is like (schematic, not the full template):**

```text
Task: ... predict whether "Disorders of Lipid Metabolism" will be present ...

[Target Definition]
... hypercholesterolemia, hyperlipidemia, ...

[Prediction Clarification]
... e.g. do not answer Yes based only on statins or CHF alone ...

[Current Visit Record]
- Diagnoses: Congestive heart failure; Atrial fibrillation
- Medications: Furosemide; Warfarin; Atorvastatin
- Procedures: None recorded

[Output Format]
Prediction: Yes / No on first line, then reasoning.
```

**Not included:** the ~27.6% prevalence paragraph, the long structured **[Instructions]** block used in zero-shot+, and **no** demonstration cases from the training set.

---

## 4. Zero-shot+ — what changes

Same test row, same API, same “one call per row.”

Template: `prompts/predictor_zero_shot_plus.txt`.

**Additions vs zero-shot:**

- Structured **[Instructions]** (longer analysis recipe: risk factors, medications, procedures, step-by-step reasoning).
- Explicit note that in the MIMIC-III population roughly **27.6%** of **subsequent** visits carry the target diagnosis.

**Still no** labeled examples from `train.csv`. This is **prompt engineering only**, not training.

---

## 5. Few-shot (LLM) — what changes

Same test row, still **one completion per test row**, still **no weight updates**.

Template: `prompts/predictor_few_shot.txt`.

**Before** rendering the prompt for each test patient, the pipeline selects exemplars from **`train.csv`** only (default: **6** cases — e.g. 3 positive and 3 negative — see `configs/default.yaml` under `few_shot`).

They are formatted by `format_exemplar_block` in `src/data/exemplar_selector.py` as:

```text
Case 1:
<narrative_current of exemplar 1>
Outcome: Yes

Case 2:
<narrative_current of exemplar 2>
Outcome: No

... through Case 6 ...
```

That string is inserted as **`{{ demonstration_cases }}`**. The instructions tell the model to learn from these cases, then the **current** test narrative appears under **[Current Visit Record]** as `{{ narrative }}`.

This is **in-context learning**: the “training” is **text pasted into the prompt**, not gradient descent.

---

## 6. Summary table

| Step | Zero-shot | Zero-shot+ | Few-shot (LLM) |
|------|-----------|------------|----------------|
| Read test row | Yes | Yes | Yes |
| Add rules + target definition | Yes | Yes | Yes |
| Add long instructions + prevalence | No | Yes | Yes |
| Paste 6 train examples with `Outcome: Yes/No` | No | No | Yes |
| One API call per test row | Yes | Yes | Yes |
| Update LLM weights | No | No | No |

---

## 7. Contrast: “Few Shot” for **classical ML** in `summary_table.csv`

In the ML rows labeled **Few Shot**, models such as logistic regression or random forest **actually fit parameters** on only **N = 6** labeled training examples (bag-of-codes features). That **is** real training on a tiny dataset.

LLM **few-shot** does **not** do that; it only adds **demonstration text** to the prompt.

---

## 8. Related files

| Role | Path |
|------|------|
| Zero-shot template | `prompts/predictor_zero_shot.txt` |
| Zero-shot+ template | `prompts/predictor_zero_shot_plus.txt` |
| Few-shot template | `prompts/predictor_few_shot.txt` |
| Mode → template map | `src/llm/predictor.py` (`MODE_TEMPLATE_MAP`) |
| Exemplar formatting | `src/data/exemplar_selector.py` (`format_exemplar_block`) |
| High-level methodology | `METHODOLOGY.md` |

---

*Document generated to match the project’s implemented behavior; if templates or config change, update this file or regenerate from the repo.*
