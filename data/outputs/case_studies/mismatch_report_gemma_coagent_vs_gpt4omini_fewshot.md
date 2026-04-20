# Mismatch Consistency Report

## Scope

This report documents the consistency check between probability values and predicted labels for:

- **GPT-4o-mini** (Few-Shot)
- **Gemma4** (CoAgent)

using the merged dataset:

- `data/outputs/case_studies/model_comparison_gemma_coagent_vs_gpt4omini_fewshot.csv`

## Rule Used

For each model, define a derived prediction from probability:

- `pred_from_prob = 1` if `prob >= 0.5`, else `0`

Then compute mismatch:

- `mismatch = 0` if `pred_from_prob == label` (no mismatch)
- `mismatch = 1` if `pred_from_prob != label` (mismatch)

Columns added to the CSV:

- `gpt_mismatch`
- `gemma_mismatch`

## Output Schema

Final CSV columns:

1. `case_id`
2. `true_label`
3. `gpt_prob`
4. `gemma_prob`
5. `gpt_label`
6. `gemma_label`
7. `gpt_mismatch`
8. `gemma_mismatch`

## Validation Summary

- Total rows: **2492**
- GPT mismatches (`gpt_mismatch = 1`): **0**
- Gemma mismatches (`gemma_mismatch = 1`): **10**

## Gemma Mismatch Case IDs

The following `case_id` values have `gemma_mismatch = 1`:

- `2494`
- `2579`
- `3056`
- `3237`
- `3385`
- `4364`
- `4697`
- `6305`
- `10492`
- `11290`

## Spot Check Example

For `case_id = 2494`:

- `gpt_prob = 0.75`, `gpt_label = 1` -> `gpt_mismatch = 0`
- `gemma_prob = 0.27`, `gemma_label = 1` -> `gemma_mismatch = 1`

This matches the defined rule and confirms the expected inconsistency in Gemma for this case.
