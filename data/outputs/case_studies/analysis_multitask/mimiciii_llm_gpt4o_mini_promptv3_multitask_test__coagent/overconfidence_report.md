# Multitask overconfidence report

Per-task probability histograms and descriptive bin statistics (GPT-4o-mini vs Gemma when available).
This report is descriptive only; it does not apply calibration or retraining.

## Data sources

- **GPT CSV:** `/users/PCS0229/imankhazrak/EHR-Agentic-AI/data/outputs/mimiciii_llm_gpt4o_mini_promptv3_multitask_test/llm_coagent_results.csv`
- **Test CSV:** `/users/PCS0229/imankhazrak/EHR-Agentic-AI/data/processed/mimiciii_multitask/test.csv`
- **Gemma:** omitted

**Note:** Gemma was omitted or not merged; GPT-only metrics are shown.

## Lipid Disorder (Next Visit) (`lipid`)

- Rows with non-null **GPT** probability for this task: **2492**
- **Gemma:** not included in this run (no Gemma source or no usable probabilities).

### Extreme probabilities

| Model | Count(0) | Count(1) | Total | % Extreme |
| --- | --- | --- | --- | --- |
| GPT-4o-mini | 1663 | 0 | 2492 | 66.73% |

### Probability bin analysis

#### GPT-4o-mini

| Probability Bin | Count | Avg True Label |
| --- | --- | --- |
| [0.0, 0.2] | 1721 | 0.1238 |
| (0.2, 0.4] | 137 | 0.4015 |
| (0.4, 0.6] | 100 | 0.5700 |
| (0.6, 0.8] | 534 | 0.6704 |
| (0.8, 1.0] | 0 | NA |


## Diabetes (Current Visit) (`diabetes`)

- Rows with non-null **GPT** probability for this task: **2492**
- **Gemma:** not included in this run (no Gemma source or no usable probabilities).

### Extreme probabilities

| Model | Count(0) | Count(1) | Total | % Extreme |
| --- | --- | --- | --- | --- |
| GPT-4o-mini | 1007 | 517 | 2492 | 61.16% |

### Probability bin analysis

#### GPT-4o-mini

| Probability Bin | Count | Avg True Label |
| --- | --- | --- |
| [0.0, 0.2] | 1014 | 0.0000 |
| (0.2, 0.4] | 0 | NA |
| (0.4, 0.6] | 0 | NA |
| (0.6, 0.8] | 611 | 0.0147 |
| (0.8, 1.0] | 867 | 0.9412 |


## Hypertension (Current Visit) (`hypertension`)

- Rows with non-null **GPT** probability for this task: **2492**
- **Gemma:** not included in this run (no Gemma source or no usable probabilities).

### Extreme probabilities

| Model | Count(0) | Count(1) | Total | % Extreme |
| --- | --- | --- | --- | --- |
| GPT-4o-mini | 603 | 393 | 2492 | 39.97% |

### Probability bin analysis

#### GPT-4o-mini

| Probability Bin | Count | Avg True Label |
| --- | --- | --- |
| [0.0, 0.2] | 648 | 0.0000 |
| (0.2, 0.4] | 0 | NA |
| (0.4, 0.6] | 12 | 0.0000 |
| (0.6, 0.8] | 964 | 0.5290 |
| (0.8, 1.0] | 868 | 0.9724 |


## Obesity (Current Visit) (`obesity`)

- Rows with non-null **GPT** probability for this task: **2492**
- **Gemma:** not included in this run (no Gemma source or no usable probabilities).

### Extreme probabilities

| Model | Count(0) | Count(1) | Total | % Extreme |
| --- | --- | --- | --- | --- |
| GPT-4o-mini | 1931 | 66 | 2492 | 80.14% |

### Probability bin analysis

#### GPT-4o-mini

| Probability Bin | Count | Avg True Label |
| --- | --- | --- |
| [0.0, 0.2] | 2336 | 0.0000 |
| (0.2, 0.4] | 0 | NA |
| (0.4, 0.6] | 4 | 0.0000 |
| (0.6, 0.8] | 23 | 0.6087 |
| (0.8, 1.0] | 129 | 0.9690 |


## Cardiovascular Condition (Next Visit) (`cardio`)

- Rows with non-null **GPT** probability for this task: **2492**
- **Gemma:** not included in this run (no Gemma source or no usable probabilities).

### Extreme probabilities

| Model | Count(0) | Count(1) | Total | % Extreme |
| --- | --- | --- | --- | --- |
| GPT-4o-mini | 462 | 18 | 2492 | 19.26% |

### Probability bin analysis

#### GPT-4o-mini

| Probability Bin | Count | Avg True Label |
| --- | --- | --- |
| [0.0, 0.2] | 473 | 0.1099 |
| (0.2, 0.4] | 0 | NA |
| (0.4, 0.6] | 232 | 0.1897 |
| (0.6, 0.8] | 1560 | 0.6109 |
| (0.8, 1.0] | 227 | 0.8678 |


## Kidney Condition (Next Visit) (`kidney`)

- Rows with non-null **GPT** probability for this task: **2492**
- **Gemma:** not included in this run (no Gemma source or no usable probabilities).

### Extreme probabilities

| Model | Count(0) | Count(1) | Total | % Extreme |
| --- | --- | --- | --- | --- |
| GPT-4o-mini | 1128 | 86 | 2492 | 48.72% |

### Probability bin analysis

#### GPT-4o-mini

| Probability Bin | Count | Avg True Label |
| --- | --- | --- |
| [0.0, 0.2] | 1366 | 0.2423 |
| (0.2, 0.4] | 1 | 0.0000 |
| (0.4, 0.6] | 67 | 0.3134 |
| (0.6, 0.8] | 540 | 0.6241 |
| (0.8, 1.0] | 518 | 0.7703 |


## Stroke (Next Visit) (`stroke`)

- Rows with non-null **GPT** probability for this task: **2492**
- **Gemma:** not included in this run (no Gemma source or no usable probabilities).

### Extreme probabilities

| Model | Count(0) | Count(1) | Total | % Extreme |
| --- | --- | --- | --- | --- |
| GPT-4o-mini | 1252 | 6 | 2492 | 50.48% |

### Probability bin analysis

#### GPT-4o-mini

| Probability Bin | Count | Avg True Label |
| --- | --- | --- |
| [0.0, 0.2] | 1471 | 0.0449 |
| (0.2, 0.4] | 2 | 0.0000 |
| (0.4, 0.6] | 331 | 0.0755 |
| (0.6, 0.8] | 664 | 0.0753 |
| (0.8, 1.0] | 24 | 0.2500 |


## Notes

- Counts can differ between models when one side has parse failures or missing task probabilities.
- Extreme counts treat exactly 0.0 and 1.0 as boundary spikes.
