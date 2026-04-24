# Overconfidence Report

## Section 1 — Distribution Overview
Gemma and GPT-4o-mini probability histograms summarize the spread of predicted probabilities over the full dataset.
Both distributions can be inspected for concentration near boundaries and overall shape across the probability range.

## Section 2 — Extreme Probabilities
| Model | Count(0) | Count(1) | Total | % Extreme |
| --- | --- | --- | --- | --- |
| Gemma | 1391 | 1 | 2492 | 55.86% |
| GPT-4o-mini | 0 | 0 | 2492 | 0.00% |

## Section 3 — Probability Bin Analysis

### Gemma
| Probability Bin | Count | Avg True Label |
| --- | --- | --- |
| [0.0, 0.2] | 1452 | 0.0909 |
| (0.2, 0.4] | 10 | 0.5000 |
| (0.4, 0.6] | 0 | NA |
| (0.6, 0.8] | 195 | 0.2667 |
| (0.8, 1.0] | 835 | 0.5916 |

### GPT-4o-mini
| Probability Bin | Count | Avg True Label |
| --- | --- | --- |
| [0.0, 0.2] | 1504 | 0.1011 |
| (0.2, 0.4] | 0 | NA |
| (0.4, 0.6] | 0 | NA |
| (0.6, 0.8] | 560 | 0.4500 |
| (0.8, 1.0] | 428 | 0.6519 |

## Section 4 — Notes (IMPORTANT)
- A portion of predictions fall at extreme probability values (0 or 1).
- The empirical positive rate varies across probability bins.
- This report is descriptive only and does not apply calibration, threshold changes, or model retraining.
