#!/usr/bin/env bash
# Full held-out fusion: LLM logit scores (train, test, train OOF), build datasets, train all fusion models, summary table.
# Requires: processed MIMIC train/test, Gemma+LoRA adapter, GPU.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CFG="configs/default.yaml"
BASE_OVERRIDES=(configs/fusion_holdout_mimiciii.yaml)
INFOLD_OVERRIDES=(configs/fusion_holdout_mimiciii.yaml configs/fusion_holdout_infold.yaml)
OOF_OVERRIDES=(configs/fusion_holdout_mimiciii.yaml configs/fusion_holdout_oof.yaml)

echo "=== LLM scoring: train (infold), test ==="
python -m src.scripts.score_llm_logits_split --config "$CFG" --overrides "${BASE_OVERRIDES[@]}" --split train
python -m src.scripts.score_llm_logits_split --config "$CFG" --overrides "${BASE_OVERRIDES[@]}" --split test

echo "=== LLM scoring: train OOF (5-fold) ==="
NSPLIT="${FUSION_OOF_SPLITS:-5}"
python -m src.scripts.score_llm_logits_train_oof --config "$CFG" --overrides "${BASE_OVERRIDES[@]}" --n-splits "$NSPLIT"

echo "=== Fusion dataset + experiments: infold train LLM ==="
python -m src.scripts.build_fusion_dataset --config "$CFG" --overrides "${INFOLD_OVERRIDES[@]}"
python -m src.scripts.run_fusion_experiments --config "$CFG" --overrides "${INFOLD_OVERRIDES[@]}" --skip-llm-score --skip-build

echo "=== Fusion dataset + experiments: OOF train LLM ==="
python -m src.scripts.build_fusion_dataset --config "$CFG" --overrides "${OOF_OVERRIDES[@]}"
python -m src.scripts.run_fusion_experiments --config "$CFG" --overrides "${OOF_OVERRIDES[@]}" --skip-llm-score --skip-build

EXP_DIR="$PROJECT_DIR/data/outputs/mimiciii_fusion_holdout_run"
echo "=== Summary table ==="
python -m src.scripts.summarize_fusion_holdout \
  --experiment-dir "$EXP_DIR" \
  --ml-csv "$PROJECT_DIR/data/outputs/mimiciii_gpt4o_mini/ml_results_fully_supervised.csv" \
  --llm-csv "$PROJECT_DIR/data/outputs/mimiciii_gemma/llm_finetuned_test_results.csv"

echo "Done. Table: $EXP_DIR/fusion_final_comparison_table.csv"
