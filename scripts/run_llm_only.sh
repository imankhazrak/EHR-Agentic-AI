#!/usr/bin/env bash
# Run only LLM prompt experiments + summary table (skip preprocessing and ML).
# Requires existing data/processed/mimiciii/train.csv and test.csv from a prior run.
#
# Usage (from repo root, venv active):
#   bash scripts/run_llm_only.sh
#
# Default config now targets local Gemma (`llm.provider=local_gemma`).
# To switch Gemma variants without editing YAML:
#   export LLM_MODEL_NAME="google/gemma-4-e2b-it"
#
# If you already finished zero-shot and only want the remaining modes:
#   START_FROM=zero_shot_plus bash scripts/run_llm_only.sh
#
# Valid START_FROM values: zero_shot (default), zero_shot_plus, few_shot, coagent
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CFG=(--config configs/default.yaml)
STAGE="${STAGE:-full}"
if [[ "$STAGE" == "smoke" ]]; then
  CFG+=(--overrides configs/debug_small.yaml)
fi

START_FROM="${START_FROM:-zero_shot}"

run_zero_shot() {
  echo "python -m src.scripts.run_zero_shot ${CFG[*]}"
  python -m src.scripts.run_zero_shot "${CFG[@]}"
}
run_zero_shot_plus() {
  echo "python -m src.scripts.run_zero_shot_plus ${CFG[*]}"
  python -m src.scripts.run_zero_shot_plus "${CFG[@]}"
}
run_few_shot() {
  echo "python -m src.scripts.run_few_shot ${CFG[@]}"
  python -m src.scripts.run_few_shot "${CFG[@]}"
}
run_coagent() {
  echo "python -m src.scripts.run_coagent ${CFG[*]}"
  python -m src.scripts.run_coagent "${CFG[@]}"
}

echo "LLM-only pipeline  ROOT=$ROOT  START_FROM=$START_FROM  STAGE=$STAGE"

case "$START_FROM" in
  zero_shot)
    run_zero_shot; run_zero_shot_plus; run_few_shot; run_coagent
    ;;
  zero_shot_plus)
    run_zero_shot_plus; run_few_shot; run_coagent
    ;;
  few_shot)
    run_few_shot; run_coagent
    ;;
  coagent)
    run_coagent
    ;;
  *)
    echo "ERROR: START_FROM must be zero_shot, zero_shot_plus, few_shot, or coagent" >&2
    exit 1
    ;;
esac

python -c "from src.evaluation.summarize_results import collect_results; collect_results('data/outputs/mimiciii')"
echo "Done. See data/outputs/mimiciii/summary_table.csv"
