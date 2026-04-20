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
#
# Preserve prior result trees: do not reuse their paths.outputs. Default config uses a
# dedicated folder; for ad-hoc runs you can also set:
#   export EHR_OUTPUTS_DIR="data/outputs/mimiciii_llm_$(date +%Y%m%d_%H%M)"
#   (optional) export EHR_LLM_CACHE_DIR="$EHR_OUTPUTS_DIR/llm_cache"
#
# Full run with extra YAML overrides (e.g. OpenAI + dedicated output folder):
#   export LLM_OVERRIDES="configs/experiments/openai_gpt4o_mini_promptv2_full.yaml"
#   STAGE=full bash scripts/run_llm_only.sh
#
# Smoke + optional extra merges (e.g. OpenAI on login node):
#   export LLM_EXTRA_OVERRIDES="configs/smoke_openai.yaml"
#   STAGE=smoke bash scripts/run_llm_only.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CFG=(--config configs/default.yaml)
STAGE="${STAGE:-full}"
if [[ "$STAGE" == "smoke" ]]; then
  OVR_LIST=(configs/debug_small.yaml)
  if [[ -n "${LLM_EXTRA_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA=(${LLM_EXTRA_OVERRIDES})
    OVR_LIST+=("${EXTRA[@]}")
  fi
  CFG+=(--overrides "${OVR_LIST[@]}")
elif [[ -n "${LLM_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  OVR_LIST=(${LLM_OVERRIDES})
  CFG+=(--overrides "${OVR_LIST[@]}")
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

OUTPUT_DIR="$(python -m src.scripts.print_output_dir "${CFG[@]}")"
python -c "from src.evaluation.summarize_results import collect_results; import sys; collect_results(sys.argv[1])" "${OUTPUT_DIR}"
echo "Done. See ${OUTPUT_DIR}/summary_table.csv"
