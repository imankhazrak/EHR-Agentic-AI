#!/usr/bin/env bash
# Run MIMIC-III reproduction steps in order. Usage from repo root:
#   STAGE=smoke bash scripts/run_mimic_pipeline.sh
#   STAGE=full  bash scripts/run_mimic_pipeline.sh
set -euo pipefail

STAGE="${STAGE:-full}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CFG=(--config configs/default.yaml)
if [[ "$STAGE" == "smoke" ]]; then
  CFG+=(--overrides configs/debug_small.yaml)
fi

echo "STAGE=$STAGE  ROOT=$ROOT"
echo "python -m src.scripts.run_preprocessing ${CFG[*]}"
python -m src.scripts.run_preprocessing "${CFG[@]}"

echo "python -m src.scripts.run_ml_baselines ${CFG[*]}"
python -m src.scripts.run_ml_baselines "${CFG[@]}"

for m in run_zero_shot run_zero_shot_plus run_few_shot run_coagent; do
  echo "python -m src.scripts.$m ${CFG[*]}"
  python -m "src.scripts.$m" "${CFG[@]}"
done

python -c "from src.evaluation.summarize_results import collect_results; collect_results('data/outputs/mimiciii')"
echo "Done. See data/outputs/mimiciii/summary_table.csv"
