#!/bin/bash
#SBATCH --job-name=unsloth_v2_test
#SBATCH --account=pcs0229
#SBATCH --partition=gpu-exp
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --output=logs/slurm-unsloth-strict-v2-smoke-test-%j.out
#SBATCH --error=logs/slurm-unsloth-strict-v2-smoke-test-%j.err

# Strict-v2 smoke: 50 test rows + strict parse coverage report.
# RUN_DIR=outputs/unsloth_mt_lora_strict_v2_smoke_<timestamp> sbatch ...

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"
mkdir -p logs

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "ERROR: set RUN_DIR to the train output directory."
  exit 1
fi

RUN_DIR="${RUN_DIR#"$PROJECT_DIR"/}"
RUN_DIR="$PROJECT_DIR/$RUN_DIR"
MODEL_LORA="${MODEL_LORA:-$RUN_DIR/final_lora}"
OUT_JSONL="$RUN_DIR/test_predictions_smoke50.jsonl"
METRICS_MD="$RUN_DIR/test_predictions_smoke50_metrics_strict.md"

module purge
module load cuda/11.8.0 2>/dev/null || true

CONDA_ROOT="${CONDA_ROOT:-/users/PCS0229/imankhazrak/miniconda3}"
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate unsloth_env
pip install -q "scikit-learn>=1.3"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== Unsloth strict_v2 SMOKE TEST + SCORE ==="
echo "RUN_DIR=${RUN_DIR}  Start: $(date)"

if [[ "${SCORE_ONLY:-0}" != "1" ]]; then
  python "$PROJECT_DIR/scripts/test_unsloth_router.py" \
    --model-path "$MODEL_LORA" \
    --test-jsonl "$PROJECT_DIR/dataset_for_unsloth_multitask_strict_v2/test.jsonl" \
    --output-jsonl "$OUT_JSONL" \
    --max-samples 50 \
    --max-seq-length 2048 \
    --summary-json "$RUN_DIR/test_predictions_smoke50.strict_summary.json"
else
  echo "SCORE_ONLY=1: using ${OUT_JSONL}"
fi

python "$PROJECT_DIR/scripts/score_unsloth_multitask_jsonl.py" \
  "$OUT_JSONL" \
  --parse-mode strict \
  --markdown-out "$METRICS_MD"

echo "End: $(date)"
