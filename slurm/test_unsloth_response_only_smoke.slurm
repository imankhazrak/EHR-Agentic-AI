#!/bin/bash
#SBATCH --job-name=unsloth_ro_test
#SBATCH --account=pcs0229
#SBATCH --partition=gpu-exp
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --output=logs/slurm-unsloth-response-only-smoke-test-%j.out
#SBATCH --error=logs/slurm-unsloth-response-only-smoke-test-%j.err

# Inference (50 test rows) + strict scoring for response-only smoke LoRA.
# Required: RUN_DIR must point at a completed train run, e.g.:
#   RUN_DIR=outputs/unsloth_mt_lora_response_only_smoke_20260520_123456 sbatch slurm/test_unsloth_response_only_smoke.slurm
# Score only (skip GPU inference if predictions JSONL already exists):
#   SCORE_ONLY=1 RUN_DIR=outputs/... sbatch slurm/test_unsloth_response_only_smoke.slurm

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"
mkdir -p logs

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "ERROR: set RUN_DIR to the train output directory (see train job log OUT_DIR=...)."
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
# unsloth_env often lacks sklearn; install from repo requirements for scoring.
pip install -q "scikit-learn>=1.3"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== Unsloth response-only SMOKE TEST + SCORE ==="
echo "Job ID: ${SLURM_JOB_ID:-local}  RUN_DIR=${RUN_DIR}  Start: $(date)"
which python
nvidia-smi -L || true

if [[ ! -d "$MODEL_LORA" ]]; then
  echo "ERROR: MODEL_LORA not found: $MODEL_LORA"
  exit 1
fi

if [[ "${SCORE_ONLY:-0}" != "1" ]]; then
  python "$PROJECT_DIR/scripts/test_unsloth_router.py" \
    --model-path "$MODEL_LORA" \
    --test-jsonl "$PROJECT_DIR/dataset_for_unsloth/test.jsonl" \
    --output-jsonl "$OUT_JSONL" \
    --max-samples 50 \
    --max-seq-length 2048 \
    --summary-json "$RUN_DIR/test_predictions_smoke50.strict_summary.json"
else
  echo "SCORE_ONLY=1: skipping inference; using existing ${OUT_JSONL}"
  if [[ ! -f "$OUT_JSONL" ]]; then
    echo "ERROR: SCORE_ONLY=1 but predictions file missing: $OUT_JSONL"
    exit 1
  fi
fi

python "$PROJECT_DIR/scripts/score_unsloth_multitask_jsonl.py" \
  "$OUT_JSONL" \
  --parse-mode strict \
  --markdown-out "$METRICS_MD"

echo "Predictions: $OUT_JSONL"
echo "Strict metrics: $METRICS_MD"
echo "Summary JSON: $RUN_DIR/test_predictions_smoke50.strict_summary.json"
echo "End: $(date)"
