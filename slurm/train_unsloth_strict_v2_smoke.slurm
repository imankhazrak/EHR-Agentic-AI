#!/bin/bash
#SBATCH --job-name=unsloth_v2_smoke
#SBATCH --account=pcs0229
#SBATCH --partition=gpu-exp
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --output=logs/slurm-unsloth-strict-v2-smoke-train-%j.out
#SBATCH --error=logs/slurm-unsloth-strict-v2-smoke-train-%j.err

# Strict-v2 multitask JSONL + response-only loss smoke (500 train / 50 eval / 500 steps).

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"
mkdir -p logs

module purge
module load cuda/11.8.0 2>/dev/null || true

CONDA_ROOT="${CONDA_ROOT:-/users/PCS0229/imankhazrak/miniconda3}"
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate unsloth_env
pip install -q "scikit-learn>=1.3"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DATA_DIR="$PROJECT_DIR/dataset_for_unsloth_multitask_strict_v2"
RUN_NAME="strict_v2_smoke"

echo "=== Unsloth strict_v2 response-only SMOKE TRAIN ==="
echo "Job ID: ${SLURM_JOB_ID:-local}  DATA_DIR=${DATA_DIR}  Start: $(date)"
nvidia-smi -L || true

python "$PROJECT_DIR/scripts/train_unsloth_router.py" \
  --train-jsonl "$DATA_DIR/train.jsonl" \
  --eval-jsonl "$DATA_DIR/test.jsonl" \
  --run-name "$RUN_NAME" \
  --train-max-samples 500 \
  --eval-max-samples 50 \
  --max-steps 500 \
  --save-steps 100 \
  --max-seq-length 2048 \
  --debug-label-mask-check 5

LATEST=$(ls -td "$PROJECT_DIR"/outputs/unsloth_mt_lora_${RUN_NAME}_* 2>/dev/null | head -1 || true)
if [[ -n "${LATEST}" && -f "${LATEST}/train_manifest.json" ]]; then
  echo "OUT_DIR=${LATEST}"
  echo "RUN_DIR=${LATEST}"
  echo "MODEL_LORA=${LATEST}/final_lora"
  echo "Submit test: RUN_DIR=${LATEST} sbatch slurm/test_unsloth_strict_v2_smoke.slurm"
fi
echo "End: $(date)"
