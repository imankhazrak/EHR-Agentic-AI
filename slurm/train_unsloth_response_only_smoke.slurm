#!/bin/bash
#SBATCH --job-name=unsloth_ro_smoke
#SBATCH --account=pcs0229
#SBATCH --partition=gpu-exp
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --output=logs/slurm-unsloth-response-only-smoke-%j.out
#SBATCH --error=logs/slurm-unsloth-response-only-smoke-%j.err

# Response-only loss smoke: 500 train rows, 50 eval rows, 500 steps (~8 epochs on subset).
# Does not overwrite prior outputs (timestamped run dir).
# After success, set RUN_DIR from log / train_manifest.json for test job.

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

RUN_NAME="response_only_smoke"

echo "=== Unsloth response-only multitask SMOKE TRAIN ==="
echo "Job ID: ${SLURM_JOB_ID:-local}  RUN_NAME=${RUN_NAME}  Host: $(hostname)  Start: $(date)"
which python
python -c "import sys; print(sys.executable)"
nvidia-smi -L || true

python "$PROJECT_DIR/scripts/train_unsloth_router.py" \
  --train-jsonl "$PROJECT_DIR/dataset_for_unsloth/train.jsonl" \
  --eval-jsonl "$PROJECT_DIR/dataset_for_unsloth/test.jsonl" \
  --run-name "$RUN_NAME" \
  --train-max-samples 500 \
  --eval-max-samples 50 \
  --max-steps 500 \
  --save-steps 100 \
  --max-seq-length 2048 \
  --debug-label-mask-check 5

# Print newest matching run dir for follow-up test job
LATEST=$(ls -td "$PROJECT_DIR"/outputs/unsloth_mt_lora_${RUN_NAME}_* 2>/dev/null | head -1 || true)
if [[ -n "${LATEST}" && -f "${LATEST}/train_manifest.json" ]]; then
  echo "OUT_DIR=${LATEST}"
  echo "MODEL_LORA=${LATEST}/final_lora"
  echo "Submit test job with: RUN_DIR=${LATEST} sbatch slurm/test_unsloth_response_only_smoke.slurm"
else
  echo "WARNING: could not auto-detect OUT_DIR under outputs/unsloth_mt_lora_${RUN_NAME}_*"
fi

echo "End: $(date)"
