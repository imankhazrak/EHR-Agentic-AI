#!/bin/bash
#SBATCH --job-name=unsloth_mt_smoke
#SBATCH --account=pcs0229
#SBATCH --partition=gpu-exp
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --output=logs/slurm-unsloth_mt_smoke-%j.out
#SBATCH --error=logs/slurm-unsloth_mt_smoke-%j.err

# Very small run: unsloth_env, tiny train/eval slices, few steps.
# Submit before the full natural-distribution job.

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

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN_NAME="smoke_natural_${SLURM_JOB_ID:-local}"

echo "=== Unsloth multitask SMOKE ==="
echo "Job ID: ${SLURM_JOB_ID:-local}  RUN_NAME=${RUN_NAME}  Host: $(hostname)  Start: $(date)"
which python
python -c "import sys; print(sys.executable)"
nvidia-smi -L || true

python "$PROJECT_DIR/scripts/train_unsloth_router.py" \
  --run-name "$RUN_NAME" \
  --train-jsonl "$PROJECT_DIR/dataset_for_unsloth/train.jsonl" \
  --eval-jsonl "$PROJECT_DIR/dataset_for_unsloth/test.jsonl" \
  --train-max-samples 48 \
  --eval-max-samples 24 \
  --max-steps 5 \
  --save-steps 1 \
  --logging-steps 1 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 1024 \
  --save-total-limit 2

echo "Smoke OK. Latest run directory is under outputs/ (see train_manifest.json inside)."
echo "RUN_NAME was: ${RUN_NAME}"
echo "End: $(date)"
