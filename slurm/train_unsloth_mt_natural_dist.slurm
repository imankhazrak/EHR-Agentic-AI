#!/bin/bash
#SBATCH --job-name=unsloth_mt_natural
#SBATCH --account=pcs0229
#SBATCH --partition=gpu-exp
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --output=logs/slurm-unsloth_mt_natural-%j.out
#SBATCH --error=logs/slurm-unsloth_mt_natural-%j.err

# Full multitask LoRA: natural class distribution, train on train.jsonl,
# eval loss on held-out test.jsonl. Checkpoints + final_lora live under a
# timestamped outputs/unsloth_mt_lora_natural_<jobid>_<UTC>/ folder.

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

RUN_NAME="natural_dist_job${SLURM_JOB_ID:-local}"

echo "=== Unsloth multitask NATURAL distribution train + test eval ==="
echo "Job ID: ${SLURM_JOB_ID:-local}  RUN_NAME=${RUN_NAME}  Host: $(hostname)  Start: $(date)"
which python
python -c "import sys; print(sys.executable)"
nvidia-smi -L || true

python "$PROJECT_DIR/scripts/train_unsloth_router.py" \
  --run-name "$RUN_NAME" \
  --train-jsonl "$PROJECT_DIR/dataset_for_unsloth/train.jsonl" \
  --eval-jsonl "$PROJECT_DIR/dataset_for_unsloth/test.jsonl" \
  --max-steps 1500 \
  --save-steps 50 \
  --logging-steps 5 \
  --gradient-accumulation-steps 8 \
  --per-device-train-batch-size 1 \
  --max-seq-length 2048

echo "Training finished. Artifacts under outputs/unsloth_mt_lora_${RUN_NAME}_*"
echo "End: $(date)"
