#!/bin/bash
#SBATCH --job-name=test_unsloth_router
#SBATCH --account=pcs0229
#SBATCH --partition=gpu-exp
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --output=logs/slurm-test-unsloth-router-%j.out
#SBATCH --error=logs/slurm-test-unsloth-router-%j.err

# Requires Conda env unsloth_env (see scripts/test_unsloth_router.py).

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

echo "Job ID: ${SLURM_JOB_ID:-local}  Host: $(hostname)  Start: $(date)"
which python
python -c "import sys; print(sys.executable)"
nvidia-smi -L || true

MODEL_LORA="${MODEL_LORA:-$PROJECT_DIR/outputs/unsloth_mt_lora_natural_dist_job47427323_20260513_021406/final_lora}"
OUT_JSONL="${OUT_JSONL:-$PROJECT_DIR/outputs/unsloth_mt_lora_natural_dist_job47427323_20260513_021406/test_predictions_2nd_try_fixed.jsonl}"

python "$PROJECT_DIR/scripts/test_unsloth_router.py" \
  --model-path "$MODEL_LORA" \
  --test-jsonl "$PROJECT_DIR/dataset_for_unsloth/test.jsonl" \
  --output-jsonl "$OUT_JSONL" \
  --max-seq-length 2048 \
  "$@"

echo "End: $(date)"
