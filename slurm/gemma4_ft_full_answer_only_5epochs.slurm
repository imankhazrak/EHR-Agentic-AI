#!/bin/bash
#SBATCH --job-name=gemma4_ft_ans_5ep
#SBATCH --account=pcs0229
#SBATCH --partition=gpu-exp
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192GB
#SBATCH --output=logs/slurm-gemma4-ft-answer-only-5epochs-%j.out
#SBATCH --error=logs/slurm-gemma4-ft-answer-only-5epochs-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu
# 5 epochs, adapter under models/gemma_finetuned_answer_only_5epochs/ (see configs/finetune_answer_only_5epochs.yaml).

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"

module purge
module load cuda/11.8.0 2>/dev/null || true

mkdir -p "$PROJECT_DIR/logs"

CONDA_ROOT="${CONDA_ROOT:-/users/PCS0229/imankhazrak/miniconda3}"
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate gemma4-smoke

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID:-local}  5 epochs -> models/gemma_finetuned_answer_only_5epochs  Start: $(date)"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())" || true

python -m src.scripts.run_finetune_gemma --config configs/default.yaml --overrides configs/finetune_answer_only_5epochs.yaml

echo "End: $(date)"
