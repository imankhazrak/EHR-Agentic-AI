#!/bin/bash
#SBATCH --job-name=mimiciii_fusion_holdout
#SBATCH --account=pcs0229
#SBATCH --partition=gpu
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/slurm-mimiciii-fusion-holdout-%j.out
#SBATCH --error=logs/slurm-mimiciii-fusion-holdout-%j.err

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

echo "Job ID: ${SLURM_JOB_ID:-local}  Start: $(date)"
nvidia-smi -L || true
python -c "import torch; print('cuda', torch.cuda.is_available())" || true

bash scripts/run_fusion_holdout_full.sh

echo "End: $(date)"
echo "Outputs: $PROJECT_DIR/data/outputs/mimiciii_fusion_holdout_run/"
