#!/bin/bash
#SBATCH --job-name=m3_g4_p2_fullds_coag
#SBATCH --account=pcs0229
#SBATCH --partition=gpu
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/slurm-llm-gemma4-promptv2-fullds-coagent-%j.out
#SBATCH --error=logs/slurm-llm-gemma4-promptv2-fullds-coagent-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"

module purge
module load cuda/11.8.0 2>/dev/null || true
mkdir -p "$PROJECT_DIR/logs" logs

CONDA_ROOT="${CONDA_ROOT:-/users/PCS0229/imankhazrak/miniconda3}"
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate gemma4-smoke

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export EHR_LLM_SKIP_API_KEY_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STAGE=full
export LLM_OVERRIDES="configs/experiments/gemma4_promptv2_full_dataset.yaml"

FULL_DATASET="$PROJECT_DIR/data/processed/mimiciii/full_dataset.csv"
PROC_DIR="$PROJECT_DIR/data/processed/mimiciii_full_dataset"
mkdir -p "$PROC_DIR"

if [[ ! -f "$FULL_DATASET" ]]; then
  echo "ERROR: Missing $FULL_DATASET" >&2
  exit 1
fi
if [[ ! -f "$PROC_DIR/train.csv" ]]; then
  echo "ERROR: Missing $PROC_DIR/train.csv (needed for coagent)." >&2
  exit 1
fi

cp "$FULL_DATASET" "$PROC_DIR/test.csv"

echo "Job ID: ${SLURM_JOB_ID:-local}  Start: $(date)"
echo "Mode: coagent"
echo "Override: $LLM_OVERRIDES"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())" || true

python -m src.scripts.run_coagent --config configs/default.yaml --overrides "${LLM_OVERRIDES}"

echo "End: $(date)"
