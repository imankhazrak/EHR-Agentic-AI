#!/bin/bash
#SBATCH --job-name=m3_gemma4_p3_mt_test
#SBATCH --account=pcs0229
#SBATCH --partition=gpu
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/slurm-gemma4-promptv3-multitask-test-%j.out
#SBATCH --error=logs/slurm-gemma4-promptv3-multitask-test-%j.err
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

PROC_DIR="$PROJECT_DIR/data/processed/mimiciii_multitask"
for f in "$PROC_DIR/train.csv" "$PROC_DIR/test.csv"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing $f" >&2
    exit 1
  fi
done

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
echo "Git branch: ${BRANCH}"
if [[ "${BRANCH}" != "multitask" ]]; then
  echo "WARNING: expected branch 'multitask' but got '${BRANCH}'"
fi

export STAGE=full
export START_FROM=zero_shot
export LLM_OVERRIDES="configs/experiments/gemma4_promptv3_multitask_test.yaml"

echo "Job ID: ${SLURM_JOB_ID:-local} Start: $(date)"
echo "LLM_OVERRIDES=${LLM_OVERRIDES}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())" || true

bash "$PROJECT_DIR/scripts/run_llm_only.sh"

echo "End: $(date)"
OUT="$(python -m src.scripts.print_output_dir --config configs/default.yaml --overrides "${LLM_OVERRIDES}")"
echo "Summary: ${PROJECT_DIR}/${OUT}/summary_table.csv"
