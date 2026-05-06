#!/bin/bash
#SBATCH --job-name=m3_g4_p3_fs
#SBATCH --account=pcs0229
# v100-32g GRES is on gpu-exp (see `sinfo`), not on `gpu`.
#SBATCH --partition=gpu-exp
# Walltime: 2 days (Slurm D-HH:MM:SS)
#SBATCH --time=2-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100-32g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/slurm-gemma4-p3-mt-few-shot-%j.out
#SBATCH --error=logs/slurm-gemma4-p3-mt-few-shot-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu
#
# Gemma-4, prompts_v3 multitask — Few-Shot (N=6) only.

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
export PYTORCH_ALLOC_CONF=expandable_segments:True

OVR="configs/experiments/gemma4_promptv3_multitask_test.yaml"
PROC_DIR="$PROJECT_DIR/data/processed/mimiciii_multitask"
for f in "$PROC_DIR/train.csv" "$PROC_DIR/test.csv"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing $f" >&2
    exit 1
  fi
done

echo "Job ID: ${SLURM_JOB_ID:-local} Start: $(date)"
echo "Mode: few_shot  OVERRIDES=${OVR}"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true

python -m src.scripts.run_few_shot --config configs/default.yaml --overrides "$OVR"

echo "End: $(date)"
OUT="$(python -m src.scripts.print_output_dir --config configs/default.yaml --overrides "$OVR")"
echo "Outputs: ${PROJECT_DIR}/${OUT}"
