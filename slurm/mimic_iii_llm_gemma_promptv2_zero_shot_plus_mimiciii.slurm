#!/bin/bash
#SBATCH --job-name=m3_pv2_zsp
#SBATCH --account=pcs0229
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --output=logs/slurm-m3-pv2-zsp-mimiciii-%j.out
#SBATCH --error=logs/slurm-m3-pv2-zsp-mimiciii-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu
#
# Prompt V2 zero-shot+ on standard MIMIC test split (configs/default.yaml → mimiciii_llm_promptv2_gemma).
# Uses 2 GPUs so Gemma-4 can shard with device_map=auto (avoids 16GB single-GPU OOM from job 46836010).

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"

module purge
module load cuda/11.8.0 2>/dev/null || true
mkdir -p "$PROJECT_DIR/logs" logs

export EHR_LLM_SKIP_API_KEY_CHECK=1
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_DIR/.env"
  set +a
fi

CONDA_ROOT="${CONDA_ROOT:-/users/PCS0229/imankhazrak/miniconda3}"
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate gemma4-smoke

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

for f in "$PROJECT_DIR/data/processed/mimiciii/train.csv" "$PROJECT_DIR/data/processed/mimiciii/test.csv"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing $f — run preprocessing first." >&2
    exit 1
  fi
done

echo "Job ID: ${SLURM_JOB_ID:-local}  Mode: zero_shot_plus  Start: $(date)"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())" || true

if [[ -n "${LOCAL_GEMMA_PATH:-}" ]]; then
  export LLM_MODEL_NAME="${LOCAL_GEMMA_PATH}"
fi

python -m src.scripts.warmup_local_gemma --config configs/default.yaml
python -m src.scripts.run_zero_shot_plus --config configs/default.yaml

echo "End: $(date)"
OUT="$(python -m src.scripts.print_output_dir --config configs/default.yaml)"
echo "Outputs: ${PROJECT_DIR}/${OUT}/"
