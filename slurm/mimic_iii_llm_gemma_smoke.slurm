#!/bin/bash
#SBATCH --job-name=mimic3_llm_gemma_smoke
#SBATCH --account=pcs0229
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --output=logs/slurm-llm-gemma-smoke-%j.out
#SBATCH --error=logs/slurm-llm-gemma-smoke-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

###############################################################################
# End-to-end LLM smoke test on LOCAL Gemma 4 (google/gemma-4-e4b-it snapshot).
# Runs: python -m src.scripts.smoke_llm_prompt_pipeline
#   --config configs/default.yaml --overrides configs/debug_small.yaml
#
# Outputs: data/outputs/mimiciii_llm_smoke/ (see configs/debug_small.yaml)
# Templates: prompts_v2/ (see configs/default.yaml llm.prompt_template_dir)
#
# Requires:
#   - data/processed/mimiciii/train.csv and test.csv
#   - conda env gemma4-smoke (torch + transformers + CUDA)
#   - HF_TOKEN in .env if the snapshot must be pulled (prefer pre-downloaded weights)
#
# Usage:
#   mkdir -p logs
#   sbatch slurm/mimic_iii_llm_gemma_smoke.slurm
#
# With local snapshot path (recommended):
#   sbatch --export=ALL,LOCAL_GEMMA_PATH=$PWD/models/hf_snapshots/google--gemma-4-e4b-it \
#     slurm/mimic_iii_llm_gemma_smoke.slurm
###############################################################################

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

python -m pip install -q \
  "openai>=1.0" "tenacity>=8.2" "pandas>=2.0" "pyyaml>=6.0" "python-dotenv>=1.0" \
  "jinja2>=3.1" "tqdm>=4.66" "scikit-learn>=1.3" "numpy>=1.24" "protobuf" "sentencepiece" \
  "anthropic>=0.18" || true

for f in "$PROJECT_DIR/data/processed/mimiciii/train.csv" "$PROJECT_DIR/data/processed/mimiciii/test.csv"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing $f — run preprocessing first." >&2
    exit 1
  fi
done

echo "Job ID: ${SLURM_JOB_ID:-local}  Start: $(date)"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())" || true

if [[ -n "${LOCAL_GEMMA_PATH:-}" ]]; then
  export LLM_MODEL_NAME="${LOCAL_GEMMA_PATH}"
fi

echo "=== warmup_local_gemma (debug_small merge) ==="
python -m src.scripts.warmup_local_gemma --config configs/default.yaml --overrides configs/debug_small.yaml

echo "=== smoke_llm_prompt_pipeline (Gemma 4, prompts_v2) ==="
python -m src.scripts.smoke_llm_prompt_pipeline --config configs/default.yaml --overrides configs/debug_small.yaml

echo "End: $(date)"
OUT="$(python -m src.scripts.print_output_dir --config configs/default.yaml --overrides configs/debug_small.yaml)"
echo "Smoke outputs + summary: ${PROJECT_DIR}/${OUT}/"
echo "Summary CSV: ${PROJECT_DIR}/${OUT}/summary_table.csv"
