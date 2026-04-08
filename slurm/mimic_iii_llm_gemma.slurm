#!/bin/bash
#SBATCH --job-name=mimic3_llm_gemma
#SBATCH --account=pcs0229
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --output=logs/slurm-llm-gemma-%j.out
#SBATCH --error=logs/slurm-llm-gemma-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

###############################################################################
# Full prompt-based LLM pipeline (local Gemma): zero-shot, zero-shot+, few-shot,
# EHR-CoAgent. Uses configs/default.yaml (llm.provider=local_gemma).
# Requests 2 GPUs so Transformers device_map="auto" can shard Gemma 4 across devices.
#
# Requires:
#   - data/processed/mimiciii/train.csv and test.csv
#   - conda env gemma4-smoke with V100-compatible torch (e.g. torch 2.4.1+cu118)
#     and transformers; see README / prior GPU smoke setup.
#   - Hugging Face: accept Gemma model license; set HF_TOKEN in .env for gated pulls.
#
# Recommended: download weights once (login node, HF_TOKEN in .env), then point the job at the folder:
#   export PYTHONPATH="$(pwd)"
#   python -m src.scripts.download_gemma_model --model-id google/gemma-4-e4b-it
#   sbatch --export=ALL,LOCAL_GEMMA_PATH=$PWD/models/hf_snapshots/google--gemma-4-e4b-it slurm/mimic_iii_llm_gemma.slurm
#
# Usage:
#   mkdir -p logs
#   sbatch slurm/mimic_iii_llm_gemma.slurm
#
# Resume after partial run:
#   sbatch --export=ALL,START_FROM=few_shot slurm/mimic_iii_llm_gemma.slurm
###############################################################################

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"

module purge
module load cuda/11.8.0 2>/dev/null || true

mkdir -p "$PROJECT_DIR/logs" logs

# Local Gemma: no OpenAI key required (still load .env for HF_TOKEN if present).
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

# Runtime deps for pipeline (torch/transformers already in env; avoid upgrading torch).
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
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true

# Optional: use a pre-downloaded snapshot (set LOCAL_GEMMA_PATH before sbatch).
if [[ -n "${LOCAL_GEMMA_PATH:-}" ]]; then
  export LLM_MODEL_NAME="${LOCAL_GEMMA_PATH}"
fi

# Fail fast if model cannot load (avoids hundreds of empty predictions).
python -m src.scripts.warmup_local_gemma --config configs/default.yaml

bash "$PROJECT_DIR/scripts/run_llm_only.sh"

echo "End: $(date)"
echo "Summary: ${PROJECT_DIR}/data/outputs/mimiciii_gemma/summary_table.csv"
