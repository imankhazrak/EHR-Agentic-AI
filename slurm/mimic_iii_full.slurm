#!/bin/bash
#SBATCH --job-name=mimic3_full
#SBATCH --account=pcs0229
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --output=logs/slurm-full-%j.out
#SBATCH --error=logs/slurm-full-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

###############################################################################
# Full MIMIC-III pipeline: preprocessing, ML baselines, all LLM modes, summary.
# Requires a real API key in $PROJECT_DIR/.env (OpenAI, Anthropic, or Azure).
#
# Usage:
#   mkdir -p logs
#   sbatch slurm/mimic_iii_full.slurm
###############################################################################

set -euo pipefail

echo "=========================================="
echo "MIMIC-III FULL RUN — STAGE=full"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start: $(date)"
echo ""

module purge
module load python/3.10

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"

if [[ -d "$PROJECT_DIR/.venv" ]]; then
  source "$PROJECT_DIR/.venv/bin/activate"
elif [[ -n "${VENV_PATH:-}" && -d "$VENV_PATH" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "ERROR: activate a venv (set VENV_PATH or create $PROJECT_DIR/.venv)" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs" logs

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export STAGE=full

# shellcheck source=/dev/null
source "$PROJECT_DIR/scripts/slurm_llm_env.sh"
slurm_load_and_require_llm_env || exit 1

bash "$PROJECT_DIR/scripts/run_mimic_pipeline.sh"

echo ""
echo "End: $(date)"
echo "Outputs: data/outputs/mimiciii/"
