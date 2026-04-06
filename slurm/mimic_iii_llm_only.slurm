#!/bin/bash
#SBATCH --job-name=mimic3_llm
#SBATCH --account=pcs0229
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/slurm-llm-%j.out
#SBATCH --error=logs/slurm-llm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

###############################################################################
# LLM-only: skip preprocessing and ML; requires processed train/test CSVs.
# .env: OPENAI_API_KEY (loaded via scripts/slurm_llm_env.sh).
# Prefer: sbatch slurm/mimic_iii_llm_smoke.slurm
#
# Usage:
#   mkdir -p logs
#   sbatch slurm/mimic_iii_llm_only.slurm
#
# Skip completed modes (e.g. zero-shot already done):
#   sbatch --export=ALL,START_FROM=zero_shot_plus slurm/mimic_iii_llm_only.slurm
###############################################################################

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"

module purge
module load python/3.10
source "$PROJECT_DIR/.venv/bin/activate"
mkdir -p "$PROJECT_DIR/logs" logs

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# shellcheck source=/dev/null
source "$PROJECT_DIR/scripts/slurm_llm_env.sh"
slurm_load_and_require_llm_env || exit 1

for f in "$PROJECT_DIR/data/processed/mimiciii/train.csv" "$PROJECT_DIR/data/processed/mimiciii/test.csv"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing $f — run preprocessing first." >&2
    exit 1
  fi
done

bash "$PROJECT_DIR/scripts/run_llm_only.sh"
