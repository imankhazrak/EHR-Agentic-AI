#!/bin/bash
#SBATCH --job-name=mimic3_smoke
#SBATCH --account=pcs0229
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --output=logs/slurm-smoke-%j.out
#SBATCH --error=logs/slurm-smoke-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

# Smoke test: configs/default.yaml + configs/debug_small.yaml (small data, limited LLM rows).
# Requires a real API key in $PROJECT_DIR/.env.
set -euo pipefail

echo "MIMIC-III smoke — $(date)"
module purge
module load python/3.10

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
mkdir -p "$PROJECT_DIR/logs" logs

export PYTHONPATH="$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export STAGE=smoke

# shellcheck source=/dev/null
source "$PROJECT_DIR/scripts/slurm_llm_env.sh"
slurm_load_and_require_llm_env || exit 1

bash "$PROJECT_DIR/scripts/run_mimic_pipeline.sh"
echo "Smoke done — $(date)"
