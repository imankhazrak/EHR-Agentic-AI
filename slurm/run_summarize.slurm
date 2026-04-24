#!/bin/bash
#SBATCH --job-name=summarize_results
#SBATCH --account=PCS0229
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/summarize_%j.out
#SBATCH --error=logs/summarize_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Urban-Air-Mobility}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# shellcheck source=/dev/null
source "$HOME/venvs/unsloth_env/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S %Z')"

if [[ ! -f "scripts/summarize_all_results.py" ]]; then
  echo "ERROR: Missing scripts/summarize_all_results.py in $PROJECT_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/summarize_all_results.py

echo "Completed summarization at $(date '+%Y-%m-%d %H:%M:%S %Z')"
