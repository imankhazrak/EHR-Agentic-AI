#!/bin/bash
#SBATCH --job-name=ablation_study
#SBATCH --account=PCS0229
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

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

if [[ ! -f "scripts/run_ablation_study.py" ]]; then
  echo "ERROR: Missing scripts/run_ablation_study.py in $PROJECT_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/run_ablation_study.py

echo "Completed ablation study at $(date '+%Y-%m-%d %H:%M:%S %Z')"
