#!/bin/bash
#SBATCH --job-name=sequence_models
#SBATCH --account=PCS0229
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sequence_%j.out
#SBATCH --error=logs/sequence_%j.err

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
nvidia-smi -L || true

if [[ ! -f "scripts/run_sequence_models.py" ]]; then
  echo "ERROR: Missing scripts/run_sequence_models.py in $PROJECT_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/run_sequence_models.py

echo "Completed sequence models at $(date '+%Y-%m-%d %H:%M:%S %Z')"
