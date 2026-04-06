#!/bin/bash
#SBATCH --job-name=mimic3_llm_smoke
#SBATCH --account=pcs0229
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm-llm-smoke-%j.out
#SBATCH --error=logs/slurm-llm-smoke-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

###############################################################################
# Real LLM smoke (no mocks): scripts/smoke_llm_real.sh — connection test + four modes, ~12 test rows.
#
#   mkdir -p logs
#   sbatch slurm/mimic_iii_llm_smoke.slurm
#
# Requires OPENAI_API_KEY in $PROJECT_DIR/.env (or cluster env).
#
# Full production LLM after this succeeds:
#   sbatch slurm/mimic_iii_llm_only.slurm
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

bash "$PROJECT_DIR/scripts/smoke_llm_real.sh" || {
  echo "ERROR: smoke_llm_real.sh failed — check OPENAI_API_KEY, billing, and model access; see logs." >&2
  exit 1
}
