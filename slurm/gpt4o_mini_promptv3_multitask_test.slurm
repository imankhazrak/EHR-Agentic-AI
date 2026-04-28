#!/bin/bash
#SBATCH --job-name=m3_gpt4omini_p3_mt_test
#SBATCH --account=pcs0229
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/slurm-gpt4omini-promptv3-multitask-test-%j.out
#SBATCH --error=logs/slurm-gpt4omini-promptv3-multitask-test-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

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

PROC_DIR="$PROJECT_DIR/data/processed/mimiciii_multitask"
for f in "$PROC_DIR/train.csv" "$PROC_DIR/test.csv"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing $f" >&2
    exit 1
  fi
done

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
echo "Git branch: ${BRANCH}"
if [[ "${BRANCH}" != "multitask" ]]; then
  echo "WARNING: expected branch 'multitask' but got '${BRANCH}'"
fi

export STAGE=full
export START_FROM=zero_shot
export LLM_OVERRIDES="configs/experiments/gpt4o_mini_promptv3_multitask_test.yaml"

echo "Job ID: ${SLURM_JOB_ID:-local} Start: $(date)"
echo "LLM_OVERRIDES=${LLM_OVERRIDES}"

bash "$PROJECT_DIR/scripts/run_llm_only.sh"

echo "End: $(date)"
OUT="$(python -m src.scripts.print_output_dir --config configs/default.yaml --overrides "${LLM_OVERRIDES}")"
echo "Summary: ${PROJECT_DIR}/${OUT}/summary_table.csv"
