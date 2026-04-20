#!/bin/bash
#SBATCH --job-name=m3_g4_p2_fullds_sum
#SBATCH --account=pcs0229
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/slurm-llm-gemma4-promptv2-fullds-summarize-%j.out
#SBATCH --error=logs/slurm-llm-gemma4-promptv2-fullds-summarize-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"

module purge
mkdir -p "$PROJECT_DIR/logs" logs

CONDA_ROOT="${CONDA_ROOT:-/users/PCS0229/imankhazrak/miniconda3}"
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate gemma4-smoke

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export LLM_OVERRIDES="configs/experiments/gemma4_promptv2_full_dataset.yaml"

echo "Job ID: ${SLURM_JOB_ID:-local}  Start: $(date)"
echo "This job must be submitted with --dependency=afterok:<zs>:<zsp>:<fs>:<coagent>"
echo "Override: $LLM_OVERRIDES"

OUTPUT_DIR="$(python -m src.scripts.print_output_dir --config configs/default.yaml --overrides "${LLM_OVERRIDES}")"
python -c "from src.evaluation.summarize_results import collect_results; import sys; collect_results(sys.argv[1])" "${OUTPUT_DIR}"

echo "End: $(date)"
echo "Summary CSV: ${PROJECT_DIR}/${OUTPUT_DIR}/summary_table.csv"
