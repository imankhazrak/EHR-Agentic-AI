#!/bin/bash
#SBATCH --job-name=m3_g4_p3_sum
#SBATCH --account=pcs0229
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-gemma4-p3-mt-summarize-%j.out
#SBATCH --error=logs/slurm-gemma4-p3-mt-summarize-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ikhazra@bgsu.edu
#
# Rebuild summary_table.csv from metrics JSONs under the Gemma prompt_v3 multitask output dir.
# Run after the four GPU jobs (zero_shot, zero_shot_plus, few_shot, coagent) have written
# llm_*_metrics.json files. No GPU required.
#
# Optional dependency when submitting after the four jobs:
#   sbatch --dependency=afterok:JOBID1:JOBID2:JOBID3:JOBID4 slurm/gemma4_promptv3_multitask_summarize.slurm

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/PCS0229/imankhazrak/EHR-Agentic-AI}"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs" logs

CONDA_ROOT="${CONDA_ROOT:-/users/PCS0229/imankhazrak/miniconda3}"
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate gemma4-smoke

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OVR="configs/experiments/gemma4_promptv3_multitask_test.yaml"

echo "Job ID: ${SLURM_JOB_ID:-local} Start: $(date)"
echo "Summarize outputs for OVERRIDES=${OVR}"

OUTPUT_DIR="$(python -m src.scripts.print_output_dir --config configs/default.yaml --overrides "$OVR")"
ABS_OUT="${PROJECT_DIR}/${OUTPUT_DIR}"
echo "Output dir: ${ABS_OUT}"

python -c "from src.evaluation.summarize_results import collect_results; import sys; collect_results(sys.argv[1])" "$ABS_OUT"

echo "Done: ${ABS_OUT}/summary_table.csv"
echo "End: $(date)"
