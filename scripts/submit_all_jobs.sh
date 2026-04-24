#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Urban-Air-Mobility}"
SLURM_DIR="${SLURM_DIR:-$HOME/EHR-Agentic-AI/slurm}"

cd "$PROJECT_DIR"
mkdir -p logs

submit_and_get_id() {
  local script_path="$1"
  local output
  local job_id

  output="$(sbatch "$script_path")"
  job_id="$(awk '{print $4}' <<<"$output")"
  if [[ -z "$job_id" ]]; then
    echo "ERROR: Failed to parse job ID from: $output" >&2
    exit 1
  fi
  echo "$job_id"
}

echo "Submitting baseline, ablation, uncertainty, and sequence jobs..."
BASELINE_ID="$(submit_and_get_id "$SLURM_DIR/run_baseline.slurm")"
ABLATION_ID="$(submit_and_get_id "$SLURM_DIR/run_ablation.slurm")"
UNCERTAINTY_ID="$(submit_and_get_id "$SLURM_DIR/run_uncertainty.slurm")"
SEQUENCE_ID="$(submit_and_get_id "$SLURM_DIR/run_sequence.slurm")"

echo "baseline:    $BASELINE_ID"
echo "ablation:    $ABLATION_ID"
echo "uncertainty: $UNCERTAINTY_ID"
echo "sequence:    $SEQUENCE_ID"

DEPENDENCY="afterok:${BASELINE_ID}:${ABLATION_ID}:${UNCERTAINTY_ID}:${SEQUENCE_ID}"
SUMMARIZE_OUTPUT="$(sbatch --dependency="$DEPENDENCY" "$SLURM_DIR/run_summarize.slurm")"
SUMMARIZE_ID="$(awk '{print $4}' <<<"$SUMMARIZE_OUTPUT")"

if [[ -z "$SUMMARIZE_ID" ]]; then
  echo "ERROR: Failed to parse summarize job ID from: $SUMMARIZE_OUTPUT" >&2
  exit 1
fi

echo "summarize:   $SUMMARIZE_ID (dependency=$DEPENDENCY)"
echo "All jobs submitted."
