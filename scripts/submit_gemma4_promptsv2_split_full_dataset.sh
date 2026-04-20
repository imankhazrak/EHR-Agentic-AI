#!/usr/bin/env bash
# Submit four prompt-mode jobs plus a dependency-gated summarization job.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ZS_JOB_ID="$(sbatch slurm/mimic_iii_llm_gemma_promptsv2_full_dataset_zero_shot.slurm | awk '{print $4}')"
ZSP_JOB_ID="$(sbatch slurm/mimic_iii_llm_gemma_promptsv2_full_dataset_zero_shot_plus.slurm | awk '{print $4}')"
FS_JOB_ID="$(sbatch slurm/mimic_iii_llm_gemma_promptsv2_full_dataset_few_shot.slurm | awk '{print $4}')"
COAG_JOB_ID="$(sbatch slurm/mimic_iii_llm_gemma_promptsv2_full_dataset_coagent.slurm | awk '{print $4}')"

DEP="afterok:${ZS_JOB_ID}:${ZSP_JOB_ID}:${FS_JOB_ID}:${COAG_JOB_ID}"
SUM_JOB_ID="$(sbatch --dependency="${DEP}" slurm/mimic_iii_llm_gemma_promptsv2_full_dataset_summarize.slurm | awk '{print $4}')"

echo "Submitted jobs:"
echo "  zero_shot      : ${ZS_JOB_ID}"
echo "  zero_shot_plus : ${ZSP_JOB_ID}"
echo "  few_shot       : ${FS_JOB_ID}"
echo "  coagent        : ${COAG_JOB_ID}"
echo "  summarize      : ${SUM_JOB_ID} (dependency=${DEP})"
