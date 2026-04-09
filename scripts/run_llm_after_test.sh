#!/usr/bin/env bash
# Run scripts/test_llm_connection.py first; only then run LLM pipeline.
# Usage (from repo root, venv active):
#   bash scripts/run_llm_after_test.sh smoke    # debug_small overrides, ~12 test rows
#   bash scripts/run_llm_after_test.sh full     # production LLM-only run
#
# Optional smoke wrapper: bash scripts/smoke_llm_real.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: bash scripts/run_llm_after_test.sh smoke|full" >&2
  exit 1
fi

python scripts/test_llm_connection.py

if [[ "$MODE" == "smoke" ]]; then
  STAGE=smoke bash "$ROOT/scripts/run_llm_only.sh"
else
  STAGE=full bash "$ROOT/scripts/run_llm_only.sh"
fi
