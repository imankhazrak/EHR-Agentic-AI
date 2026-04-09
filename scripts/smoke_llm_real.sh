#!/usr/bin/env bash
# Real LLM smoke (no mocks): connection test + four modes, ~12 test rows (configs/debug_small.yaml).
# Requires OPENAI_API_KEY in .env (or environment).
#
#   bash scripts/smoke_llm_real.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "=== smoke_llm_real.sh ==="

python scripts/test_llm_connection.py
STAGE=smoke bash "$ROOT/scripts/run_llm_only.sh"

echo "=== smoke_llm_real.sh done ==="
