# shellcheck shell=bash
# Sourced by SLURM scripts after PROJECT_DIR is set. Loads .env and requires a provider key.

slurm_load_and_require_llm_env() {
  local d="${PROJECT_DIR:-}"
  if [[ -f "$d/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$d/.env"
    set +a
  fi
  if [[ -n "${EHR_LLM_SKIP_API_KEY_CHECK:-}" ]]; then
    return 0
  fi
  if [[ -n "${OPENAI_BASE_URL:-}" ]] || [[ -n "${OPENAI_API_KEY:-}" ]] || [[ -n "${ANTHROPIC_API_KEY:-}" ]] || [[ -n "${AZURE_OPENAI_API_KEY:-}" ]]; then
    return 0
  fi
  echo "ERROR: No LLM API key found. Create $d/.env with OPENAI_API_KEY=... (or Anthropic/Azure vars)." >&2
  echo "Never commit .env. Do not paste API keys into chat or version control." >&2
  return 1
}
