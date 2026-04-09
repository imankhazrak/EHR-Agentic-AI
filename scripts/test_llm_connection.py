#!/usr/bin/env python3
"""One-shot check that OpenAI Chat Completions work with the configured model.

Run from repo root with venv active:
  python scripts/test_llm_connection.py
  python scripts/test_llm_connection.py --overrides configs/debug_small.yaml

Exit code 0 on success, 1 on failure. Does not print secrets.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config_utils import load_config
from src.llm.api_clients import LLMClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify one real LLM chat completion (no mocks).")
    parser.add_argument("--config", default="configs/default.yaml", help="Base YAML config path.")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Optional override YAML files merged after the base config.",
    )
    args = parser.parse_args()

    os.chdir(_ROOT)
    ov = args.overrides or None
    cfg = load_config(args.config, ov)
    llm = cfg["llm"]
    provider = llm.get("provider", "openai")
    model = llm.get("model", "")

    base_url = (llm.get("base_url") or "").strip()
    print(f"provider={provider}  model={model}  base_url={base_url or '(default api.openai.com)'}")
    if base_url:
        print("Custom base_url: ensure that endpoint is reachable and accepts the OpenAI chat schema.")

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key and base_url:
            key = "not-used"
        print(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}  effective_key_for_client: {bool(key)}")
        if not key:
            print("Fix: set OPENAI_API_KEY in .env at repo root.")
            return 1
        if not base_url and key.startswith("sk-proj-"):
            print(
                "Note: sk-proj-… keys are project-scoped. If you see 403 model access errors, open "
                "https://platform.openai.com/ → API keys → the project for this key → enable billing "
                "and GPT models, OR create a new key under a project that already has chat access."
            )

    try:
        import openai as openai_sdk

        _key = os.getenv("OPENAI_API_KEY") or ("not-used" if base_url else "")
        _kw: dict = {"api_key": _key}
        if base_url:
            _kw["base_url"] = base_url
        c = openai_sdk.OpenAI(**_kw)
        listed = [m.id for m in c.models.list().data]
        chat_like = [i for i in listed if any(x in i for x in ("gpt-", "o1", "o3", "o4"))]
        print(f"models.list() count={len(listed)}  chat-like ids={len(chat_like)}")
        if not base_url and not chat_like and listed:
            print(
                "Warning: no chat model ids in models.list(). Your OpenAI project may only "
                "have non-chat products enabled. Enable Chat Completions / a GPT model for this "
                "project in https://platform.openai.com/ (API keys → project → limits / model access)."
            )
    except Exception as e:
        print(f"models.list() skipped: {e}")

    try:
        client = LLMClient(llm)
        r = client.complete(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            use_cache=False,
        )
        text = (r.text or "").strip()
        print(f"chat.completions test reply: {text!r}")
        if not text:
            print("Empty reply — check logs above for API errors.")
            return 1
    except Exception as e:
        err = str(e)
        print(f"FAILED: {type(e).__name__}: {err[:500]}")
        if base_url:
            print(
                "\nCustom endpoint troubleshooting:\n"
                "  • Confirm OPENAI_BASE_URL and that the server implements OpenAI-compatible /v1/chat/completions.\n"
            )
        elif "does not have access to model" in err or "model_not_found" in err:
            print(
                "\nThis key’s OpenAI project does not allow this model. Options:\n"
                "  • Platform → API keys: enable GPT access + billing for that project.\n"
                "  • Set OPENAI_MODEL or LLM_MODEL in .env to a model your project can use.\n"
            )
        return 1

    print("OK — LLM calls should work for the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
