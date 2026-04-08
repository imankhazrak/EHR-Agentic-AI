"""Load local Gemma once and run a tiny generation (verify GPU + weights before full pipeline).

Usage:
    export PYTHONPATH="$(pwd)"
    python -m src.scripts.warmup_local_gemma --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys

from src.llm.api_clients import LLMClient
from src.utils.config_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    llm = cfg.get("llm", {})
    if llm.get("provider") != "local_gemma":
        print("ERROR: llm.provider must be local_gemma", file=sys.stderr)
        sys.exit(1)

    client = LLMClient(llm)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 'Reply with exactly one word: "ok".'},
    ]
    resp = client.complete(messages, use_cache=False)
    text = (resp.text or "").strip()
    print("warmup_ok model=", resp.model or client.model_name)
    print("warmup_reply:", text[:200])
    if not text:
        print("ERROR: empty completion", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
