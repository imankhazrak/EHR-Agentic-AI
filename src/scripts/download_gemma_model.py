"""Download a Hugging Face model snapshot to a local directory (offline-friendly runs).

Usage (from repo root, with HF_TOKEN in .env or env for gated models):
    export PYTHONPATH="$(pwd)"
    python -m src.scripts.download_gemma_model --model-id google/gemma-4-e4b-it

Then point configs/default.yaml llm.model_name at the printed path, or:
    export LLM_MODEL_NAME="/abs/path/to/models/hf_snapshots/google--gemma-4-e4b-it"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.utils.config_utils import load_config


def _safe_dir_name(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "--", model_id.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot-download a HF model to disk.")
    parser.add_argument(
        "--model-id",
        default="google/gemma-4-e4b-it",
        help="Hugging Face repo id (e.g. google/gemma-4-e4b-it).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to store files (default: models/hf_snapshots/<sanitized-id>).",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="For loading .env via load_config.")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    load_config(args.config, args.overrides)  # loads .env

    from huggingface_hub import snapshot_download

    out = Path(args.output_dir) if args.output_dir else Path("models/hf_snapshots") / _safe_dir_name(args.model_id)
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    print("Downloaded snapshot to:", path)
    print("Set in config or environment:")
    print(f'  llm.model_name: "{path}"')
    print(f'  export LLM_MODEL_NAME="{path}"')


if __name__ == "__main__":
    main()
