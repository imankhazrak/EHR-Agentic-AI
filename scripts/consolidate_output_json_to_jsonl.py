#!/usr/bin/env python3
"""Merge all *.json under an output directory into one JSONL file.

Root-level JSON files are written as ``{"__artifact__": "<relpath>", "json": {...}}``.
Per-sample files under ``raw_llm_responses/<mode>/<pair_id>.json`` are written as
their original objects (they already include ``mode`` and ``sample_id``).

Example::

    python scripts/consolidate_output_json_to_jsonl.py data/outputs/mimiciii_gpt4o_mini \\
        --out data/outputs/mimiciii_gpt4o_mini/all_outputs.jsonl --delete-sources
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Merge directory JSON files into one JSONL.")
    p.add_argument("root", type=Path, help="Output folder (e.g. data/outputs/mimiciii_gpt4o_mini)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination JSONL (default: <root>/all_outputs.jsonl)",
    )
    p.add_argument(
        "--delete-sources",
        action="store_true",
        help="Remove source .json files after a successful write",
    )
    args = p.parse_args()

    root = args.root.resolve()
    out = (args.out or (root / "all_outputs.jsonl")).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    all_paths = sorted(root.rglob("*.json"), key=lambda x: str(x))
    # Do not read an existing output at the same path
    all_paths = [x for x in all_paths if x.resolve() != out]

    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as w:
        for fp in all_paths:
            rel = fp.relative_to(root)
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON: {fp}: {e}") from e

            if rel.parts[:1] == ("raw_llm_responses",) and len(rel.parts) == 3:
                line = payload
            else:
                line = {"__artifact__": str(rel).replace("\\", "/"), "json": payload}
            w.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            n += 1

    print(f"Wrote {n} lines -> {out}")

    if args.delete_sources:
        for fp in all_paths:
            fp.unlink()
        print(f"Deleted {len(all_paths)} source JSON files")
        # Drop empty mode directories under raw_llm_responses/
        rr = root / "raw_llm_responses"
        if rr.is_dir():
            for sub in sorted(rr.iterdir(), key=lambda p: str(p), reverse=True):
                if sub.is_dir():
                    try:
                        sub.rmdir()
                    except OSError:
                        pass
            try:
                rr.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    main()
