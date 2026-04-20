"""Print merged config ``paths.outputs`` (for shell wrappers).

Usage::
    python -m src.scripts.print_output_dir --config configs/default.yaml
    python -m src.scripts.print_output_dir --config configs/default.yaml --overrides configs/debug_small.yaml
"""

from __future__ import annotations

import argparse

from src.utils.config_utils import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Print paths.outputs from merged YAML config.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--overrides", nargs="*", default=[])
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides or None)
    print(cfg["paths"]["outputs"], end="")


if __name__ == "__main__":
    main()
