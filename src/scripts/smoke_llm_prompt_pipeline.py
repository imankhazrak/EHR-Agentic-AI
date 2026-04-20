"""End-to-end smoke test for the MIMIC-III prompt-based LLM pipeline.

Runs the same entrypoints as ``scripts/run_llm_only.sh`` (STAGE=smoke) with
merged config, then checks metrics artifacts. Requires a working LLM backend
and existing ``data/processed/mimiciii/{train,test}.csv``.

Usage (from repo root)::

    python -m src.scripts.smoke_llm_prompt_pipeline \\
        --config configs/default.yaml --overrides configs/debug_small.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.evaluation.summarize_results import collect_results
from src.utils.config_utils import load_config
from src.utils.io import load_json


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd), env=env)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    p = argparse.ArgumentParser(description="Smoke-test LLM prompt pipeline (small split).")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--overrides", nargs="*", default=["configs/debug_small.yaml"])
    p.add_argument("--start-from", default="zero_shot", choices=("zero_shot", "zero_shot_plus", "few_shot", "coagent"))
    args = p.parse_args()
    overrides = list(args.overrides) if args.overrides else None
    cfg = load_config(args.config, overrides)

    out = Path(cfg["paths"]["outputs"])
    pdir = cfg.get("llm", {}).get("prompt_template_dir", "prompts_v2")
    print(f"smoke_llm_prompt_pipeline: outputs={out} prompt_template_dir={pdir}", flush=True)

    base = [sys.executable, "-m"]
    cfg_args: list[str] = ["--config", args.config]
    if overrides:
        cfg_args += ["--overrides", *overrides]

    order = ["zero_shot", "zero_shot_plus", "few_shot", "coagent"]
    start_i = order.index(args.start_from)
    for mode in order[start_i:]:
        mod = {
            "zero_shot": "src.scripts.run_zero_shot",
            "zero_shot_plus": "src.scripts.run_zero_shot_plus",
            "few_shot": "src.scripts.run_few_shot",
            "coagent": "src.scripts.run_coagent",
        }[mode]
        _run(base + [mod] + cfg_args, cwd=root, env=env)

    collect_results(str(out))
    summary = out / "summary_table.csv"
    if not summary.exists():
        raise RuntimeError(f"Expected summary at {summary}")

    zm = load_json(str(out / "llm_zero_shot_metrics.json"))
    assert zm.get("n_total", 0) > 0, "no rows evaluated"
    assert zm.get("n_valid_prediction", 0) > 0, "no valid hard predictions"
    if zm.get("n_valid_probability", 0) > 0:
        assert "auc" in zm and "auprc" in zm, "metrics JSON missing auc/auprc keys"
    else:
        print("WARN: zero-shot smoke had no valid probabilities (model format?)", flush=True)

    print("smoke_llm_prompt_pipeline: OK", flush=True)


if __name__ == "__main__":
    main()
