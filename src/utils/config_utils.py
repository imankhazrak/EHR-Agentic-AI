"""Configuration loading and merging utilities."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Repo root (parent of src/) so .env loads even when cwd is not the project directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base* (returns a new dict)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(
    default_path: str = "configs/default.yaml",
    overrides: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Load the default config and merge any override YAML files on top.

    Also loads environment variables from ``.env`` if present.
    """
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv()  # cwd fallback

    with open(default_path, "r") as f:
        cfg = yaml.safe_load(f)

    for path_str in overrides or []:
        p = Path(path_str)
        if p.exists():
            with open(p, "r") as f:
                override = yaml.safe_load(f) or {}
            cfg = deep_merge(cfg, override)

    # Ensure all path values are resolved to Path objects
    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            cfg["paths"][key] = str(Path(val))

    # Optional: isolate a whole run without editing YAML (best practice for comparisons).
    run_outputs = (os.getenv("EHR_OUTPUTS_DIR") or os.getenv("EXPERIMENT_OUTPUT_DIR") or "").strip()
    if run_outputs:
        out_root = str(Path(run_outputs))
        cfg.setdefault("paths", {})["outputs"] = out_root
        if "llm" in cfg:
            cache_override = (os.getenv("EHR_LLM_CACHE_DIR") or "").strip()
            if cache_override:
                cfg["llm"]["cache_dir"] = str(Path(cache_override))
            else:
                cfg["llm"]["cache_dir"] = str(Path(out_root) / "llm_cache")
    finetune_out = (os.getenv("EHR_FINETUNE_OUTPUT_DIR") or "").strip()
    if finetune_out and "llm" in cfg:
        cfg["llm"]["finetune_output_dir"] = str(Path(finetune_out))

    # Override llm.model from env without editing YAML.
    if "llm" in cfg:
        env_model = os.getenv("LLM_MODEL_NAME") or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL")
        if env_model:
            cfg["llm"]["model_name"] = env_model.strip()
            cfg["llm"]["model"] = env_model.strip()
        elif cfg["llm"].get("model_name"):
            cfg["llm"]["model"] = cfg["llm"]["model_name"]
        # OpenAI-compatible base URL: env wins over YAML so HPC can override default localhost.
        bu_env = (os.getenv("OPENAI_BASE_URL") or "").strip()
        if bu_env:
            cfg["llm"]["base_url"] = bu_env
        elif not (cfg["llm"].get("base_url") or "").strip():
            cfg["llm"]["base_url"] = None
        vllm_env = (os.getenv("VLLM_BASE_URL") or "").strip()
        if vllm_env:
            cfg["llm"]["vllm_base_url"] = vllm_env
        mode_env = (os.getenv("LLM_MODE") or "").strip()
        if mode_env:
            cfg["llm"]["mode"] = mode_env
        # Expose HF cache env to downstream clients if provided.
        hf_home = (os.getenv("HF_HOME") or "").strip()
        if hf_home:
            cfg["llm"]["hf_home"] = hf_home

    return cfg
