"""Build prompts from Jinja2 templates for each prompt mode.

Templates are stored in the ``prompts/`` directory and use Jinja2 syntax.
This module loads them and renders with the appropriate variables.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, BaseLoader, ChainableUndefined

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Directory containing prompt template files
PROMPT_DIR = Path("prompts")


def _load_template(filename: str) -> str:
    """Load raw template text from the prompts directory."""
    fp = PROMPT_DIR / filename
    if not fp.exists():
        raise FileNotFoundError(f"Prompt template not found: {fp}")
    text = fp.read_text(encoding="utf-8")
    # Strip Jinja raw/endraw wrappers if present (used to protect {% in files)
    text = re.sub(r"\{%\s*raw\s*%\}", "", text)
    text = re.sub(r"\{%\s*endraw\s*%\}", "", text)
    return text


def _extract_blocks(template_text: str) -> Dict[str, str]:
    """Extract named Jinja2 blocks from template text.

    Returns dict with keys like ``"system"`` and ``"user"``.
    """
    blocks: Dict[str, str] = {}
    pattern = re.compile(r"\{%\s*block\s+(\w+)\s*%\}(.*?)\{%\s*endblock\s*%\}", re.DOTALL)
    for match in pattern.finditer(template_text):
        block_name = match.group(1)
        block_body = match.group(2).strip()
        blocks[block_name] = block_body
    return blocks


def _render(template_str: str, variables: Dict) -> str:
    """Render a Jinja2 template string with variables."""
    env = Environment(loader=BaseLoader(), undefined=ChainableUndefined)
    # Remove comment lines
    template_str = re.sub(r"\{#.*?#\}", "", template_str, flags=re.DOTALL)
    tpl = env.from_string(template_str)
    return tpl.render(**variables).strip()


def build_messages(
    template_file: str,
    narrative: str = "",
    demonstration_cases: str = "",
    critic_feedback: str = "",
    extra_vars: Optional[Dict] = None,
) -> List[Dict[str, str]]:
    """Build a list of chat messages [{role, content}, ...] from a prompt template.

    Parameters
    ----------
    template_file : str
        Filename inside ``prompts/``, e.g. ``"predictor_few_shot.txt"``.
    narrative : str
        The current-visit narrative to insert.
    demonstration_cases : str
        Pre-formatted few-shot exemplar block.
    critic_feedback : str
        Consolidated critic instructions (for CoAgent mode).
    extra_vars : dict, optional
        Additional template variables.

    Returns
    -------
    list of dict
        Messages suitable for chat-completion API calls.
    """
    raw = _load_template(template_file)
    blocks = _extract_blocks(raw)

    variables = {
        "narrative": narrative,
        "demonstration_cases": demonstration_cases,
        "critic_feedback": critic_feedback,
        **(extra_vars or {}),
    }

    messages: List[Dict[str, str]] = []
    if "system" in blocks:
        messages.append({"role": "system", "content": _render(blocks["system"], variables)})
    if "user" in blocks:
        messages.append({"role": "user", "content": _render(blocks["user"], variables)})

    return messages
