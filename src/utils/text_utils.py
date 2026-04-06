"""Text processing helpers."""

from __future__ import annotations

import re
from typing import List, Optional


def clean_code_name(name: str) -> str:
    """Normalise a medical code description for narrative use."""
    if not name:
        return ""
    name = name.strip()
    # Remove trailing commas / semicolons
    name = re.sub(r"[;,]+$", "", name)
    return name


def join_items(items: List[str], max_items: Optional[int] = None, conjunction: str = "; ") -> str:
    """Join a list of text items with a conjunction, optionally truncating."""
    if not items:
        return ""
    unique = list(dict.fromkeys(items))  # deduplicate preserving order
    if max_items and len(unique) > max_items:
        unique = unique[:max_items]
    return conjunction.join(unique)


def truncate_text(text: str, max_chars: int = 12000) -> str:
    """Hard-truncate text to a character limit (safety net for LLM context)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"
