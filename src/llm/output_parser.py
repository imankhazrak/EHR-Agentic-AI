"""Parse LLM output text into structured predictions.

Supports:
- Structured triple: ``Prediction:``, ``Probability:``, ``Reasoning:`` (prompts_v2).
- Legacy outputs: plain ``Yes``/``No``, ``Prediction:`` only, or ambiguous fallbacks.

Probability is never imputed; invalid or missing values are flagged explicitly.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _empty_parse() -> Dict[str, Any]:
    return {
        "prediction": "unparseable",
        "reasoning": "",
        "parser_status": "unparseable",
        "probability": None,
        "probability_status": "missing",
        "probability_format_warning": False,
        "parse_valid_probability": False,
    }


def _probability_display_ok(raw: str) -> bool:
    """True if string matches two-decimal style (or 0 / 1 without decimals)."""
    s = raw.strip()
    if s in ("0", "1"):
        return True
    if re.fullmatch(r"0\.\d{2}", s):
        return True
    if re.fullmatch(r"1\.00", s):
        return True
    return False


def _parse_probability_token(raw: str) -> Tuple[Optional[float], str, bool]:
    """Return (value, status, format_warning). status in ok|invalid|out_of_range."""
    raw = raw.strip()
    if not raw:
        return None, "invalid", False
    if raw.endswith("%"):
        return None, "invalid", False
    try:
        val = float(raw)
    except ValueError:
        return None, "invalid", False
    if math.isnan(val) or math.isinf(val):
        return None, "invalid", False
    if val < 0.0 or val > 1.0:
        return None, "out_of_range", False
    fmt_warn = not _probability_display_ok(raw)
    return val, "ok", fmt_warn


def _extract_probability_field(text: str) -> Tuple[Optional[float], str, bool]:
    m = re.search(r"(?im)^\s*Probability:\s*(.+?)\s*$", text)
    if not m:
        return None, "missing", False
    return _parse_probability_token(m.group(1))


def _extract_reasoning_field(text: str, legacy_tail: str = "") -> str:
    parts = re.split(r"(?im)^\s*Reasoning:\s*", text, maxsplit=1)
    if len(parts) > 1:
        return parts[1].strip()
    return legacy_tail.strip()


def _extract_prediction_from_labels(text: str) -> Tuple[Optional[str], str]:
    yes_m = re.search(r"(?im)^\s*Prediction:\s*Yes\s*$", text, re.MULTILINE)
    no_m = re.search(r"(?im)^\s*Prediction:\s*No\s*$", text, re.MULTILINE)
    if yes_m and not no_m:
        return "Yes", "structured"
    if no_m and not yes_m:
        return "No", "structured"
    if yes_m and no_m:
        return ("Yes" if yes_m.start() <= no_m.start() else "No"), "structured"
    m = re.search(r"(?im)Prediction:\s*(Yes|No)\b", text)
    if m:
        return m.group(1).capitalize(), "fallback"
    return None, ""


def parse_prediction(text: str) -> Dict[str, Any]:
    """Parse prediction, optional probability, and reasoning.

    Returns
    -------
    dict with keys:
        - ``prediction``: ``"Yes"`` | ``"No"`` | ``"unparseable"``
        - ``reasoning``: str
        - ``parser_status``: ``"first_line"`` | ``"structured"`` | ``"fallback"`` | ``"unparseable"``
        - ``probability``: float or None
        - ``probability_status``: ``"ok"`` | ``"missing"`` | ``"invalid"`` | ``"out_of_range"``
        - ``probability_format_warning``: bool
        - ``parse_valid_probability``: bool
    """
    text = (text or "").strip()
    if not text:
        return _empty_parse()

    lines = text.split("\n")
    first_line = lines[0].strip()
    prob_val, prob_status, prob_warn = _extract_probability_field(text)

    # ---- Structured: first line ``Prediction: Yes/No`` ----
    m_pred_line = re.match(r"(?i)^Prediction\s*:\s*(Yes|No)\b", first_line)
    if m_pred_line:
        pred = m_pred_line.group(1).capitalize()
        reasoning = _extract_reasoning_field(text)
        out = {
            "prediction": pred,
            "reasoning": reasoning,
            "parser_status": "structured",
            "probability": prob_val if prob_status == "ok" else None,
            "probability_status": prob_status,
            "probability_format_warning": prob_warn,
            "parse_valid_probability": prob_status == "ok",
        }
        if prob_status in ("invalid", "out_of_range"):
            logger.warning("Probability parse issue (structured): %s", text[:200].replace("\n", " "))
        elif prob_status == "ok" and prob_warn:
            logger.info("Probability numeric but not two-decimal display: %s", text[:120].replace("\n", " "))
        return out

    # ---- Legacy: first line plain Yes/No (no ``Prediction:`` on line 1) ----
    match_fl = re.match(r"^(Yes|No)\b", first_line, re.IGNORECASE)
    if match_fl:
        pred = match_fl.group(1).capitalize()
        legacy_tail = "\n".join(lines[1:])
        reasoning = _extract_reasoning_field(text, legacy_tail=legacy_tail)
        if not reasoning:
            reasoning = legacy_tail
        out = {
            "prediction": pred,
            "reasoning": reasoning,
            "parser_status": "first_line",
            "probability": prob_val if prob_status == "ok" else None,
            "probability_status": prob_status,
            "probability_format_warning": prob_warn,
            "parse_valid_probability": prob_status == "ok",
        }
        if prob_status in ("invalid", "out_of_range"):
            logger.warning("Invalid probability on first-line Yes/No parse: %s", text[:200].replace("\n", " "))
        return out

    # ---- ``Prediction:`` elsewhere or other patterns ----
    pred_label, hint = _extract_prediction_from_labels(text)
    if pred_label in ("Yes", "No"):
        reasoning = _extract_reasoning_field(text)
        if not reasoning:
            reasoning = text
        out = {
            "prediction": pred_label,
            "reasoning": reasoning,
            "parser_status": "structured" if hint == "structured" else "fallback",
            "probability": prob_val if prob_status == "ok" else None,
            "probability_status": prob_status,
            "probability_format_warning": prob_warn,
            "parse_valid_probability": prob_status == "ok",
        }
        if prob_status in ("invalid", "out_of_range"):
            logger.warning("Probability parse issue (fallback prediction path): %s", text[:200].replace("\n", " "))
        return out

    # ---- Legacy: dominant Yes/No ----
    first_yes = re.search(r"\bYes\b", text, re.IGNORECASE)
    first_no = re.search(r"\bNo\b", text, re.IGNORECASE)
    if first_yes and (not first_no or first_yes.start() < first_no.start()):
        return {
            "prediction": "Yes",
            "reasoning": text,
            "parser_status": "fallback",
            "probability": prob_val if prob_status == "ok" else None,
            "probability_status": prob_status,
            "probability_format_warning": prob_warn,
            "parse_valid_probability": prob_status == "ok",
        }
    if first_no and (not first_yes or first_no.start() < first_yes.start()):
        return {
            "prediction": "No",
            "reasoning": text,
            "parser_status": "fallback",
            "probability": prob_val if prob_status == "ok" else None,
            "probability_status": prob_status,
            "probability_format_warning": prob_warn,
            "parse_valid_probability": prob_status == "ok",
        }

    logger.warning("Could not parse prediction from response (first 200 chars): %s", text[:200])
    out = _empty_parse()
    out["reasoning"] = text
    return out


def extract_logprob_confidence(
    logprobs: Optional[List[Dict]],
    yes_tokens: Tuple[str, ...] = ("Yes", "yes", "YES"),
    no_tokens: Tuple[str, ...] = ("No", "no", "NO"),
) -> Optional[Dict[str, float]]:
    """Extract normalised probability for Yes/No from the first token's logprobs."""
    if not logprobs:
        return None

    first_token = logprobs[0] if logprobs else None
    if not first_token or not first_token.get("top_logprobs"):
        return None

    yes_logprob = -float("inf")
    no_logprob = -float("inf")

    for top_lp_dict in first_token["top_logprobs"]:
        for token, lp in top_lp_dict.items():
            token_clean = token.strip()
            if token_clean in yes_tokens:
                yes_logprob = max(yes_logprob, lp)
            elif token_clean in no_tokens:
                no_logprob = max(no_logprob, lp)

    if yes_logprob == -float("inf") and no_logprob == -float("inf"):
        return None

    max_lp = max(yes_logprob, no_logprob)
    exp_yes = math.exp(yes_logprob - max_lp) if yes_logprob > -float("inf") else 0.0
    exp_no = math.exp(no_logprob - max_lp) if no_logprob > -float("inf") else 0.0
    total = exp_yes + exp_no

    if total == 0:
        return None

    return {
        "prob_yes": exp_yes / total,
        "prob_no": exp_no / total,
    }
