"""Parse LLM output text into structured predictions.

Supports:
- Structured triple: ``Prediction:``, ``Probability:``, ``Reasoning:`` (prompts_v2).
- Legacy outputs: plain ``Yes``/``No``, ``Prediction:`` only, or ambiguous fallbacks.
- Multitask strict JSON (prompt_v3+): see ``parse_multitask_output``.

Probability is never imputed; invalid or missing values are flagged explicitly.
"""

from __future__ import annotations

import json
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


# ---- Multitask JSON (prompt_v3) -------------------------------------------------
#
# Expected top-level object (strict):
#   {
#     "lipid_next": {"prediction": "Yes"|"No", "probability": <float in [0,1]>},
#     "diabetes_current": {...},
#     "hypertension_current": {...},
#     "obesity_current": {...},
#     "cardio_next": {...},
#     "kidney_next": {...},
#     "stroke_next": {...},
#     "reasoning": "<string>"
#   }
#
# Failure cases (function returns None — no imputation):
#   - Text empty, or not valid JSON object
#   - Any required task key or "reasoning" missing
#   - Any task value not a dict, or missing prediction/probability
#   - prediction not exactly Yes/No (case-insensitive)
#   - probability NaN/inf or outside [0, 1]
#   - reasoning not a str
#   - Extra non-whitespace characters after the JSON object (trailing prose)

MULTITASK_JSON_TASK_KEYS: Tuple[str, ...] = (
    "lipid_next",
    "diabetes_current",
    "hypertension_current",
    "obesity_current",
    "cardio_next",
    "kidney_next",
    "stroke_next",
)

MULTITASK_TASK_KEY_TO_PREFIX: Dict[str, str] = {
    "lipid_next": "lipid",
    "diabetes_current": "diabetes",
    "hypertension_current": "hypertension",
    "obesity_current": "obesity",
    "cardio_next": "cardio",
    "kidney_next": "kidney",
    "stroke_next": "stroke",
}


def multitask_flat_column_names() -> List[str]:
    """Column names appended to predictor CSV when ``multitask=True`` (excludes ``reasoning``)."""
    cols: List[str] = []
    for key in MULTITASK_JSON_TASK_KEYS:
        p = MULTITASK_TASK_KEY_TO_PREFIX[key]
        cols.append(f"{p}_prob")
        cols.append(f"{p}_pred")
    return cols


def _strip_markdown_json_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_top_level_json_object(text: str) -> Optional[Any]:
    """Parse a single JSON object; trailing non-whitespace after the object is rejected."""
    text = (text or "").strip()
    if not text:
        return None
    text = _strip_markdown_json_fence(text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            return None
        if text[end:].strip():
            return None
        return obj
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _multitask_prob_float(val: Any) -> Optional[float]:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        x = float(val)
        if math.isnan(x) or math.isinf(x):
            return None
        if x < 0.0 or x > 1.0:
            return None
        return x
    return None


def _multitask_yes_no_pred(val: Any) -> Optional[int]:
    if not isinstance(val, str):
        return None
    s = val.strip().lower()
    if s == "yes":
        return 1
    if s == "no":
        return 0
    return None


def parse_multitask_output(text: str) -> Optional[Dict[str, Any]]:
    """Parse multitask strict JSON into a flat dict of scores plus ``reasoning`` str.

    Returns None on any validation failure (see module comments above).
    On success, keys are: ``{prefix}_prob``, ``{prefix}_pred`` for each clinical task,
    and ``reasoning`` (string). ``prefix`` is e.g. ``lipid``, ``diabetes``, ...
    """
    data = _parse_top_level_json_object(text)
    if data is None:
        return None

    for tk in MULTITASK_JSON_TASK_KEYS:
        if tk not in data:
            return None
    if "reasoning" not in data:
        return None

    if not isinstance(data["reasoning"], str):
        return None

    out: Dict[str, Any] = {"reasoning": data["reasoning"]}

    for tk in MULTITASK_JSON_TASK_KEYS:
        block = data[tk]
        if not isinstance(block, dict):
            return None
        if "prediction" not in block or "probability" not in block:
            return None
        pred_i = _multitask_yes_no_pred(block["prediction"])
        prob = _multitask_prob_float(block["probability"])
        if pred_i is None or prob is None:
            return None
        prefix = MULTITASK_TASK_KEY_TO_PREFIX[tk]
        out[f"{prefix}_prob"] = prob
        out[f"{prefix}_pred"] = pred_i

    return out


def test_parse_multitask_output_example() -> None:
    """Sanity check: one valid prompt_v3-style JSON blob; asserts all expected keys exist."""
    sample = (
        '{"lipid_next": {"prediction": "Yes", "probability": 0.72}, '
        '"diabetes_current": {"prediction": "No", "probability": 0.10}, '
        '"hypertension_current": {"prediction": "Yes", "probability": 0.80}, '
        '"obesity_current": {"prediction": "No", "probability": 0.20}, '
        '"cardio_next": {"prediction": "No", "probability": 0.30}, '
        '"kidney_next": {"prediction": "No", "probability": 0.15}, '
        '"stroke_next": {"prediction": "No", "probability": 0.10}, '
        '"reasoning": "smoke test"}'
    )
    parsed = parse_multitask_output(sample)
    assert parsed is not None
    for name in multitask_flat_column_names():
        assert name in parsed
    assert parsed["reasoning"] == "smoke test"
    assert parsed["lipid_pred"] == 1 and parsed["lipid_prob"] == 0.72
    assert parsed["diabetes_pred"] == 0


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


def _smoke_parse_multitask_five_samples() -> None:
    """Dev smoke: five parse attempts (1 valid + 4 expected None); prints one successful parse."""
    valid = (
        '{"lipid_next": {"prediction": "Yes", "probability": 0.72}, '
        '"diabetes_current": {"prediction": "No", "probability": 0.10}, '
        '"hypertension_current": {"prediction": "Yes", "probability": 0.80}, '
        '"obesity_current": {"prediction": "No", "probability": 0.20}, '
        '"cardio_next": {"prediction": "No", "probability": 0.30}, '
        '"kidney_next": {"prediction": "No", "probability": 0.15}, '
        '"stroke_next": {"prediction": "No", "probability": 0.10}, '
        '"reasoning": "example"}'
    )
    invalid_cases = [
        "not json",
        '{"lipid_next": {"prediction": "Yes", "probability": 0.5}}',
        valid + " trailing",
        '{"lipid_next": {"prediction": "Maybe", "probability": 0.5}, '
        '"diabetes_current": {"prediction": "No", "probability": 0.1}, '
        '"hypertension_current": {"prediction": "No", "probability": 0.1}, '
        '"obesity_current": {"prediction": "No", "probability": 0.1}, '
        '"cardio_next": {"prediction": "No", "probability": 0.1}, '
        '"kidney_next": {"prediction": "No", "probability": 0.1}, '
        '"stroke_next": {"prediction": "No", "probability": 0.1}, '
        '"reasoning": "x"}',
    ]
    samples = [valid, *invalid_cases]
    ok = None
    for i, s in enumerate(samples):
        r = parse_multitask_output(s)
        print(f"sample[{i}] -> {'OK' if r else 'None'}")
        if i == 0:
            ok = r
    if ok is not None:
        print("parsed[0] =", ok)


if __name__ == "__main__":
    test_parse_multitask_output_example()
    _smoke_parse_multitask_five_samples()
