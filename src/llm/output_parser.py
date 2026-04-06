"""Parse LLM output text into structured predictions.

Parsing strategy:
1. Look for ``Prediction: Yes`` or ``Prediction: No`` on the first line.
2. If not found, search the full text for standalone "Yes" / "No".
3. If ambiguous or not found, mark as ``unparseable``.

Also supports optional logprob-based probability extraction.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_prediction(text: str) -> Dict[str, str]:
    """Parse a prediction and reasoning from LLM response text.

    Returns
    -------
    dict with keys:
        - ``prediction``: ``"Yes"`` | ``"No"`` | ``"unparseable"``
        - ``reasoning``: str (everything after the prediction line)
        - ``parser_status``: ``"first_line"`` | ``"fallback"`` | ``"unparseable"``
    """
    text = text.strip()
    if not text:
        return {"prediction": "unparseable", "reasoning": "", "parser_status": "unparseable"}

    lines = text.split("\n")
    first_line = lines[0].strip()

    # Strategy 1: strict first-line parsing
    match = re.match(r"(?:Prediction\s*:\s*)?(Yes|No)\b", first_line, re.IGNORECASE)
    if match:
        pred = match.group(1).capitalize()
        reasoning = "\n".join(lines[1:]).strip()
        return {"prediction": pred, "reasoning": reasoning, "parser_status": "first_line"}

    # Strategy 2: search full text for Yes/No patterns
    yes_match = re.search(r"\bPrediction\s*:\s*Yes\b", text, re.IGNORECASE)
    no_match = re.search(r"\bPrediction\s*:\s*No\b", text, re.IGNORECASE)

    if yes_match and not no_match:
        return {"prediction": "Yes", "reasoning": text, "parser_status": "fallback"}
    if no_match and not yes_match:
        return {"prediction": "No", "reasoning": text, "parser_status": "fallback"}

    # Strategy 3: look for standalone Yes/No anywhere
    yes_count = len(re.findall(r"\bYes\b", text, re.IGNORECASE))
    no_count = len(re.findall(r"\bNo\b", text, re.IGNORECASE))

    # Only use this if one is clearly dominant as the first occurrence
    first_yes = re.search(r"\bYes\b", text, re.IGNORECASE)
    first_no = re.search(r"\bNo\b", text, re.IGNORECASE)

    if first_yes and (not first_no or first_yes.start() < first_no.start()):
        return {"prediction": "Yes", "reasoning": text, "parser_status": "fallback"}
    if first_no and (not first_yes or first_no.start() < first_yes.start()):
        return {"prediction": "No", "reasoning": text, "parser_status": "fallback"}

    logger.warning("Could not parse prediction from response (first 200 chars): %s", text[:200])
    return {"prediction": "unparseable", "reasoning": text, "parser_status": "unparseable"}


def extract_logprob_confidence(
    logprobs: Optional[List[Dict]],
    yes_tokens: Tuple[str, ...] = ("Yes", "yes", "YES"),
    no_tokens: Tuple[str, ...] = ("No", "no", "NO"),
) -> Optional[Dict[str, float]]:
    """Extract normalised probability for Yes/No from the first token's logprobs.

    This follows the paper's approach: extract top-5 logprobs for the predicting
    token and normalise over the Yes/No candidates.

    Returns
    -------
    dict with ``prob_yes``, ``prob_no`` or None if logprobs unavailable.
    """
    if not logprobs:
        return None

    # Look at the first token's top logprobs
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

    # Normalise via softmax
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
