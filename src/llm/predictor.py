"""Predictor module — run LLM inference over a set of patient records.

Handles all four prompt modes:
  - zero_shot
  - zero_shot_plus
  - few_shot
  - coagent (few-shot + critic feedback)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.llm.api_clients import LLMClient, LLMResponse
from src.llm.prompt_builder import build_messages
from src.llm.output_parser import parse_prediction, extract_logprob_confidence
from src.utils.io import save_json, save_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Maps mode name → prompt template filename
MODE_TEMPLATE_MAP = {
    "zero_shot": "predictor_zero_shot.txt",
    "zero_shot_plus": "predictor_zero_shot_plus.txt",
    "few_shot": "predictor_few_shot.txt",
    "coagent": "predictor_coagent_base.txt",
}


def run_predictions(
    client: LLMClient,
    df: pd.DataFrame,
    mode: str,
    demonstration_cases: str = "",
    critic_feedback: str = "",
    output_dir: str = "data/outputs",
    narrative_col: str = "narrative_current",
    id_col: str = "pair_id",
) -> pd.DataFrame:
    """Run LLM predictions on all rows in *df*.

    Parameters
    ----------
    client : LLMClient
        Initialised LLM client.
    df : DataFrame
        Must contain *narrative_col* and *id_col*.
    mode : str
        One of ``zero_shot``, ``zero_shot_plus``, ``few_shot``, ``coagent``.
    demonstration_cases : str
        Pre-formatted exemplar block (for few_shot and coagent modes).
    critic_feedback : str
        Consolidated critic instructions (for coagent mode).
    output_dir : str
        Where to save raw responses and prompts.

    Returns
    -------
    DataFrame with additional columns:
        raw_response, parsed_prediction, reasoning, parser_status,
        prob_yes, prob_no, raw_response_path
    """
    template_file = MODE_TEMPLATE_MAP.get(mode)
    if not template_file:
        raise ValueError(f"Unknown mode: {mode}")

    odir = Path(output_dir)
    resp_dir = odir / "raw_llm_responses" / mode
    resp_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = odir / "prompts_used" / mode
    prompt_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"LLM {mode}"):
        sample_id = row[id_col]
        narrative = row[narrative_col]

        messages = build_messages(
            template_file=template_file,
            narrative=narrative,
            demonstration_cases=demonstration_cases,
            critic_feedback=critic_feedback,
        )

        prompt_text = "\n---\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        save_text(prompt_text, prompt_dir / f"prompt_{sample_id}.txt")

        try:
            resp: LLMResponse = client.complete(messages)
            raw_text = resp.text
        except Exception as e:
            logger.error("LLM call failed for sample %s: %s", sample_id, e)
            raw_text = ""
            resp = LLMResponse(text="", model=client.model_name)

        # Save raw response
        resp_path = resp_dir / f"{sample_id}.json"
        save_json({"sample_id": sample_id, "mode": mode, "model": resp.model, **resp.to_dict()}, resp_path)

        # Parse
        parsed = parse_prediction(raw_text)
        conf = extract_logprob_confidence(resp.logprobs)

        results.append({
            id_col: sample_id,
            "prompt_mode": mode,
            "model_name": resp.model or client.model_name,
            "raw_response": raw_text,
            "parsed_prediction": parsed["prediction"],
            "reasoning": parsed["reasoning"],
            "parser_status": parsed["parser_status"],
            "prob_yes": conf["prob_yes"] if conf else None,
            "prob_no": conf["prob_no"] if conf else None,
            "raw_response_path": str(resp_path),
        })

    result_df = pd.DataFrame(results)
    return result_df
