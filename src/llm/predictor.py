"""Predictor module — run LLM inference over a set of patient records.

Handles all four prompt modes:
  - zero_shot
  - zero_shot_plus
  - few_shot
  - coagent (few-shot + critic feedback)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from src.llm.api_clients import LLMClient, LLMResponse
from src.llm.prompt_builder import build_messages
from src.llm.output_parser import (
    extract_logprob_confidence,
    multitask_flat_column_names,
    parse_multitask_output_with_meta,
    parse_prediction,
)
from src.utils.io import save_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Maps mode name → prompt template filename
MODE_TEMPLATE_MAP = {
    "zero_shot": "predictor_zero_shot.txt",
    "zero_shot_plus": "predictor_zero_shot_plus.txt",
    "few_shot": "predictor_few_shot.txt",
    "coagent_calibration": "predictor_few_shot.txt",
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
    prompt_template_dir: Optional[Union[str, Path]] = None,
    multitask: bool = False,
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
    prompt_template_dir : str or Path, optional
        Directory with Jinja templates (defaults via ``build_messages``).

    Returns
    -------
    DataFrame with additional columns:
        raw_response, parsed_prediction, parsed_probability, probability_parse_status,
        parse_valid_probability, reasoning, parser_status, prob_yes, prob_no,
        raw_response_path (``responses.jsonl`` for this mode)

    When ``multitask=True`` (prompt_v3 JSON), columns from ``multitask_flat_column_names()``
    are appended (``lipid_prob``, ``lipid_pred``, …). Legacy lipid columns
    (``parsed_prediction``, ``parsed_probability``, ``pred_binary``) are filled from
    ``lipid_next`` when JSON parses; on parse failure they are left unparseable/NaN
    without falling back to ``parse_prediction`` (avoids mis-reading JSON fragments).
    """
    template_file = MODE_TEMPLATE_MAP.get(mode)
    if not template_file:
        raise ValueError(f"Unknown mode: {mode}")

    odir = Path(output_dir)
    resp_dir = odir / "raw_llm_responses" / mode
    resp_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = odir / "prompts_used" / mode
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # One JSONL per mode run (avoids huge inode counts on shared filesystems).
    resp_path = resp_dir / "responses.jsonl"
    results = []

    with open(resp_path, "w", encoding="utf-8") as jsonl_f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"LLM {mode}"):
            sample_id = row[id_col]
            narrative = row[narrative_col]

            messages = build_messages(
                template_file=template_file,
                narrative=narrative,
                demonstration_cases=demonstration_cases,
                critic_feedback=critic_feedback,
                prompt_template_dir=prompt_template_dir,
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

            record = {"sample_id": sample_id, "mode": mode, "model": resp.model, **resp.to_dict()}
            jsonl_f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
            jsonl_f.flush()

            # Parse (single-task text vs multitask JSON — inference above is unchanged)
            conf = extract_logprob_confidence(resp.logprobs)
            pred_map = {"Yes": 1, "No": 0}
            mt_cols = multitask_flat_column_names()
            nan = float("nan")

            if multitask:
                mt = parse_multitask_output_with_meta(raw_text)
                mt_fragment = {c: mt.get(c, nan) for c in mt_cols}
                reasoning = str(mt.get("reasoning", ""))
                lipid_pred = mt_fragment.get("lipid_pred", nan)
                lipid_prob = mt_fragment.get("lipid_prob", nan)
                if pd.notna(lipid_pred):
                    parsed_prediction = "Yes" if int(lipid_pred) == 1 else "No"
                    pred_binary = int(lipid_pred)
                else:
                    parsed_prediction = "unparseable"
                    pred_binary = None
                if pd.notna(lipid_prob):
                    parsed_probability = float(lipid_prob)
                    probability_parse_status = "ok"
                    parse_valid_probability = True
                else:
                    parsed_probability = nan
                    probability_parse_status = "missing"
                    parse_valid_probability = False
                probability_format_warning = False
                parser_status = str(mt.get("parser_status", "multitask_parse_failed"))
                row = {
                    id_col: sample_id,
                    "prompt_mode": mode,
                    "model_name": resp.model or client.model_name,
                    "raw_response": raw_text,
                    "parsed_prediction": parsed_prediction,
                    "pred_binary": pred_binary,
                    "parsed_probability": parsed_probability,
                    "probability_parse_status": probability_parse_status,
                    "parse_valid_probability": parse_valid_probability,
                    "probability_format_warning": probability_format_warning,
                    "reasoning": reasoning,
                    "parser_status": parser_status,
                    "parse_failure_reason": mt.get("parse_failure_reason", ""),
                    "salvage_used": bool(mt.get("salvage_used", False)),
                    "n_tasks_salvaged": int(mt.get("n_tasks_salvaged", 0)),
                    "prob_yes": conf["prob_yes"] if conf else None,
                    "prob_no": conf["prob_no"] if conf else None,
                    "raw_response_path": str(resp_path),
                    **mt_fragment,
                }
            else:
                parsed = parse_prediction(raw_text)
                row = {
                    id_col: sample_id,
                    "prompt_mode": mode,
                    "model_name": resp.model or client.model_name,
                    "raw_response": raw_text,
                    "parsed_prediction": parsed["prediction"],
                    "pred_binary": pred_map.get(parsed["prediction"]),
                    "parsed_probability": parsed.get("probability"),
                    "probability_parse_status": parsed.get("probability_status", "missing"),
                    "parse_valid_probability": bool(parsed.get("parse_valid_probability")),
                    "probability_format_warning": bool(parsed.get("probability_format_warning")),
                    "reasoning": parsed["reasoning"],
                    "parser_status": parsed["parser_status"],
                    "prob_yes": conf["prob_yes"] if conf else None,
                    "prob_no": conf["prob_no"] if conf else None,
                    "raw_response_path": str(resp_path),
                }
            results.append(row)

    result_df = pd.DataFrame(results)
    return result_df
