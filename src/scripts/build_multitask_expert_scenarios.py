"""Build expert-facing multitask scenario pack (GPT-4o-mini vs Gemma4, prompt v3).

Reads:
  - Processed multitask test labels + narrative
  - GPT-4o-mini multitask test: llm_zero_shot_results.csv (fair match to Gemma zero-shot)
  - Gemma4 multitask test: raw_llm_responses/zero_shot/responses.jsonl

Writes under data/outputs/case_studies/:
  - multitask_expert_scenarios.jsonl
  - multitask_expert_scenarios_readable.md / .html
  - multitask_expert_scenarios_index.csv
  - multitask_expert_scenarios_manifest.json

Usage::
    python -m src.scripts.build_multitask_expert_scenarios
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.llm.output_parser import multitask_flat_column_names, parse_multitask_output_with_meta

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

TASK_SPECS: List[Tuple[str, str, str, str]] = [
    ("lipid", "lipid_pred", "lipid_prob", "label_lipid_next"),
    ("diabetes", "diabetes_pred", "diabetes_prob", "label_diabetes_current"),
    ("hypertension", "hypertension_pred", "hypertension_prob", "label_hypertension_current"),
    ("obesity", "obesity_pred", "obesity_prob", "label_obesity_current"),
    ("cardio", "cardio_pred", "cardio_prob", "label_cardio_next"),
    ("kidney", "kidney_pred", "kidney_prob", "label_kidney_next"),
    ("stroke", "stroke_pred", "stroke_prob", "label_stroke_next"),
]

NARRATIVE_MAX = 900


def _truncate_text(x: Any, max_chars: int) -> str:
    s = str(x or "")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _safe_int(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or np.isnan(x))):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _safe_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or np.isnan(x))):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load_gemma_from_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    mt_cols = multitask_flat_column_names()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = int(obj["sample_id"])
            raw_text = str(obj.get("text") or "")
            mt = parse_multitask_output_with_meta(raw_text)
            row: Dict[str, Any] = {
                "pair_id": sid,
                "gemma_model_name": str(obj.get("model") or ""),
                "gemma_raw_response": raw_text,
                "gemma_reasoning": str(mt.get("reasoning") or ""),
                "gemma_parser_status": str(mt.get("parser_status") or ""),
                "gemma_parse_failure_reason": str(mt.get("parse_failure_reason") or ""),
                "gemma_salvage_used": bool(mt.get("salvage_used", False)),
                "gemma_n_tasks_salvaged": int(mt.get("n_tasks_salvaged", 0)),
            }
            for c in mt_cols:
                v = mt.get(c)
                row[f"gemma_{c}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def _clinical_summary(row: pd.Series) -> str:
    """Short de-identified summary from narrative + codes."""
    nar = str(row.get("narrative_current") or "")
    nar = re.sub(r"\s+", " ", nar).strip()
    if len(nar) > NARRATIVE_MAX:
        nar = nar[: NARRATIVE_MAX - 3] + "..."
    codes = str(row.get("diagnoses_codes_current") or "")[:120]
    return (
        "De-identified cohort record (pair_id only). "
        f"Diagnosis codes (truncated): {codes}. "
        f"Current-visit narrative (truncated): {nar}"
    )


def _raw_case_snapshot(row: pd.Series) -> Dict[str, Any]:
    """Compact raw row excerpt to mirror single-task case-study style."""
    keys = [
        "pair_id",
        "diagnoses_codes_current",
        "procedures_codes_current",
        "medications_current",
        "label_lipid_next",
        "label_diabetes_current",
        "label_hypertension_current",
        "label_obesity_current",
        "label_cardio_next",
        "label_kidney_next",
        "label_stroke_next",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = row.get(k)
    out["narrative_current"] = _truncate_text(row.get("narrative_current"), 3500)
    return out


def _per_task_state(
    row: pd.Series, prefix: str
) -> Dict[str, Dict[str, Any]]:
    """prefix '' for gpt columns (lipid_pred), 'gemma_' for gemma."""
    out: Dict[str, Dict[str, Any]] = {}
    for task, pred_c, prob_c, _ in TASK_SPECS:
        pc = f"{prefix}{pred_c}"
        qc = f"{prefix}{prob_c}"
        p = _safe_int(row.get(pc))
        q = _safe_float(row.get(qc))
        out[task] = {"pred": p, "prob": q}
    return out


def _disagreement(
    gpt: Dict[str, Dict[str, Any]], gem: Dict[str, Dict[str, Any]]
) -> Tuple[int, List[str], Dict[str, Optional[float]]]:
    tasks_differ: List[str] = []
    delta_probs: Dict[str, Optional[float]] = {}
    n = 0
    for task in gpt:
        gp, gg = gpt[task]["pred"], gem[task]["pred"]
        if gp is None or gg is None:
            delta_probs[task] = None
            continue
        pg, pm = gpt[task]["prob"], gem[task]["prob"]
        if pg is not None and pm is not None:
            delta_probs[task] = abs(pg - pm)
        else:
            delta_probs[task] = None
        if gp != gg:
            n += 1
            tasks_differ.append(task)
    return n, tasks_differ, delta_probs


def _wrong_tasks_vs_gold(
    preds: Dict[str, Dict[str, Any]], labels: Dict[str, int]
) -> List[str]:
    wrong = []
    for task, pred_c, _, lab_c in TASK_SPECS:
        lab = labels.get(task)
        p = preds[task]["pred"]
        if lab is None or p is None:
            continue
        if int(p) != int(lab):
            wrong.append(task)
    return wrong


def _collect_tags(
    row: pd.Series,
    gpt_state: Dict[str, Dict[str, Any]],
    gem_state: Dict[str, Dict[str, Any]],
    n_dis: int,
    tasks_dis: List[str],
    delta_probs: Dict[str, Optional[float]],
    gpt_wrong: List[str],
    gem_wrong: List[str],
) -> List[str]:
    tags: Set[str] = set()
    if n_dis >= 1:
        tags.add("cross_model_disagreement")
    if n_dis >= 3:
        tags.add("cross_model_disagreement_high")
    if "lipid" in tasks_dis:
        tags.add("lipid_task_disagreement")
    if row.get("parse_failure_reason") == "reasoning_unclosed" or str(
        row.get("parser_status", "")
    ).startswith("multitask_salvaged"):
        tags.add("gpt_parse_salvage")
    if row.get("gemma_parse_failure_reason") == "reasoning_unclosed" or str(
        row.get("gemma_parser_status", "")
    ).startswith("multitask_salvaged"):
        tags.add("gemma_parse_salvage")
    if int(row.get("gemma_n_tasks_salvaged", 0)) < 7 and int(row.get("gemma_n_tasks_salvaged", 0)) > 0:
        tags.add("gemma_partial_output")
    max_dp = max((d or 0) for d in delta_probs.values() if d is not None) if any(
        d is not None for d in delta_probs.values()
    ) else 0.0
    if n_dis >= 1 and max_dp >= 0.5:
        tags.add("high_probability_gap_disagreement")
    if not tasks_dis and (gpt_wrong or gem_wrong):
        if gpt_wrong and gem_wrong and set(gpt_wrong) == set(gem_wrong):
            tags.add("both_models_same_wrong_tasks")
        elif gpt_wrong and not gem_wrong:
            tags.add("gpt_wrong_gem_correct_some")
        elif gem_wrong and not gpt_wrong:
            tags.add("gem_wrong_gpt_correct_some")
    if not gpt_wrong and not gem_wrong:
        tags.add("both_models_all_tasks_correct")
    return sorted(tags)


def _pick_stratified(
    df_scored: pd.DataFrame,
    seed: int,
    n_total: int = 24,
) -> pd.DataFrame:
    """Quota-based selection with deterministic tie-break (stable sort + head)."""
    _ = seed  # reserved for future weighted sampling
    df = df_scored.copy()

    def pool(mask: pd.Series, k: int, sort_cols: List[str], ascending: List[bool]) -> pd.DataFrame:
        sub = df.loc[mask].sort_values(sort_cols + ["pair_id"], ascending=ascending + [True], kind="mergesort")
        return sub.head(k)

    chosen_ids: Set[int] = set()
    chunks: List[pd.DataFrame] = []

    # 1) Strong disagreement
    m1 = (df["n_disagree_tasks"] >= 3) & (~df["pair_id"].isin(chosen_ids))
    p1 = pool(m1, 8, ["n_disagree_tasks", "max_delta_prob"], [False, False])
    chunks.append(p1)
    chosen_ids.update(p1["pair_id"].tolist())

    # 2) Moderate disagreement
    m2 = (df["n_disagree_tasks"].isin([1, 2])) & (~df["pair_id"].isin(chosen_ids))
    p2 = pool(m2, 6, ["n_disagree_tasks", "max_delta_prob"], [False, False])
    chunks.append(p2)
    chosen_ids.update(p2["pair_id"].tolist())

    # 3) Parse / salvage interest
    m3 = (
        df["scenario_tags"]
        .apply(lambda t: "gpt_parse_salvage" in t or "gemma_parse_salvage" in t or "gemma_partial_output" in t)
        & (~df["pair_id"].isin(chosen_ids))
    )
    p3 = pool(m3, 4, ["n_disagree_tasks"], [False])
    chunks.append(p3)
    chosen_ids.update(p3["pair_id"].tolist())

    # 4) Same wrong pattern
    m4 = df["scenario_tags"].apply(lambda t: "both_models_same_wrong_tasks" in t) & (~df["pair_id"].isin(chosen_ids))
    p4 = pool(m4, 3, ["n_disagree_tasks"], [True])
    chunks.append(p4)
    chosen_ids.update(p4["pair_id"].tolist())

    # 5) Lipid-focused disagreement
    m5 = df["scenario_tags"].apply(lambda t: "lipid_task_disagreement" in t) & (~df["pair_id"].isin(chosen_ids))
    p5 = pool(m5, 3, ["max_delta_prob"], [False])
    chunks.append(p5)
    chosen_ids.update(p5["pair_id"].tolist())

    picked = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if len(picked) < n_total:
        rest = df[~df["pair_id"].isin(chosen_ids)].sort_values(
            ["n_disagree_tasks", "max_delta_prob"], ascending=[False, False], kind="mergesort"
        )
        need = n_total - len(picked)
        extra = rest.head(need)
        picked = pd.concat([picked, extra], ignore_index=True)

    picked = picked.drop_duplicates(subset=["pair_id"]).head(n_total)
    if len(picked) < n_total:
        rest = df[~df["pair_id"].isin(set(picked["pair_id"].tolist()))].sort_values(
            "pair_id", kind="mergesort"
        )
        picked = pd.concat([picked, rest.head(n_total - len(picked))], ignore_index=True)
    return picked.sort_values("pair_id", kind="mergesort").reset_index(drop=True)


def _build_payload(
    row: pd.Series,
    scenario_idx: int,
    labels: Dict[str, int],
    gpt_state: Dict[str, Dict[str, Any]],
    gem_state: Dict[str, Dict[str, Any]],
    n_dis: int,
    tasks_dis: List[str],
    delta_probs: Dict[str, Optional[float]],
    tags: List[str],
) -> Dict[str, Any]:
    gpt_wrong = _wrong_tasks_vs_gold(gpt_state, labels)
    gem_wrong = _wrong_tasks_vs_gold(gem_state, labels)
    return {
        "scenario_id": f"mt_expert_{scenario_idx:03d}",
        "pair_id": int(row["pair_id"]),
        "scenario_tags": tags,
        "ground_truth_labels": labels,
        "clinical_summary": _clinical_summary(row),
        "narrative_excerpt_phi_safe": (
            _truncate_text(row.get("narrative_current"), 3500)
        ),
        "raw_data_example": _raw_case_snapshot(row),
        "primary_model": {
            "role": "primary",
            "name": str(row.get("model_name") or "gpt-4o-mini-2024-07-18"),
            "prompt_family": "prompt_v3_multitask",
            "prompt_mode": str(row.get("prompt_mode") or "zero_shot"),
            "source_csv": "data/outputs/mimiciii_llm_gpt4o_mini_promptv3_multitask_test/llm_zero_shot_results.csv",
        },
        "comparison_model": {
            "role": "comparison",
            "name": str(row.get("gemma_model_name") or "google/gemma-4"),
            "prompt_family": "prompt_v3_multitask",
            "prompt_mode": "zero_shot",
            "source_jsonl": "data/outputs/mimiciii_llm_gemma4_promptv3_multitask_test/raw_llm_responses/zero_shot/responses.jsonl",
        },
        "predictions": {
            "gpt4o_mini": {
                "per_task": gpt_state,
                "reasoning": str(row.get("reasoning") or ""),
                "raw_response_excerpt": str(row.get("raw_response") or "")[:2000],
                "parser_status": str(row.get("parser_status") or ""),
                "parse_failure_reason": str(row.get("parse_failure_reason") or ""),
                "salvage_used": bool(row.get("salvage_used", False)),
                "n_tasks_salvaged": int(row.get("n_tasks_salvaged", 0)),
                "tasks_wrong_vs_gold": gpt_wrong,
            },
            "gemma4": {
                "per_task": gem_state,
                "reasoning": str(row.get("gemma_reasoning") or ""),
                "raw_response_excerpt": str(row.get("gemma_raw_response") or "")[:2000],
                "parser_status": str(row.get("gemma_parser_status") or ""),
                "parse_failure_reason": str(row.get("gemma_parse_failure_reason") or ""),
                "salvage_used": bool(row.get("gemma_salvage_used", False)),
                "n_tasks_salvaged": int(row.get("gemma_n_tasks_salvaged", 0)),
                "tasks_wrong_vs_gold": gem_wrong,
            },
        },
        "cross_model_disagreement": {
            "n_tasks_disagree": n_dis,
            "tasks": tasks_dis,
            "abs_probability_delta_by_task": delta_probs,
        },
        "clinician_questions": [
            "For each clinical task, does the model's probability align with the chart evidence?",
            "Where GPT-4o-mini and Gemma disagree, which rationale fits the record better?",
            "Are any predictions driven by missing or truncated model output (parse/salvage)?",
            "Which errors would be most consequential if used for decision support?",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--gpt-csv", type=Path, default=None)
    ap.add_argument("--gemma-jsonl", type=Path, default=None)
    ap.add_argument("--test-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-scenarios", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.root
    gpt_csv = args.gpt_csv or (
        root / "data/outputs/mimiciii_llm_gpt4o_mini_promptv3_multitask_test/llm_zero_shot_results.csv"
    )
    gemma_jsonl = args.gemma_jsonl or (
        root / "data/outputs/mimiciii_llm_gemma4_promptv3_multitask_test/raw_llm_responses/zero_shot/responses.jsonl"
    )
    test_csv = args.test_csv or (root / "data/processed/mimiciii_multitask/test.csv")
    out_dir = args.out_dir or (root / "data/outputs/case_studies")
    out_dir.mkdir(parents=True, exist_ok=True)

    label_cols = [
        "pair_id",
        "narrative_current",
        "diagnoses_codes_current",
        "procedures_codes_current",
        "medications_current",
    ] + [t[3] for t in TASK_SPECS]
    test_full = pd.read_csv(test_csv)
    test_df = test_full[[c for c in label_cols if c in test_full.columns]].copy()

    gpt = pd.read_csv(gpt_csv)
    # Drop duplicate label columns from gpt merge if present
    for c in [t[3] for t in TASK_SPECS]:
        if c in gpt.columns:
            gpt = gpt.drop(columns=[c])
    gem = _load_gemma_from_jsonl(gemma_jsonl)

    merged = test_df.merge(gpt, on="pair_id", how="inner").merge(gem, on="pair_id", how="inner")
    if len(merged) == 0:
        raise RuntimeError("No rows after merge; check paths and pair_id alignment.")

    rows_meta = []
    for _, row in merged.iterrows():
        labels = {t[0]: int(row[t[3]]) for t in TASK_SPECS if t[3] in row and pd.notna(row[t[3]])}
        gpt_state = _per_task_state(row, "")
        gem_state = _per_task_state(row, "gemma_")
        n_dis, tasks_dis, delta_probs = _disagreement(gpt_state, gem_state)
        gpt_wrong = _wrong_tasks_vs_gold(gpt_state, labels)
        gem_wrong = _wrong_tasks_vs_gold(gem_state, labels)
        tags = _collect_tags(row, gpt_state, gem_state, n_dis, tasks_dis, delta_probs, gpt_wrong, gem_wrong)
        dvals = [d for d in delta_probs.values() if d is not None]
        max_delta = max(dvals) if dvals else 0.0
        rows_meta.append(
            {
                "pair_id": int(row["pair_id"]),
                "n_disagree_tasks": n_dis,
                "max_delta_prob": max_delta,
                "scenario_tags": tags,
                "_row": row,
                "_labels": labels,
                "_gpt_state": gpt_state,
                "_gem_state": gem_state,
                "_tasks_dis": tasks_dis,
                "_delta_probs": delta_probs,
            }
        )

    meta_df = pd.DataFrame(
        [
            {
                "pair_id": r["pair_id"],
                "n_disagree_tasks": r["n_disagree_tasks"],
                "max_delta_prob": r["max_delta_prob"],
                "scenario_tags": r["scenario_tags"],
            }
            for r in rows_meta
        ]
    )
    row_by_id = {int(r["pair_id"]): r for r in rows_meta}

    picked_meta = _pick_stratified(meta_df, seed=args.seed, n_total=args.n_scenarios)
    picked_ids = picked_meta["pair_id"].tolist()

    jsonl_path = out_dir / "multitask_expert_scenarios.jsonl"
    payloads: List[Dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for i, pid in enumerate(picked_ids, start=1):
            r = row_by_id[int(pid)]
            row = r["_row"]
            payload = _build_payload(
                row,
                i,
                r["_labels"],
                r["_gpt_state"],
                r["_gem_state"],
                r["n_disagree_tasks"],
                r["_tasks_dis"],
                r["_delta_probs"],
                r["scenario_tags"],
            )
            payloads.append(payload)
            jf.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    # Index CSV
    index_rows = []
    for p in payloads:
        index_rows.append(
            {
                "scenario_id": p["scenario_id"],
                "pair_id": p["pair_id"],
                "tags": "|".join(p["scenario_tags"]),
                "n_disagree": p["cross_model_disagreement"]["n_tasks_disagree"],
                "gpt_wrong_n": len(p["predictions"]["gpt4o_mini"]["tasks_wrong_vs_gold"]),
                "gemma_wrong_n": len(p["predictions"]["gemma4"]["tasks_wrong_vs_gold"]),
            }
        )
    pd.DataFrame(index_rows).to_csv(out_dir / "multitask_expert_scenarios_index.csv", index=False)

    # Markdown
    md_lines = [
        "# Multitask expert scenarios (GPT-4o-mini vs Gemma4, prompt v3)",
        "",
        f"- **Count:** {len(payloads)} scenarios",
        "- **Primary:** GPT-4o-mini multitask zero-shot (`llm_zero_shot_results.csv`)",
        "- **Comparison:** Gemma4 multitask zero-shot (`responses.jsonl`)",
        "- **Machine-readable:** `multitask_expert_scenarios.jsonl`",
        "",
    ]
    for p in payloads:
        md_lines.extend(
            [
                f"## {p['scenario_id']} — pair_id {p['pair_id']}",
                "",
                f"- **Tags:** {', '.join(p['scenario_tags'])}",
                f"- **Disagreeing tasks ({p['cross_model_disagreement']['n_tasks_disagree']}):** "
                f"{', '.join(p['cross_model_disagreement']['tasks']) or '—'}",
                f"- **Ground truth (task → label):** `{p['ground_truth_labels']}`",
                "",
                "### GPT-4o-mini",
                "",
                f"- Parser: `{p['predictions']['gpt4o_mini']['parser_status']}` "
                f"(salvage={p['predictions']['gpt4o_mini']['salvage_used']}, "
                f"n_tasks_salvaged={p['predictions']['gpt4o_mini']['n_tasks_salvaged']})",
                f"- Wrong vs gold: {p['predictions']['gpt4o_mini']['tasks_wrong_vs_gold']}",
                f"- Per-task: `{p['predictions']['gpt4o_mini']['per_task']}`",
                "",
                "**Reasoning:**",
                "",
                "```",
                (p["predictions"]["gpt4o_mini"]["reasoning"] or "")[:4000],
                "```",
                "",
                "### Gemma4",
                "",
                f"- Parser: `{p['predictions']['gemma4']['parser_status']}` "
                f"(salvage={p['predictions']['gemma4']['salvage_used']}, "
                f"n_tasks_salvaged={p['predictions']['gemma4']['n_tasks_salvaged']})",
                f"- Wrong vs gold: {p['predictions']['gemma4']['tasks_wrong_vs_gold']}",
                f"- Per-task: `{p['predictions']['gemma4']['per_task']}`",
                "",
                "**Reasoning:**",
                "",
                "```",
                (p["predictions"]["gemma4"]["reasoning"] or "")[:4000],
                "```",
                "",
                "### Clinical summary",
                "",
                p["clinical_summary"],
                "",
                "### Current-visit narrative",
                "",
                "```",
                p["narrative_excerpt_phi_safe"],
                "```",
                "",
                "### Raw data example (exact source row excerpt)",
                "",
                "```json",
                json.dumps(p["raw_data_example"], ensure_ascii=False, indent=2, default=str),
                "```",
                "",
                "### Clinician prompts",
                "",
            ]
        )
        for q in p["clinician_questions"]:
            md_lines.append(f"- {q}")
        md_lines.append("")

    (out_dir / "multitask_expert_scenarios_readable.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Simple HTML
    body = ["<html><head><meta charset='utf-8'><title>Multitask expert scenarios</title>",
            "<style>body{font-family:system-ui,Segoe UI,sans-serif;max-width:1200px;margin:24px;} "
            "pre{white-space:pre-wrap;background:#f6f8fa;padding:12px;border-radius:8px;} "
            "h2{border-top:1px solid #ddd;padding-top:16px;}</style></head><body>",
            "<h1>Multitask expert scenarios</h1>",
            f"<p>{len(payloads)} scenarios — GPT-4o-mini vs Gemma4 (prompt v3 zero-shot).</p>"]
    for p in payloads:
        body.append(f"<h2>{html.escape(p['scenario_id'])} — pair_id {p['pair_id']}</h2>")
        body.append(f"<p><b>Tags:</b> {html.escape(', '.join(p['scenario_tags']))}</p>")
        body.append("<h3>GPT-4o-mini reasoning</h3><pre>")
        body.append(html.escape((p["predictions"]["gpt4o_mini"]["reasoning"] or "")[:8000]))
        body.append("</pre><h3>Gemma4 reasoning</h3><pre>")
        body.append(html.escape((p["predictions"]["gemma4"]["reasoning"] or "")[:8000]))
        body.append("</pre><h3>Clinical summary</h3><p>")
        body.append(html.escape(p["clinical_summary"]))
        body.append("</p><h3>Current-visit narrative</h3><pre>")
        body.append(html.escape(p["narrative_excerpt_phi_safe"]))
        body.append("</pre><h3>Raw data example</h3><pre>")
        body.append(html.escape(json.dumps(p["raw_data_example"], ensure_ascii=False, indent=2, default=str)))
        body.append("</pre>")
    body.append("</body></html>")
    (out_dir / "multitask_expert_scenarios_readable.html").write_text("\n".join(body), encoding="utf-8")

    # Manifest
    tag_counts: Dict[str, int] = {}
    for p in payloads:
        for t in p["scenario_tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    manifest = {
        "n_scenarios": len(payloads),
        "seed": args.seed,
        "gpt_csv": str(gpt_csv),
        "gemma_jsonl": str(gemma_jsonl),
        "test_csv": str(test_csv),
        "tag_counts_in_selection": tag_counts,
        "outputs": {
            "jsonl": str(out_dir / "multitask_expert_scenarios.jsonl"),
            "markdown": str(out_dir / "multitask_expert_scenarios_readable.md"),
            "html": str(out_dir / "multitask_expert_scenarios_readable.html"),
            "index_csv": str(out_dir / "multitask_expert_scenarios_index.csv"),
        },
    }
    (out_dir / "multitask_expert_scenarios_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(payloads)} scenarios -> {jsonl_path}")


if __name__ == "__main__":
    main()
