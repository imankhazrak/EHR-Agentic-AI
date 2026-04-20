"""Build clinician-ready case studies from existing LLM predictions (read-only inputs).

Merges test JSONL with prediction CSVs on ``pair_id``, selects balanced TP/TN/FP/FN
examples by probability rank, and writes JSON / CSV / Markdown under ``data/outputs/case_studies/``.

Primary default: Gemma 4 Prompt V2 EHR-CoAgent (``llm_coagent_results.csv``).
Optional comparison: GPT-4o-mini Prompt V2 few-shot.

Usage::
    python -m src.scripts.generate_clinician_case_studies
    python -m src.scripts.generate_clinician_case_studies --per-quadrant 3
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.code_mappings import CodeMapper

# ---------------------------------------------------------------------------
# Defaults (repo-relative from project root)
# ---------------------------------------------------------------------------
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEST_JSONL = _DEFAULT_ROOT / "data/mimic III/data_ready_for_finetuning_next_timing_and_lipid/test.jsonl"
_DEFAULT_PRIMARY_CSV = _DEFAULT_ROOT / "data/outputs/mimiciii_llm_promptv2_gemma/llm_coagent_results.csv"
_DEFAULT_COMPARE_CSV = _DEFAULT_ROOT / "data/outputs/mimiciii_llm_gpt4o_mini_promptv2/llm_few_shot_results.csv"
_DEFAULT_OUT_DIR = _DEFAULT_ROOT / "data/outputs/case_studies"
_DEFAULT_INTERIM_DIR = _DEFAULT_ROOT / "data/interim/mimiciii"

PRIMARY_MODEL_LABEL = "Gemma4_PromptV2_EHR-CoAgent"
COMPARE_MODEL_LABEL = "GPT4o-mini_PromptV2_Few-Shot"

_DEFAULT_NARRATIVE_MAX_CHARS = 800

# ICD-9 base codes (string startswith checks after normalizing length)
_DIABETES_PREFIXES = ("250",)
_OBESITY_CODES_PREFIX = ("2780", "27800", "27801", "27802")
_DYSLIPIDEMIA_PREFIXES = ("272",)
_HTN_PREFIXES = ("401", "402", "403", "404", "405")
_ISCHEMIC_HEART_PREFIXES = ("410", "411", "412", "413", "414")
_CHF_PREFIXES = ("428",)
_ATHERO_PREFIXES = ("440",)


def _split_icd_codes(field: Any) -> List[str]:
    if field is None or (isinstance(field, float) and math.isnan(field)):
        return []
    s = str(field).strip()
    if not s or s.lower() == "nan":
        return []
    parts = re.split(r"[;,\s]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def _code_prefix5(code: str) -> str:
    """First 3–5 chars for numeric ICD-9 codes (handles 99662 style)."""
    return "".join(c for c in code if c.isdigit())[:5]


def _flag_comorbidities(codes: Sequence[str]) -> Dict[str, bool]:
    """Rule-based flags from ICD-9 diagnosis codes (coarse, for case narratives)."""
    diabetes = obesity = dyslipidemia = htn = ihd = chf = athero = False
    for raw in codes:
        c = _code_prefix5(raw)
        if not c:
            continue
        if c.startswith(_DIABETES_PREFIXES):
            diabetes = True
        if any(c.startswith(p) for p in _OBESITY_CODES_PREFIX):
            obesity = True
        if any(c.startswith(p) for p in _DYSLIPIDEMIA_PREFIXES):
            dyslipidemia = True
        if any(c.startswith(p) for p in _HTN_PREFIXES):
            htn = True
        if any(c.startswith(p) for p in _ISCHEMIC_HEART_PREFIXES):
            ihd = True
        if any(c.startswith(p) for p in _CHF_PREFIXES):
            chf = True
        if any(c.startswith(p) for p in _ATHERO_PREFIXES):
            athero = True
    cvd = ihd or chf or athero
    metabolic_cluster = sum([diabetes, obesity, dyslipidemia, htn]) >= 3
    return {
        "diabetes": diabetes,
        "obesity": obesity,
        "dyslipidemia_icd": dyslipidemia,
        "hypertension": htn,
        "ischemic_heart": ihd,
        "chf": chf,
        "atherosclerosis": athero,
        "cardiovascular_burden": cvd,
        "metabolic_syndrome_pattern": metabolic_cluster,
    }


def _statin_signal(meds: Any) -> bool:
    if meds is None or (isinstance(meds, float) and math.isnan(meds)):
        return False
    m = str(meds).lower()
    if "statin" in m:
        return True
    for name in ("atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin", "pitavastatin", "fluvastatin"):
        if name in m:
            return True
    return False


def _clean_narrative(text: Any, max_chars: int = _DEFAULT_NARRATIVE_MAX_CHARS) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = " ".join(str(text).split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 2].rstrip() + " …"


def _age_decade(dob: Any, admittime: Any) -> Optional[str]:
    """Approximate age decade from year difference (MIMIC-shifted dates)."""
    try:
        if dob is None or admittime is None:
            return None
        ds = str(dob).strip()[:10]
        ads = str(admittime).strip()[:10]
        by = int(ds[:4])
        ay = int(ads[:4])
        age = max(0, min(120, ay - by))
        decade = (age // 10) * 10
        return f"{decade}-{decade + 9}"
    except (ValueError, TypeError):
        return None


def _case_type_row(pred: int, label: int) -> str:
    if pred == 1 and label == 1:
        return "TP"
    if pred == 0 and label == 0:
        return "TN"
    if pred == 1 and label == 0:
        return "FP"
    if pred == 0 and label == 1:
        return "FN"
    return "UNKNOWN"


def _selection_tags(prob: float) -> List[str]:
    tags: List[str] = []
    if prob > 0.8 or prob < 0.2:
        tags.append("high_confidence")
    if 0.4 <= prob <= 0.6:
        tags.append("borderline")
    return tags


def _unique_preserve_order(indices: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _pick_positions(n: int, per_quadrant: int) -> List[int]:
    """Positions into sorted-by-ascending-probability list [0, n)."""
    if n == 0:
        return []
    if per_quadrant == 2:
        if n == 1:
            return [0]
        return _unique_preserve_order([0, n - 1])
    if per_quadrant == 3:
        if n == 1:
            return [0, 0, 0]
        if n == 2:
            return [0, 0, 1]
        return [0, n // 2, n - 1]
    raise ValueError("per_quadrant must be 2 or 3")


def _maybe_swap_mid_for_borderline(
    sub: pd.DataFrame, prob_col: str, ordered_idx: List[int]
) -> List[int]:
    """If middle probability not in [0.4,0.6] but such rows exist, swap mid for closest to 0.5."""
    if len(ordered_idx) < 3:
        return ordered_idx
    lo, mid, hi = ordered_idx[0], ordered_idx[1], ordered_idx[2]
    mid_p = float(sub.loc[mid, prob_col])
    if 0.4 <= mid_p <= 0.6:
        return ordered_idx
    band = sub[(sub[prob_col] >= 0.4) & (sub[prob_col] <= 0.6)]
    if band.empty:
        return ordered_idx
    closest_row = (band[prob_col] - 0.5).abs().idxmin()
    if closest_row in (lo, hi):
        return ordered_idx
    return [lo, closest_row, hi]


def _model_explanation(case_type: str, flags: Dict[str, bool], statin: bool) -> str:
    parts: List[str] = []
    if case_type == "TP":
        base = "Prediction aligns with lipid-related signals in the record"
        if flags.get("dyslipidemia_icd"):
            parts.append(base + ", including dyslipidemia-related diagnosis codes.")
        elif statin:
            parts.append(base + ", including statin therapy on the medication list.")
        elif flags.get("metabolic_syndrome_pattern") or flags.get("diabetes"):
            parts.append(
                "Prediction likely driven by metabolic risk factors (e.g., diabetes or clustered cardiometabolic diagnoses)."
            )
        else:
            parts.append(
                "Prediction likely driven by clinical cues in the narrative or code list suggestive of lipid disorder risk."
            )
    elif case_type == "TN":
        parts.append("No strong indicators of lipid disorder in structured codes relative to the task label (negative next-visit lipid disorder).")
    elif case_type == "FP":
        parts.append(
            "Model may rely on indirect signals such as statin therapy or cardiovascular medications without a documented dyslipidemia label at the next visit."
            if statin
            else "Model may be over-reading cardiometabolic context or narrative phrasing as lipid-disorder risk."
        )
    elif case_type == "FN":
        if not flags.get("dyslipidemia_icd") and not statin:
            parts.append(
                "Possible miss due to sparse explicit lipid-related codes or therapies in the current-admission snapshot."
            )
        else:
            parts.append(
                "Despite some lipid-adjacent signals, the model underweighted evidence; calibration or label timing ambiguity may contribute."
            )
    else:
        parts.append("See structured evidence and narrative for manual review.")
    return " ".join(parts)


def _clinical_summary(
    gender: Any,
    age_dec: Optional[str],
    flags: Dict[str, bool],
    major_codes_preview: str,
    narrative_short: str,
) -> str:
    g = str(gender).strip() if gender is not None and str(gender) != "nan" else "Unknown"
    age_phrase = f"Approximate age band (from shifted dates): {age_dec} years." if age_dec else ""
    flag_bits = []
    if flags["diabetes"]:
        flag_bits.append("diabetes-related codes")
    if flags["obesity"]:
        flag_bits.append("obesity-related codes")
    if flags["dyslipidemia_icd"]:
        flag_bits.append("dyslipidemia-related codes")
    if flags["cardiovascular_burden"]:
        flag_bits.append("cardiovascular diagnoses")
    if flags["metabolic_syndrome_pattern"]:
        flag_bits.append("a cardiometabolic risk pattern (rule-based)")
    flags_str = ", ".join(flag_bits) if flag_bits else "no major coded comorbidity hits under simple rules"
    nar_hook = narrative_short[:320] + (" …" if len(narrative_short) > 320 else "")
    parts = [
        f"De-identified cohort record (analysis id only). Sex: {g}. {age_phrase}".strip(),
        f"Structured signals (ICD-9 rules): {flags_str}. Example codes: {major_codes_preview}.",
        f"Current-visit narrative (truncated): {nar_hook}",
    ]
    return " ".join(p for p in parts if p)


def _key_evidence(codes: Sequence[str], flags: Dict[str, bool], statin: bool) -> str:
    code_sample = "; ".join(codes[:12])
    if len(codes) > 12:
        code_sample += " …"
    meds_note = "Statin or lipid-lowering therapy mentioned in medication list." if statin else "No statin keyword hit in medication list (keyword rule)."
    flag_line = ", ".join(k for k, v in flags.items() if v and k in ("diabetes", "obesity", "dyslipidemia_icd", "cardiovascular_burden", "metabolic_syndrome_pattern"))
    return f"Diagnosis codes (sample): {code_sample}. Rule flags: [{flag_line or 'none'}]. {meds_note}"


def _format_diagnosis_codes_with_terms(codes: Sequence[str], code_mapper: CodeMapper, max_items: int = 12) -> str:
    """Return a clinician-friendly list like: 4280 (Congestive heart failure, unspecified)."""
    formatted: List[str] = []
    for code in codes[:max_items]:
        term = code_mapper.map_diagnosis(code)
        formatted.append(f"{code} ({term})")
    out = "; ".join(formatted)
    if len(codes) > max_items:
        out += " …"
    return out


CLINICIAN_QUESTIONS = [
    "Does this prediction seem clinically reasonable?",
    "What evidence supports or contradicts this prediction?",
    "Is this case ambiguous?",
    "Would you expect a lipid disorder diagnosis at next visit?",
]


def select_case_rows(df: pd.DataFrame, prob_col: str, per_quadrant: int) -> pd.DataFrame:
    """Pick balanced TP/TN/FP/FN rows; deterministic ordering by probability within each."""
    df = df.copy()
    p = df["pred_binary"].astype(int)
    y = df["label_lipid_disorder"].astype(int)
    df["_case_type"] = np.select(
        [(p == 1) & (y == 1), (p == 0) & (y == 0), (p == 1) & (y == 0), (p == 0) & (y == 1)],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )
    picked: List[Any] = []
    for ct in ("TP", "TN", "FP", "FN"):
        sub = df[df["_case_type"] == ct]
        if sub.empty:
            raise ValueError(f"No rows for case type {ct}")
        sub_sorted = sub.sort_values(prob_col, ascending=True, kind="mergesort")
        idx_list = sub_sorted.index.tolist()
        n = len(idx_list)
        if n < per_quadrant:
            raise ValueError(f"Not enough rows for {ct}: have {n}, need {per_quadrant}")
        pos = _pick_positions(n, per_quadrant)
        chosen_idx = [idx_list[p] for p in pos]
        if per_quadrant == 3:
            chosen_idx = _maybe_swap_mid_for_borderline(sub_sorted, prob_col, chosen_idx)
        picked.extend(chosen_idx)
    out = df.loc[picked].copy()
    out.drop(columns=["_case_type"], errors="ignore", inplace=True)
    return out.loc[picked]


def build_case_records(
    picked: pd.DataFrame,
    prob_col: str,
    compare_df: Optional[pd.DataFrame],
    narrative_max_chars: int,
    code_mapper: CodeMapper,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    compare_map: Dict[int, Dict[str, Any]] = {}
    if compare_df is not None:
        for _, r in compare_df.iterrows():
            compare_map[int(r["pair_id"])] = {
                "gpt4o_mini_v2_few_shot_pred": int(r["pred_binary"])
                if pd.notna(r["pred_binary"])
                else None,
                "gpt4o_mini_v2_few_shot_prob": float(r["parsed_probability"])
                if pd.notna(r["parsed_probability"])
                else None,
            }

    for _, row in picked.iterrows():
        pair_id = int(row["pair_id"])
        pred = int(row["pred_binary"])
        lab = int(row["label_lipid_disorder"])
        prob = float(row[prob_col])
        ct = _case_type_row(pred, lab)
        codes = _split_icd_codes(row.get("diagnoses_codes_current"))
        flags = _flag_comorbidities(codes)
        statin = _statin_signal(row.get("medications_current"))
        age_dec = _age_decade(row.get("DOB"), row.get("admittime_current"))
        narr = _clean_narrative(row.get("narrative_current"), max_chars=narrative_max_chars)
        major_preview = "; ".join(codes[:8]) + (" …" if len(codes) > 8 else "")
        diagnoses_with_terms = _format_diagnosis_codes_with_terms(codes, code_mapper, max_items=12)

        rec: Dict[str, Any] = {
            "case_id": str(pair_id),
            "true_label": lab,
            "predicted_label": pred,
            "probability": round(prob, 4),
            "case_type": ct,
            "primary_model": PRIMARY_MODEL_LABEL,
            "clinical_summary": _clinical_summary(row.get("GENDER"), age_dec, flags, major_preview, narr),
            "key_evidence": (
                f"Diagnosis codes with medical terms: {diagnoses_with_terms}. "
                + _key_evidence(codes, flags, statin)
            ),
            "diagnosis_codes_with_terms": diagnoses_with_terms,
            "model_explanation": _model_explanation(ct, flags, statin),
            "clinician_questions": list(CLINICIAN_QUESTIONS),
            "selection_tags": _selection_tags(prob),
            "narrative_excerpt_phi_safe": narr,
        }
        if pair_id in compare_map:
            rec["comparison_model"] = COMPARE_MODEL_LABEL
            rec.update(compare_map[pair_id])
        records.append(rec)
    return records


def _git_sha(cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def run_qc(records: List[Dict[str, Any]], per_quadrant: int) -> None:
    by = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for r in records:
        by[r["case_type"]] += 1
    for k in ("TP", "TN", "FP", "FN"):
        assert by[k] == per_quadrant, f"Expected {per_quadrant} cases for {k}, got {by[k]}"
    for r in records:
        p = r["probability"]
        assert 0.0 <= p <= 1.0, p
        assert r["clinical_summary"]
        assert r["key_evidence"]
        assert r["model_explanation"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate clinician case studies from cached LLM outputs.")
    ap.add_argument("--test-jsonl", type=Path, default=_DEFAULT_TEST_JSONL)
    ap.add_argument("--primary-csv", type=Path, default=_DEFAULT_PRIMARY_CSV)
    ap.add_argument("--compare-csv", type=Path, default=_DEFAULT_COMPARE_CSV, help="Optional; set empty to skip")
    ap.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT_DIR)
    ap.add_argument("--interim-dir", type=Path, default=_DEFAULT_INTERIM_DIR)
    ap.add_argument("--per-quadrant", type=int, default=2, choices=(2, 3))
    ap.add_argument("--narrative-chars", type=int, default=_DEFAULT_NARRATIVE_MAX_CHARS)
    ap.add_argument("--no-compare", action="store_true", help="Skip GPT-4o-mini few-shot comparison merge")
    args = ap.parse_args()

    narrative_max = max(200, args.narrative_chars)
    code_mapper = CodeMapper(str(args.interim_dir))

    test_df = pd.read_json(args.test_jsonl, lines=True)
    pred_df = pd.read_csv(args.primary_csv)

    # Inner merge on pair_id
    merged = test_df.merge(
        pred_df,
        on="pair_id",
        how="inner",
        suffixes=("", "_pred"),
    )
    if len(merged) != len(test_df) or len(merged) != len(pred_df):
        raise ValueError(
            f"Merge size mismatch: test={len(test_df)} pred={len(pred_df)} merged={len(merged)}"
        )

    prob_col = "parsed_probability"
    if prob_col not in merged.columns:
        raise ValueError(f"Missing {prob_col} on merged frame")
    merged["_probability"] = merged[prob_col].astype(float)

    compare_df: Optional[pd.DataFrame] = None
    if (
        not args.no_compare
        and args.compare_csv
        and Path(args.compare_csv).exists()
    ):
        compare_df = pd.read_csv(args.compare_csv)
        if len(compare_df) != len(pred_df):
            raise ValueError("Comparison CSV row count must match primary predictions for QC.")

    picked_df = select_case_rows(merged, "_probability", args.per_quadrant)
    records = build_case_records(picked_df, "_probability", compare_df, narrative_max, code_mapper)
    run_qc(records, args.per_quadrant)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "case_studies.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Flat CSV
    flat_rows = []
    for r in records:
        flat = {
            "case_id": r["case_id"],
            "true_label": r["true_label"],
            "predicted_label": r["predicted_label"],
            "probability": r["probability"],
            "case_type": r["case_type"],
            "clinical_summary": r["clinical_summary"],
            "key_evidence": r["key_evidence"],
            "diagnosis_codes_with_terms": r.get("diagnosis_codes_with_terms", ""),
            "model_explanation": r["model_explanation"],
            "clinician_questions": " | ".join(r["clinician_questions"]),
            "primary_model": r.get("primary_model", ""),
            "comparison_model": r.get("comparison_model", ""),
            "gpt4o_mini_v2_few_shot_pred": r.get("gpt4o_mini_v2_few_shot_pred", ""),
            "gpt4o_mini_v2_few_shot_prob": r.get("gpt4o_mini_v2_few_shot_prob", ""),
        }
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(out_dir / "case_studies.csv", index=False)

    md_lines = [
        "# Clinician case studies (generated from cached LLM outputs)",
        "",
        f"Primary model: {PRIMARY_MODEL_LABEL}. Comparison: {COMPARE_MODEL_LABEL} (if columns present).",
        f"Questions in CSV use ` | ` between items.",
        "",
    ]
    for r in records:
        md_lines.append(f"## Case {r['case_id']} ({r['case_type']})")
        md_lines.append("")
        md_lines.append(f"- **True label:** {r['true_label']} | **Predicted:** {r['predicted_label']} | **P(yes):** {r['probability']}")
        if "gpt4o_mini_v2_few_shot_pred" in r:
            md_lines.append(
                f"- **Comparison ({COMPARE_MODEL_LABEL}):** pred={r.get('gpt4o_mini_v2_few_shot_pred')} prob={r.get('gpt4o_mini_v2_few_shot_prob')}"
            )
        md_lines.extend(
            [
                f"- **Clinical summary:** {r['clinical_summary']}",
                f"- **Key evidence:** {r['key_evidence']}",
                f"- **Diagnosis codes with medical terms:** {r.get('diagnosis_codes_with_terms', '')}",
                f"- **Model explanation (rule-based):** {r['model_explanation']}",
                "- **Clinician questions:**",
            ]
        )
        for q in r["clinician_questions"]:
            md_lines.append(f"  - {q}")
        md_lines.append("")
    (out_dir / "case_studies_readable.md").write_text("\n".join(md_lines), encoding="utf-8")

    manifest = {
        "test_jsonl": str(args.test_jsonl.resolve()),
        "primary_csv": str(args.primary_csv.resolve()),
        "compare_csv": None
        if args.no_compare
        else (str(args.compare_csv.resolve()) if args.compare_csv else None),
        "comparison_enabled": compare_df is not None,
        "icd_term_expansion_enabled": True,
        "interim_dir": str(args.interim_dir.resolve()),
        "per_quadrant": args.per_quadrant,
        "n_cases": len(records),
        "pandas_version": pd.__version__,
        "git_head": _git_sha(_DEFAULT_ROOT),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(records)} cases to {out_dir}")


if __name__ == "__main__":
    main()
