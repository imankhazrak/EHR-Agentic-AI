#!/usr/bin/env python3
"""Score Unsloth multitask test JSONL (per-label classification metrics).

Reads lines from ``scripts/test_unsloth_router.py`` output (``prediction_text``,
``gold_output``) and reports per-task and pooled metrics using ``src.evaluation.metrics``.

Three prediction parsers (``--parse-mode``):

- **strict** — ``parse_multitask_output`` (nested ``prediction`` + ``probability``).
- **salvage** — ``parse_multitask_output_with_meta`` (regex salvage per task).
- **lenient_flat** — top-level JSON with each task key = ``"Yes"`` / ``"No"`` string
  (or nested dict with valid prediction/probability).

Run from repo root (no GPU required)::

    python scripts/score_unsloth_multitask_jsonl.py \\
      outputs/.../test_predictions_2nd_try_fixed.jsonl

    python scripts/score_unsloth_multitask_jsonl.py predictions.jsonl \\
      --parse-mode salvage --markdown-out outputs/.../metrics.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.metrics import compute_metrics
from src.llm.output_parser import (
    MULTITASK_JSON_TASK_KEYS,
    MULTITASK_TASK_KEY_TO_PREFIX,
    _parse_top_level_json_object,
    parse_multitask_output,
    parse_multitask_output_with_meta,
)

ParseMode = Literal["strict", "salvage", "lenient_flat"]


def _matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MCC without requiring sklearn (unsloth_env may lack scikit-learn)."""
    try:
        from sklearn.metrics import matthews_corrcoef as sklearn_mcc

        return float(sklearn_mcc(y_true, y_pred))
    except ImportError:
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred, dtype=int)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom == 0:
            return 0.0
        return float((tp * tn - fp * fn) / denom)

DEFAULT_JSONL = (
    _ROOT
    / "outputs"
    / "unsloth_mt_lora_natural_dist_job47427323_20260513_021406"
    / "test_predictions_2nd_try_fixed.jsonl"
)


def parse_flat_yesno_multitask(text: str) -> Optional[Dict[str, Any]]:
    """Lenient parser: all seven task keys present as Yes/No (string or nested block)."""
    data = _parse_top_level_json_object(text)
    if data is None:
        return None
    out: Dict[str, Any] = {}
    for tk in MULTITASK_JSON_TASK_KEYS:
        if tk not in data:
            return None
        block = data[tk]
        prefix = MULTITASK_TASK_KEY_TO_PREFIX[tk]
        if isinstance(block, dict):
            pred = block.get("prediction")
            if not isinstance(pred, str) or pred.strip().lower() not in ("yes", "no"):
                return None
            out[f"{prefix}_pred"] = 1 if pred.strip().lower() == "yes" else 0
            prob = block.get("probability")
            if isinstance(prob, (int, float)) and not math.isnan(float(prob)):
                p = float(prob)
                if 0.0 <= p <= 1.0:
                    out[f"{prefix}_prob"] = p
        elif isinstance(block, str) and block.strip().lower() in ("yes", "no"):
            out[f"{prefix}_pred"] = 1 if block.strip().lower() == "yes" else 0
        else:
            return None
    return out


def parse_prediction(text: str, mode: ParseMode) -> Optional[Dict[str, Any]]:
    if mode == "strict":
        return parse_multitask_output(text)
    if mode == "salvage":
        meta = parse_multitask_output_with_meta(text)
        if meta.get("n_tasks_salvaged", 0) == 0:
            return None
        return meta
    return parse_flat_yesno_multitask(text)


def task_labels(
    flat: Optional[Dict[str, Any]],
    task_key: str,
) -> Tuple[Optional[int], Optional[float]]:
    if not flat:
        return None, None
    prefix = MULTITASK_TASK_KEY_TO_PREFIX[task_key]
    yp = flat.get(f"{prefix}_pred")
    prob = flat.get(f"{prefix}_prob")
    if yp is None or not isinstance(yp, (int, np.integer)):
        return None, None
    ys: Optional[float] = None
    if prob is not None and isinstance(prob, (int, float)) and not math.isnan(float(prob)):
        ys = float(prob)
    return int(yp), ys


def coverage_counts(rows: List[Dict[str, Any]], mode: ParseMode) -> Dict[str, int]:
    n = len(rows)
    pred_ok = 0
    for r in rows:
        if parse_prediction(str(r.get("prediction_text") or ""), mode) is not None:
            pred_ok += 1
    return {"n_rows": n, "rows_with_usable_prediction": pred_ok}


def evaluate_task(
    rows: List[Dict[str, Any]],
    task_key: str,
    mode: ParseMode,
) -> Dict[str, Any]:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []

    for r in rows:
        gold = parse_multitask_output(str(r.get("gold_output") or ""))
        pred = parse_prediction(str(r.get("prediction_text") or ""), mode)
        yt, _ = task_labels(gold, task_key)
        yp, ys = task_labels(pred, task_key)
        if yt is None or yp is None:
            continue
        y_true.append(yt)
        y_pred.append(yp)
        if ys is not None:
            y_score.append(ys)

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if len(y_true_arr) == 0:
        return {"task": task_key, "n": 0}

    y_score_arr: Optional[np.ndarray] = None
    if len(y_score) == len(y_true_arr) and len(np.unique(y_true_arr)) > 1:
        y_score_arr = np.asarray(y_score, dtype=float)

    m = compute_metrics(y_true_arr, y_pred_arr, y_score=y_score_arr)
    tn, fp, fn, tp = np.array(m["confusion_matrix"]["matrix"]).ravel()
    npv = (tn / (tn + fn) * 100) if (tn + fn) > 0 else 0.0
    m["npv_percent"] = round(float(npv), 2)
    m["mcc"] = round(_matthews_corrcoef(y_true_arr, y_pred_arr), 4)
    m["n"] = int(len(y_true_arr))
    m["gold_yes_percent"] = round(100.0 * float(y_true_arr.sum()) / len(y_true_arr), 2)
    m["task"] = task_key
    del m["confusion_matrix"]
    return m


def macro_mean(per_task: Dict[str, Dict[str, Any]], keys: List[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        vals = [
            float(per_task[tk][k])
            for tk in MULTITASK_JSON_TASK_KEYS
            if per_task[tk].get("n") and per_task[tk].get(k) is not None
        ]
        out[k] = round(sum(vals) / len(vals), 2) if vals else None
    return out


def evaluate_all(rows: List[Dict[str, Any]], mode: ParseMode) -> Dict[str, Any]:
    per_task = {tk: evaluate_task(rows, tk, mode) for tk in MULTITASK_JSON_TASK_KEYS}

    micro_y: List[int] = []
    micro_p: List[int] = []
    micro_s: List[float] = []
    for tk in MULTITASK_JSON_TASK_KEYS:
        for r in rows:
            gold = parse_multitask_output(str(r.get("gold_output") or ""))
            pred = parse_prediction(str(r.get("prediction_text") or ""), mode)
            yt, _ = task_labels(gold, tk)
            yp, ys = task_labels(pred, tk)
            if yt is None or yp is None:
                continue
            micro_y.append(yt)
            micro_p.append(yp)
            if ys is not None:
                micro_s.append(ys)

    micro: Optional[Dict[str, Any]] = None
    if micro_y:
        my = np.asarray(micro_y, dtype=int)
        mp = np.asarray(micro_p, dtype=int)
        ms = np.asarray(micro_s, dtype=float) if len(micro_s) == len(micro_y) else None
        micro = compute_metrics(
            my,
            mp,
            y_score=ms if ms is not None and len(np.unique(my)) > 1 else None,
        )
        tn, fp, fn, tp = np.array(micro["confusion_matrix"]["matrix"]).ravel()
        micro["npv_percent"] = round(float((tn / (tn + fn) * 100) if (tn + fn) > 0 else 0.0), 2)
        micro["mcc"] = round(_matthews_corrcoef(my, mp), 4)
        micro["n_task_rows"] = len(micro_y)
        del micro["confusion_matrix"]

    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
        "npv_percent",
        "mcc",
        "auc",
        "auprc",
    ]
    return {
        "parse_mode": mode,
        "coverage": coverage_counts(rows, mode),
        "per_task": per_task,
        "macro": macro_mean(per_task, metric_keys),
        "micro": micro,
        "metric_keys_percent_scale": metric_keys,
        "notes": (
            "Accuracy, precision, recall, sensitivity, specificity, F1, balanced_accuracy, "
            "NPV, AUC, AUPRC are on a 0–100 scale (see src/evaluation/metrics.py). "
            "MCC is in [-1, 1]. Positive class = Yes = 1."
        ),
    }


def format_markdown(report: Dict[str, Any], source: Path) -> str:
    mode = report["parse_mode"]
    cov = report["coverage"]
    lines = [
        f"# Multitask metrics — `{source.name}`",
        "",
        f"- **Source:** `{source}`",
        f"- **Parse mode:** `{mode}`",
        f"- **Rows:** {cov['n_rows']}",
        f"- **Rows with usable prediction (all tasks for strict/lenient; per-task N varies for salvage):** "
        f"{cov['rows_with_usable_prediction']}",
        "",
        report["notes"],
        "",
        "## Per-task",
        "",
        "| Task | n | Gold%Yes | Acc | PPV | NPV | Sens | Spec | F1 | BalAcc | MCC | AUC | AUPRC | TP | FP | TN | FN |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tk in MULTITASK_JSON_TASK_KEYS:
        m = report["per_task"][tk]
        if not m.get("n"):
            lines.append(f"| `{tk}` | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
            continue

        def cell(k: str) -> str:
            v = m.get(k)
            return "—" if v is None else str(v)

        lines.append(
            f"| `{tk}` | {m['n']} | {m['gold_yes_percent']} | {m['accuracy']} | {m['precision']} | "
            f"{m['npv_percent']} | {m['sensitivity']} | {m['specificity']} | {m['f1']} | "
            f"{m['balanced_accuracy']} | {cell('mcc')} | {cell('auc')} | {cell('auprc')} | "
            f"{m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |"
        )
    micro = report.get("micro")
    if micro:
        lines.extend(
            [
                "",
                f"## Micro-pooled ({micro['n_task_rows']} task-rows)",
                "",
                f"| Acc | PPV | NPV | Sens | Spec | F1 | BalAcc | MCC | AUC | AUPRC | TP | FP | TN | FN |",
                f"|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {micro['accuracy']} | {micro['precision']} | {micro['npv_percent']} | "
                f"{micro['sensitivity']} | {micro['specificity']} | {micro['f1']} | "
                f"{micro['balanced_accuracy']} | {micro.get('mcc', '—')} | {micro.get('auc', '—')} | "
                f"{micro.get('auprc', '—')} | {micro['tp']} | {micro['fp']} | {micro['tn']} | {micro['fn']} |",
            ]
        )
    macro = report["macro"]
    lines.extend(
        [
            "",
            "## Macro mean (7 tasks)",
            "",
            "| " + " | ".join(macro.keys()) + " |",
            "| " + " | ".join(str(macro[k]) if macro[k] is not None else "—" for k in macro) + " |",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score Unsloth multitask test JSONL.")
    p.add_argument(
        "predictions_jsonl",
        type=Path,
        nargs="?",
        default=DEFAULT_JSONL,
        help="JSONL from test_unsloth_router.py",
    )
    p.add_argument(
        "--parse-mode",
        choices=("strict", "salvage", "lenient_flat"),
        default="salvage",
        help="How to parse prediction_text (default: salvage).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write metrics JSON (default: <jsonl_stem>_metrics_<mode>.json beside input).",
    )
    p.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Optional markdown report path.",
    )
    p.add_argument(
        "--all-modes",
        action="store_true",
        help="Run strict, salvage, and lenient_flat; write three JSON outputs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = args.predictions_jsonl.resolve()
    if not path.is_file():
        raise SystemExit(f"Predictions JSONL not found: {path}")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    modes: List[ParseMode]
    if args.all_modes:
        modes = ["strict", "salvage", "lenient_flat"]
    else:
        modes = [args.parse_mode]  # type: ignore[list-item]

    for mode in modes:
        report = evaluate_all(rows, mode)
        report["source"] = str(path)

        json_out = args.json_out
        if json_out is None or args.all_modes:
            json_out = path.with_name(f"{path.stem}_metrics_{mode}.json")
        else:
            json_out = json_out.resolve()

        json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {json_out}")

        md_out = args.markdown_out
        if md_out is not None:
            if args.all_modes:
                md_out = path.with_name(f"{path.stem}_metrics_{mode}.md")
            else:
                md_out = md_out.resolve()
            md_out.write_text(format_markdown(report, path), encoding="utf-8")
            print(f"Wrote {md_out}")

        cov = report["coverage"]
        n_rows = int(cov["n_rows"])
        n_ok = int(cov["rows_with_usable_prediction"])
        pct = (100.0 * n_ok / n_rows) if n_rows else 0.0
        print(
            f"[{mode}] rows={n_rows} usable_pred_rows={n_ok} "
            f"strict_parse_coverage_pct={pct:.2f}%"
        )
        for tk in MULTITASK_JSON_TASK_KEYS:
            n = report["per_task"][tk].get("n", 0)
            if n:
                acc = report["per_task"][tk].get("accuracy")
                f1 = report["per_task"][tk].get("f1")
                print(f"  {tk}: n={n} acc={acc}% f1={f1}%")


if __name__ == "__main__":
    main()
