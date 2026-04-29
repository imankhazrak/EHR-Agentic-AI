from pathlib import Path
import html
import pandas as pd


def sample_codes(text: str, n: int = 10) -> str:
    if not isinstance(text, str) or not text.strip():
        return "None recorded"
    items = [x.strip() for x in text.split(";") if x.strip()]
    if not items:
        return "None recorded"
    out = "; ".join(items[:n])
    if len(items) > n:
        out += " ..."
    return out


def short_narrative(text: str, max_len: int = 420) -> str:
    if not isinstance(text, str) or not text.strip():
        return "No narrative text available."
    txt = text.replace("\n", " ").strip()
    return txt if len(txt) <= max_len else txt[:max_len] + " ..."


def main() -> None:
    base = Path("/users/PCS0229/imankhazrak/EHR-Agentic-AI")
    pred_path = base / "data/outputs/mimiciii_llm_gpt4o_mini_promptv3_multitask_test/llm_coagent_results.csv"
    test_path = base / "data/processed/mimiciii_multitask/test.csv"
    out_md = base / "docs/multitask_clinician_case_studies_coagent_balanced.md"
    out_html = base / "docs/multitask_clinician_case_studies_coagent_balanced.html"

    pred = pd.read_csv(pred_path)
    test = pd.read_csv(test_path)

    pred = pred[
        [
            "pair_id",
            "lipid_pred",
            "lipid_prob",
            "diabetes_pred",
            "diabetes_prob",
            "hypertension_pred",
            "hypertension_prob",
            "obesity_pred",
            "obesity_prob",
            "cardio_pred",
            "cardio_prob",
            "kidney_pred",
            "kidney_prob",
            "stroke_pred",
            "stroke_prob",
        ]
    ]

    test = test[
        [
            "pair_id",
            "narrative_current",
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
    ]

    df = test.merge(pred, on="pair_id", how="inner")

    tasks = [
        ("lipid_next", "label_lipid_next", "lipid_pred", "lipid_prob", "Lipid Disorder (Next Visit)"),
        ("diabetes_current", "label_diabetes_current", "diabetes_pred", "diabetes_prob", "Diabetes (Current Visit)"),
        (
            "hypertension_current",
            "label_hypertension_current",
            "hypertension_pred",
            "hypertension_prob",
            "Hypertension (Current Visit)",
        ),
        ("obesity_current", "label_obesity_current", "obesity_pred", "obesity_prob", "Obesity (Current Visit)"),
        (
            "cardio_next",
            "label_cardio_next",
            "cardio_pred",
            "cardio_prob",
            "Cardiovascular Condition (Next Visit)",
        ),
        ("kidney_next", "label_kidney_next", "kidney_pred", "kidney_prob", "Kidney Condition (Next Visit)"),
        ("stroke_next", "label_stroke_next", "stroke_pred", "stroke_prob", "Stroke (Next Visit)"),
    ]

    outcome_map = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}
    outcome_order = ["TP", "TN", "FP", "FN"]

    selected = []
    confusion = []

    for task, label_col, pred_col, prob_col, title in tasks:
        t = df[
            [
                "pair_id",
                "narrative_current",
                "diagnoses_codes_current",
                "procedures_codes_current",
                "medications_current",
                label_col,
                pred_col,
                prob_col,
            ]
        ].dropna(subset=[pred_col, prob_col]).copy()
        t[label_col] = t[label_col].astype(int)
        t[pred_col] = t[pred_col].astype(int)
        t["outcome"] = t.apply(lambda r: outcome_map[(int(r[label_col]), int(r[pred_col]))], axis=1)
        t["confidence"] = (t[prob_col] - 0.5).abs()

        vc = t["outcome"].value_counts().to_dict()
        confusion.append((task, vc.get("TP", 0), vc.get("TN", 0), vc.get("FP", 0), vc.get("FN", 0), len(t)))

        for outcome in outcome_order:
            cands = t[t["outcome"] == outcome].sort_values(["confidence", "pair_id"], ascending=[False, True])
            if cands.empty:
                continue
            row = cands.iloc[0]
            selected.append(
                (
                    task,
                    title,
                    outcome,
                    int(row["pair_id"]),
                    int(row[label_col]),
                    int(row[pred_col]),
                    float(row[prob_col]),
                    short_narrative(row["narrative_current"]),
                    sample_codes(row["diagnoses_codes_current"]),
                    sample_codes(row["procedures_codes_current"]),
                    sample_codes(row["medications_current"]),
                )
            )

    # Markdown report
    md = [
        "# Multitask Clinician Case Studies (EHR-CoAgent, Balanced)",
        "",
        "This report is designed for clinician review with balanced TP/TN/FP/FN examples per task.",
        "",
        "- **Model:** `gpt-4o-mini-2024-07-18` (`coagent`)",
        "- **Selection strategy:** highest-confidence TP/TN/FP/FN per task (confidence = |probability - 0.5|)",
        "",
        "## Task Confusion Overview (Selection Pool)",
        "",
        "| Task | TP | TN | FP | FN | Pool Size |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in confusion:
        md.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {c[5]} |")

    for task, _, _, _, title in tasks:
        md.extend(
            [
                "",
                f"## {title} (`{task}`)",
                "",
                "- **Clinician prompts:**",
                "  - Is the prediction clinically reasonable based on available context?",
                "  - Which evidence supports or contradicts this prediction?",
                "  - Would this output change management or follow-up planning?",
            ]
        )
        for outcome in outcome_order:
            rows = [x for x in selected if x[0] == task and x[2] == outcome]
            if not rows:
                continue
            x = rows[0]
            md.extend(
                [
                    "",
                    f"### Case {x[3]} ({x[2]})",
                    "",
                    f"- **True label:** {x[4]} | **Predicted:** {x[5]} | **Probability:** {x[6]:.2f}",
                    f"- **Clinical summary:** {x[7]}",
                    f"- **Key diagnosis codes (sample):** {x[8]}",
                    f"- **Key procedures (sample):** {x[9]}",
                    f"- **Key medications (sample):** {x[10]}",
                ]
            )

    md.extend(
        [
            "",
            "## Cross-Task Clinical Review Checklist",
            "",
            "- Compare whether similar clinical profiles are overcalled or undercalled across related tasks.",
            "- Check for predictions that seem driven by acute severity rather than target-specific evidence.",
            "- For false positives, assess whether risk-factor recognition may still be clinically useful.",
            "- For false negatives, assess whether subtle chronic-disease evidence was likely underweighted.",
            "",
            "## Appendix: Selected Cases in One Table",
            "",
            "| task | outcome | pair_id | true_label | pred_label | probability |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for x in sorted(selected, key=lambda z: (z[0], z[2])):
        md.append(f"| {x[0]} | {x[2]} | {x[3]} | {x[4]} | {x[5]} | {x[6]:.2f} |")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    # HTML report
    html_out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Multitask Clinician Case Studies - CoAgent</title>",
        "<style>body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0}.wrap{width:100vw;max-width:none;padding:16px;box-sizing:border-box}h1{margin:0 0 10px}h2{margin-top:24px;border-top:1px solid #ddd;padding-top:10px}.card{border:1px solid #ddd;border-radius:8px;padding:10px;margin:10px 0;background:#fafafa}table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{border:1px solid #ddd;padding:4px 6px;text-align:right}th:first-child,td:first-child{text-align:left}</style></head><body><main class='wrap'>",
        "<h1>Multitask Clinician Case Studies (EHR-CoAgent, Balanced)</h1>",
        "<p><b>Selection strategy:</b> highest-confidence TP/TN/FP/FN per task.</p>",
        "<h2>Task Confusion Overview (Selection Pool)</h2>",
        "<table><tr><th>Task</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th><th>Pool Size</th></tr>",
    ]
    for c in confusion:
        html_out.append(f"<tr><td>{html.escape(str(c[0]))}</td><td>{c[1]}</td><td>{c[2]}</td><td>{c[3]}</td><td>{c[4]}</td><td>{c[5]}</td></tr>")
    html_out.append("</table>")

    for task, _, _, _, title in tasks:
        html_out.append(f"<h2>{html.escape(title)} ({html.escape(task)})</h2>")
        for outcome in outcome_order:
            rows = [x for x in selected if x[0] == task and x[2] == outcome]
            if not rows:
                continue
            x = rows[0]
            html_out.append("<div class='card'>")
            html_out.append(f"<h3>Case {x[3]} ({x[2]})</h3>")
            html_out.append(f"<p><b>True label:</b> {x[4]} | <b>Predicted:</b> {x[5]} | <b>Probability:</b> {x[6]:.2f}</p>")
            html_out.append(f"<p><b>Clinical summary:</b> {html.escape(x[7])}</p>")
            html_out.append(f"<p><b>Key diagnosis codes (sample):</b> {html.escape(x[8])}</p>")
            html_out.append(f"<p><b>Key procedures (sample):</b> {html.escape(x[9])}</p>")
            html_out.append(f"<p><b>Key medications (sample):</b> {html.escape(x[10])}</p>")
            html_out.append("</div>")

    html_out.append("<h2>Appendix: Selected Cases in One Table</h2>")
    html_out.append("<table><tr><th>task</th><th>outcome</th><th>pair_id</th><th>true_label</th><th>pred_label</th><th>probability</th></tr>")
    for x in sorted(selected, key=lambda z: (z[0], z[2])):
        html_out.append(f"<tr><td>{x[0]}</td><td>{x[2]}</td><td>{x[3]}</td><td>{x[4]}</td><td>{x[5]}</td><td>{x[6]:.2f}</td></tr>")
    html_out.append("</table></main></body></html>")

    out_html.write_text("".join(html_out), encoding="utf-8")

    print(out_md)
    print(out_html)
    print(f"Selected cases: {len(selected)}")


if __name__ == "__main__":
    main()
