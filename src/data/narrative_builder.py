"""Convert structured visit records into natural-language narratives.

Supports two styles:
  - bullet:    "- Diagnoses made: X; Y; Z\n- Medications prescribed: ..."
  - paragraph: "The patient was diagnosed with X, Y, and Z. They were prescribed ..."
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.data.code_mappings import CodeMapper
from src.utils.text_utils import join_items
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_narrative(
    diagnoses: List[str],
    medications: List[str],
    procedures: List[str],
    style: str = "bullet",
    max_diagnoses: int = 50,
    max_medications: int = 50,
    max_procedures: int = 50,
    empty_placeholder: str = "None recorded",
) -> str:
    """Build a clinical narrative from lists of mapped names.

    Parameters
    ----------
    diagnoses, medications, procedures : list of str
        Human-readable names for each code type.
    style : str
        ``"bullet"`` or ``"paragraph"``.

    Returns
    -------
    str
        Deterministic narrative string.
    """
    dx_text = join_items(diagnoses, max_items=max_diagnoses) or empty_placeholder
    med_text = join_items(medications, max_items=max_medications) or empty_placeholder
    px_text = join_items(procedures, max_items=max_procedures) or empty_placeholder

    if style == "paragraph":
        parts = []
        parts.append(f"The patient was diagnosed with: {dx_text}.")
        parts.append(f"Medications prescribed included: {med_text}.")
        parts.append(f"Procedures performed included: {px_text}.")
        return " ".join(parts)
    else:
        # Default: bullet style (matches paper figure)
        lines = [
            f"- Diagnoses made: {dx_text}",
            f"- Medications prescribed: {med_text}",
            f"- Procedures performed: {px_text}",
        ]
        return "\n".join(lines)


def add_narratives(
    df: pd.DataFrame,
    mapper: CodeMapper,
    style: str = "bullet",
    max_diagnoses: int = 50,
    max_medications: int = 50,
    max_procedures: int = 50,
    empty_placeholder: str = "None recorded",
) -> pd.DataFrame:
    """Add a ``narrative_current`` column to the visit-pair DataFrame."""
    narratives = []
    for _, row in df.iterrows():
        dx_names = mapper.map_code_list(row.get("diagnoses_codes_current", ""), "diagnosis")
        px_names = mapper.map_code_list(row.get("procedures_codes_current", ""), "procedure")
        med_names = mapper.map_code_list(row.get("medications_current", ""), "medication")

        text = build_narrative(
            diagnoses=dx_names,
            medications=med_names,
            procedures=px_names,
            style=style,
            max_diagnoses=max_diagnoses,
            max_medications=max_medications,
            max_procedures=max_procedures,
            empty_placeholder=empty_placeholder,
        )
        narratives.append(text)

    df = df.copy()
    df["narrative_current"] = narratives
    return df
