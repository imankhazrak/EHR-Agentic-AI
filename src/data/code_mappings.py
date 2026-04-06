"""Code-to-name mapping for diagnoses, procedures, and medications.

Diagnoses and procedures are mapped via the MIMIC-III dictionary tables
(D_ICD_DIAGNOSES, D_ICD_PROCEDURES).  Medications use the DRUG name
directly from PRESCRIPTIONS (already human-readable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.text_utils import clean_code_name

logger = get_logger(__name__)


class CodeMapper:
    """Maps ICD-9 codes to human-readable names using MIMIC-III dictionaries."""

    def __init__(self, interim_dir: str):
        idir = Path(interim_dir)

        # Diagnosis dictionary
        self.dx_map: Dict[str, str] = {}
        dx_path = idir / "d_icd_diagnoses.csv"
        if dx_path.exists():
            d_dx = pd.read_csv(dx_path, dtype={"ICD9_CODE": str})
            for _, row in d_dx.iterrows():
                code = str(row["ICD9_CODE"]).strip()
                # Prefer LONG_TITLE, fall back to SHORT_TITLE
                name = row.get("LONG_TITLE") or row.get("SHORT_TITLE") or code
                self.dx_map[code] = clean_code_name(str(name))
            logger.info("Loaded %d diagnosis code mappings", len(self.dx_map))

        # Procedure dictionary
        self.px_map: Dict[str, str] = {}
        px_path = idir / "d_icd_procedures.csv"
        if px_path.exists():
            d_px = pd.read_csv(px_path, dtype={"ICD9_CODE": str})
            for _, row in d_px.iterrows():
                code = str(row["ICD9_CODE"]).strip()
                name = row.get("LONG_TITLE") or row.get("SHORT_TITLE") or code
                self.px_map[code] = clean_code_name(str(name))
            logger.info("Loaded %d procedure code mappings", len(self.px_map))

        self._missing_dx: set = set()
        self._missing_px: set = set()

    # ------------------------------------------------------------------ #
    def map_diagnosis(self, code: str) -> str:
        """Return human-readable name for an ICD-9 diagnosis code."""
        code = str(code).strip()
        if code in self.dx_map:
            return self.dx_map[code]
        self._missing_dx.add(code)
        return f"[ICD9-DX:{code}]"

    def map_procedure(self, code: str) -> str:
        """Return human-readable name for an ICD-9 procedure code."""
        code = str(code).strip()
        if code in self.px_map:
            return self.px_map[code]
        self._missing_px.add(code)
        return f"[ICD9-PX:{code}]"

    @staticmethod
    def map_medication(drug_name: str) -> str:
        """Return cleaned medication name (already human-readable in MIMIC)."""
        name = str(drug_name).strip()
        if not name or name.lower() == "nan":
            return ""
        return clean_code_name(name)

    # ------------------------------------------------------------------ #
    def map_code_list(self, codes_str: str, code_type: str) -> List[str]:
        """Map a semicolon-separated string of codes to a list of names.

        Parameters
        ----------
        codes_str : str
            Semicolon-separated codes, e.g. ``"4019;25000;2724"``.
        code_type : str
            One of ``"diagnosis"``, ``"procedure"``, ``"medication"``.
        """
        if not codes_str or pd.isna(codes_str):
            return []

        mapper = {
            "diagnosis": self.map_diagnosis,
            "procedure": self.map_procedure,
            "medication": self.map_medication,
        }[code_type]

        names = []
        for code in codes_str.split(";"):
            code = code.strip()
            if not code or code.lower() == "nan":
                continue
            mapped = mapper(code)
            if mapped:
                names.append(mapped)
        return names

    def report_missing(self) -> Dict[str, int]:
        """Return counts of codes with missing mappings."""
        report = {
            "missing_diagnosis_codes": len(self._missing_dx),
            "missing_procedure_codes": len(self._missing_px),
        }
        if self._missing_dx:
            logger.warning("Missing diagnosis mappings: %d unique codes", len(self._missing_dx))
        if self._missing_px:
            logger.warning("Missing procedure mappings: %d unique codes", len(self._missing_px))
        return report
