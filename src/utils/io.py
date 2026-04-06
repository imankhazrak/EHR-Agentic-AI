"""I/O helpers for reading MIMIC tables and saving outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def read_mimic_table(raw_dir: str, table_name: str, usecols: Optional[list] = None, dtype: Optional[dict] = None) -> pd.DataFrame:
    """Read a MIMIC-III compressed CSV table.

    Tries ``<TABLE>.csv.gz`` first, then ``<TABLE>.csv``.
    """
    raw_path = Path(raw_dir)
    for suffix in (".csv.gz", ".csv"):
        fp = raw_path / f"{table_name}{suffix}"
        if fp.exists():
            logger.info("Reading %s", fp)
            return pd.read_csv(fp, usecols=usecols, dtype=dtype, low_memory=False)
    raise FileNotFoundError(f"Cannot find {table_name}.csv(.gz) in {raw_dir}")


def save_dataframe(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Save a DataFrame to CSV, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, **kwargs)
    logger.info("Saved %d rows → %s", len(df), p)


def save_json(obj: Any, path: str | Path) -> None:
    """Save an object as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info("Saved JSON → %s", p)


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_text(text: str, path: str | Path) -> None:
    """Save a string to a text file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
