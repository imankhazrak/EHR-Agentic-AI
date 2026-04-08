"""Archive historical GPT-4o-mini artifacts for safe preservation.

Usage:
    python -m src.scripts.archive_gpt4o_results
    python -m src.scripts.archive_gpt4o_results --copy
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.utils.io import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _looks_like_gpt_artifact(path: Path) -> bool:
    needle = "gpt-4o-mini"
    p = str(path).lower()
    if needle in p:
        return True
    if path.is_file():
        try:
            return needle in path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return False
    return False


def _collect_candidates(root: Path) -> List[Path]:
    if not root.exists():
        return []
    all_paths = list(root.rglob("*"))
    return [p for p in all_paths if p.is_file() and _looks_like_gpt_artifact(p)]


def archive_gpt4o_results(
    workspace_root: Path,
    archive_root_name: str = "gpt_4o_mini_results",
    move_files: bool = True,
) -> Dict:
    outputs_dir = workspace_root / "data" / "outputs"
    logs_dir = workspace_root / "logs"
    archive_root = workspace_root / archive_root_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = archive_root / f"archive_{stamp}"
    target_dir.mkdir(parents=True, exist_ok=True)

    candidates = _collect_candidates(outputs_dir) + _collect_candidates(logs_dir)
    copied: List[str] = []
    skipped: List[str] = []

    for src in candidates:
        rel = src.relative_to(workspace_root)
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped.append(str(rel))
            continue
        shutil.copy2(src, dst)
        copied.append(str(rel))

    moved: List[str] = []
    if move_files:
        for rel_str in copied:
            src = workspace_root / rel_str
            if src.exists():
                src.unlink()
                moved.append(rel_str)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "workspace_root": str(workspace_root),
        "archive_dir": str(target_dir),
        "mode": "move" if move_files else "copy",
        "candidate_count": len(candidates),
        "archived_count": len(copied),
        "moved_count": len(moved),
        "skipped_existing_count": len(skipped),
        "archived_relative_paths": copied,
        "skipped_relative_paths": skipped,
    }
    save_json(manifest, target_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive GPT-4o-mini outputs/logs.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy only (default behavior is copy then delete source files).",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    result = archive_gpt4o_results(
        workspace_root=workspace_root,
        archive_root_name="gpt_4o_mini_results",
        move_files=not args.copy,
    )
    logger.info(
        "Archive complete. mode=%s archived=%d moved=%d",
        result["mode"],
        result["archived_count"],
        result["moved_count"],
    )


if __name__ == "__main__":
    main()
