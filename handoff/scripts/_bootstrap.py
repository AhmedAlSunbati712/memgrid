"""Shared sys.path bootstrap for handoff CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_script_imports(script_file: str | Path) -> Path:
    handoff_root = Path(script_file).resolve().parents[1]
    repo_root = handoff_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from handoff.lib.paths import ensure_handoff_paths

    ensure_handoff_paths()
    return handoff_root
