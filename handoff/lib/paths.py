from __future__ import annotations

import sys
from pathlib import Path

HANDOFF_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = HANDOFF_ROOT / "core"
DATA_ROOT = HANDOFF_ROOT / "data"
RESULTS_ROOT = HANDOFF_ROOT / "results"


def ensure_handoff_paths() -> Path:
    """Insert handoff root and vendored core on sys.path for flat imports."""
    for path in (HANDOFF_ROOT, CORE_ROOT):
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
    sys.path.insert(0, str(HANDOFF_ROOT))
    sys.path.insert(0, str(CORE_ROOT))
    return HANDOFF_ROOT
