from __future__ import annotations

import sys
from pathlib import Path

import pytest

HANDOFF_ROOT = Path(__file__).resolve().parent
REPO_ROOT = HANDOFF_ROOT.parent
CORE_ROOT = HANDOFF_ROOT / "core"
DATA_ROOT = HANDOFF_ROOT / "data"


def _prioritize_handoff_paths() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from handoff.lib.paths import ensure_handoff_paths

    ensure_handoff_paths()
    for path in (HANDOFF_ROOT, CORE_ROOT):
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
    sys.path.insert(0, str(HANDOFF_ROOT))
    sys.path.insert(0, str(CORE_ROOT))


_prioritize_handoff_paths()

if not (DATA_ROOT / "datasets_peterson.pkl").exists():
    raise RuntimeError(f"Missing bundled dataset: {DATA_ROOT / 'datasets_peterson.pkl'}")


def pytest_configure(config) -> None:
    _prioritize_handoff_paths()


@pytest.fixture(scope="session")
def handoff_root() -> Path:
    return HANDOFF_ROOT


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
