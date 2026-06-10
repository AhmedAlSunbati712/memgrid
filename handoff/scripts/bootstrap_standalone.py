"""Validate handoff layout and artifact integrity for standalone checkouts."""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from handoff.scripts._bootstrap import bootstrap_script_imports

HANDOFF_ROOT = bootstrap_script_imports(__file__)

REQUIRED = (
    HANDOFF_ROOT / "core" / "experiments.py",
    HANDOFF_ROOT / "vision" / "__init__.py",
    HANDOFF_ROOT / "data" / "datasets_peterson.pkl",
    HANDOFF_ROOT / "results" / "MANIFEST.json",
    HANDOFF_ROOT / "lib" / "paths.py",
)


def main() -> int:
    missing = [str(p.relative_to(HANDOFF_ROOT)) for p in REQUIRED if not p.exists()]
    if missing:
        print("Missing required paths:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 1

    symlinks = [p for p in HANDOFF_ROOT.rglob("*") if p.is_symlink()]
    if symlinks:
        print("Unexpected symlinks under handoff/:", file=sys.stderr)
        for link in symlinks[:10]:
            print(f"  - {link.relative_to(HANDOFF_ROOT)}", file=sys.stderr)
        return 1

    from handoff.lib.artifact_manifest import verify_manifest

    errors = verify_manifest(handoff_root=HANDOFF_ROOT)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"OK: handoff layout valid at {HANDOFF_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
