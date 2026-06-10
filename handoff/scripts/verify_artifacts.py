from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import json

from handoff.scripts._bootstrap import bootstrap_script_imports

bootstrap_script_imports(__file__)

from handoff.lib.artifact_manifest import MANIFEST_PATH, verify_manifest


def main() -> int:
    errors = verify_manifest(MANIFEST_PATH)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text())
    print(f"OK: {MANIFEST_PATH.name} verified.")
    print(f"Verified {manifest.get('artifact_count', 0)} artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
