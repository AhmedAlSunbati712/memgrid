from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from handoff.lib.paths import HANDOFF_ROOT, RESULTS_ROOT

MANIFEST_PATH = RESULTS_ROOT / "MANIFEST.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_artifact_paths(handoff_root: Path | None = None) -> list[Path]:
    root = handoff_root or HANDOFF_ROOT
    results = root / "results"
    paths: list[Path] = []

    for setting in ("easy_s40", "easy_s80", "occ50_s40", "occ50_s80", "occ50_dnoise001_s40", "occ50_dnoise001_s80"):
        setting_dir = results / "image_transformers" / setting
        for name in ("baseline_aggregated.csv", "config.txt"):
            candidate = setting_dir / name
            if candidate.exists():
                paths.append(candidate)

    vision_key_files = (
        "results/vision/naturalistic/fruits_combined.csv",
        "results/vision/naturalistic/vegetables_combined.csv",
        "results/vision/naturalistic/animals_combined.csv",
        "results/vision/naturalistic_dam/vit_vs_clip_head_to_head.csv",
        "results/vision/model_comparison/model_comparison_combined.csv",
    )
    for rel in vision_key_files:
        candidate = root / rel
        if candidate.exists():
            paths.append(candidate)

    return sorted(paths)


def build_manifest(handoff_root: Path | None = None) -> dict[str, object]:
    root = handoff_root or HANDOFF_ROOT
    entries: list[dict[str, str]] = []
    for path in canonical_artifact_paths(root):
        rel = path.relative_to(root).as_posix()
        entries.append({"path": rel, "sha256": _sha256(path)})
    return {
        "handoff_root": str(root),
        "artifact_count": len(entries),
        "entries": entries,
    }


def write_manifest(path: Path | None = None, handoff_root: Path | None = None) -> Path:
    target = path or MANIFEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(handoff_root=handoff_root)
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def verify_manifest(path: Path | None = None, handoff_root: Path | None = None) -> list[str]:
    root = handoff_root or HANDOFF_ROOT
    target = path or MANIFEST_PATH
    if not target.exists():
        return [f"Missing manifest: {target}"]

    manifest = json.loads(target.read_text())
    errors: list[str] = []
    for entry in manifest.get("entries", []):
        rel = entry["path"]
        expected = entry["sha256"]
        file_path = root / rel
        if not file_path.exists():
            errors.append(f"Missing artifact: {rel}")
            continue
        actual = _sha256(file_path)
        if actual != expected:
            errors.append(f"Hash mismatch: {rel}")
    return errors


def iter_manifest_paths(manifest_path: Path | None = None) -> Iterable[str]:
    target = manifest_path or MANIFEST_PATH
    manifest = json.loads(target.read_text())
    for entry in manifest.get("entries", []):
        yield entry["path"]
