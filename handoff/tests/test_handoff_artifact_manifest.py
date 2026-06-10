import json
import subprocess
import sys
from pathlib import Path

HANDOFF_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HANDOFF_ROOT.parent
MANIFEST_PATH = HANDOFF_ROOT / "results" / "MANIFEST.json"
PYTHON = sys.executable


def test_manifest_exists_and_covers_key_deliverables():
    assert MANIFEST_PATH.exists()
    manifest = json.loads(MANIFEST_PATH.read_text())
    paths = {entry["path"] for entry in manifest["entries"]}
    assert manifest["artifact_count"] >= 17
    assert "results/vision/naturalistic/fruits_combined.csv" in paths
    assert "results/vision/naturalistic/vegetables_combined.csv" in paths
    assert "results/vision/naturalistic/animals_combined.csv" in paths
    assert "results/vision/model_comparison/model_comparison_combined.csv" in paths
    assert "results/vision/naturalistic_dam/vit_vs_clip_head_to_head.csv" in paths
    image_settings = {
        p.split("/")[2]
        for p in paths
        if p.startswith("results/image_transformers/") and p.endswith("baseline_aggregated.csv")
    }
    assert {
        "easy_s40",
        "easy_s80",
        "occ50_s40",
        "occ50_s80",
        "occ50_dnoise001_s40",
        "occ50_dnoise001_s80",
    }.issubset(image_settings)


def test_verify_artifacts_script_exits_zero():
    result = subprocess.run(
        [PYTHON, str(HANDOFF_ROOT / "scripts" / "verify_artifacts.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_verify_manifest_helper():
    from handoff.lib.artifact_manifest import verify_manifest

    errors = verify_manifest(MANIFEST_PATH, handoff_root=HANDOFF_ROOT)
    assert errors == []
