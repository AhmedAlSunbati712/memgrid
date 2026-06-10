import subprocess
import sys
from pathlib import Path

import pytest

HANDOFF_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HANDOFF_ROOT.parent
PYTHON = sys.executable

pytest.importorskip("timm")


def _run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = HANDOFF_ROOT / "scripts" / script_name
    return subprocess.run(
        [PYTHON, str(script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_vision_synthetic_help():
    result = _run_script("run_vision_synthetic.py", "--help")
    assert result.returncode == 0
    assert "--quick" in result.stdout


def test_run_vision_model_comparison_help():
    result = _run_script("run_vision_model_comparison.py", "--help")
    assert result.returncode == 0


def test_run_vision_naturalistic_baseline_help():
    result = _run_script("run_vision_naturalistic_baseline.py", "--help")
    assert result.returncode == 0


def test_run_vision_naturalistic_dam_help():
    result = _run_script("run_vision_naturalistic_dam.py", "--help")
    assert result.returncode == 0


def test_run_vision_animals_graphs_help():
    result = _run_script("run_vision_animals_graphs.py", "--help")
    assert result.returncode == 0


def test_run_vision_animals_similarity_ordered_help():
    result = _run_script("run_vision_animals_similarity_ordered.py", "--help")
    assert result.returncode == 0


def test_run_vision_animals_similarity_ordered_fixed_help():
    result = _run_script("run_vision_animals_similarity_ordered_fixed.py", "--help")
    assert result.returncode == 0


def test_run_vision_synthetic_quick(tmp_path):
    output_dir = tmp_path / "synthetic"
    result = _run_script(
        "run_vision_synthetic.py",
        "--quick",
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    summaries = list(output_dir.glob("baseline_summary_*.json"))
    assert summaries


def test_run_vision_animals_graphs_quick(tmp_path):
    output_dir = tmp_path / "graphs"
    result = _run_script(
        "run_vision_animals_graphs.py",
        "--quick",
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--seed",
        "0",
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "hard_s40" / "baseline_aggregated.csv").exists()
