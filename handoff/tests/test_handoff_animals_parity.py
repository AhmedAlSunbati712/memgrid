import csv
import subprocess
import sys
from pathlib import Path

import pytest

HANDOFF_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HANDOFF_ROOT.parent
PYTHON = sys.executable

pytest.importorskip("timm")


def test_animals_graph_quick_run_uses_hard_ident_setting(tmp_path):
    output_dir = tmp_path / "graphs"
    result = subprocess.run(
        [
            PYTHON,
            str(HANDOFF_ROOT / "scripts" / "run_vision_animals_graphs.py"),
            "--quick",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--seed",
            "0",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    new_csv = output_dir / "hard_s40" / "baseline_aggregated.csv"
    config_txt = output_dir / "hard_s40" / "config.txt"
    assert new_csv.exists()
    assert config_txt.exists()
    assert "noise_shift_occlusion" in config_txt.read_text()
    assert "decision_noise_std: 0.01" in config_txt.read_text()

    with new_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert float(row["gen_accuracy_mean"]) >= 0.0
        assert float(row["ident_accuracy_mean"]) >= 0.0
