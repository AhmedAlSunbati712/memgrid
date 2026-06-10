import json
import subprocess
import sys
from pathlib import Path

HANDOFF_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HANDOFF_ROOT.parent
PYTHON = sys.executable


def _run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = HANDOFF_ROOT / "scripts" / script_name
    return subprocess.run(
        [PYTHON, str(script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_grid_multiscale_help():
    result = _run_script("run_grid_multiscale.py", "--help")
    assert result.returncode == 0
    assert "--quick" in result.stdout
    assert "--save-csv" in result.stdout


def test_run_grid_breakit_help():
    result = _run_script("run_grid_breakit.py", "--help")
    assert result.returncode == 0
    assert "--experiment" in result.stdout
    assert "--save-csv" in result.stdout


def test_run_grid_multiscale_quick(tmp_path):
    output_dir = tmp_path / "multiscale"
    result = _run_script(
        "run_grid_multiscale.py",
        "--quick",
        "--output-dir",
        str(output_dir),
        "--seed",
        "7",
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "ident_results.json").exists()
    assert (output_dir / "gen_results.json").exists()
    assert (output_dir / "tradeoff_by_n.png").exists()
    assert not (output_dir / "tradeoff_summary.csv").exists()
    payload = json.loads((output_dir / "ident_results.json").read_text())
    assert isinstance(payload, dict)
    assert payload


def test_run_grid_breakit_quick(tmp_path):
    output_dir = tmp_path / "breakit"
    result = _run_script(
        "run_grid_breakit.py",
        "--experiment",
        "orientations",
        "--quick",
        "--output-dir",
        str(output_dir),
        "--seed",
        "7",
    )
    assert result.returncode == 0, result.stderr
    section_dir = output_dir / "orientations"
    assert (section_dir / "orientations_raw.json").exists()
    assert (section_dir / "orientations_optimum_summary.json").exists()
    assert not (section_dir / "orientations_optimum_c.csv").exists()
    assert list(section_dir.glob("orientations_*_tradeoff.png"))

    from handoff.lib.grid_runner import load_breakit_raw_entry, scale_factors_from_results

    ident_res, gen_res, plot_scales = load_breakit_raw_entry(
        section_dir / "orientations_raw.json",
        "3",
    )
    aligned = scale_factors_from_results(ident_res, [1.0, 2.1, 3.2])
    assert len(aligned) == len(plot_scales)
    assert all(c in ident_res for c in aligned)
