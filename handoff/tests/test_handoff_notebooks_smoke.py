import json
from pathlib import Path

import numpy as np
import pytest

HANDOFF_ROOT = Path(__file__).resolve().parents[1]

GRID_NOTEBOOK = HANDOFF_ROOT / "notebooks" / "GridDAM.ipynb"
VISION_NOTEBOOK = HANDOFF_ROOT / "notebooks" / "VisionExperiments.ipynb"


def _read_notebook(path: Path):
    assert path.exists(), f"Missing notebook: {path}"
    return json.loads(path.read_text())["cells"]


def _code_source(cell) -> str:
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else str(src)


def _compile_notebook(path: Path):
    cells = _read_notebook(path)
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert code_cells
    for idx, cell in enumerate(code_cells):
        compile(_code_source(cell), f"{path.stem}_cell_{idx}", "exec")


def test_grid_notebook_compiles():
    _compile_notebook(GRID_NOTEBOOK)


def test_vision_notebook_compiles():
    _compile_notebook(VISION_NOTEBOOK)


def test_grid_notebook_quick_execute_smoke(tmp_path):
    cells = _read_notebook(GRID_NOTEBOOK)
    namespace = {
        "__builtins__": __builtins__,
        "Path": Path,
        "np": np,
        "plt": __import__("matplotlib.pyplot"),
    }
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = _code_source(cell)
        if "QUICK_MODE =" in source and "HANDOFF_ROOT" in source:
            patched = []
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("QUICK_MODE"):
                    patched.append("QUICK_MODE = True")
                elif stripped.startswith("LOAD_EXISTING"):
                    patched.append("LOAD_EXISTING = False")
                elif stripped.startswith("OUTPUT_DIR"):
                    patched.append(f"OUTPUT_DIR = Path({str(tmp_path)!r}) / 'grid'")
                else:
                    patched.append(line)
            source = "\n".join(patched)
        if "plt.show()" in source:
            source = source.replace("plt.show()", "plt.close('all')")
        exec(compile(source, f"GridDAM_cell_{idx}", "exec"), namespace)

    assert namespace["QUICK_MODE"] is True
    assert (namespace["OUTPUT_DIR"] / "multiscale" / "ident_results.json").exists()
    assert (namespace["OUTPUT_DIR"] / "breakit" / "orientations" / "orientations_raw.json").exists()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("timm") is None,
    reason="timm required for vision notebook execution smoke.",
)
def test_vision_notebook_load_existing_smoke():
    cells = _read_notebook(VISION_NOTEBOOK)
    namespace = {
        "__builtins__": __builtins__,
        "Path": Path,
        "np": np,
        "plt": __import__("matplotlib.pyplot"),
        "display": lambda obj: None,
        "Image": type("Image", (), {}),
    }
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = _code_source(cell)
        if source.strip().startswith("if RUN_EXPERIMENTS"):
            continue
        if "plt.show()" in source:
            source = source.replace("plt.show()", "plt.close('all')")
        exec(compile(source, f"VisionExperiments_cell_{idx}", "exec"), namespace)

    assert namespace["RUN_EXPERIMENTS"] is False
