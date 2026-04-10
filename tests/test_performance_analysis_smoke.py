import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "PerformanceAnalysis.ipynb"


def _read_notebook_cells():
    assert NOTEBOOK_PATH.exists(), f"Missing notebook: {NOTEBOOK_PATH}"
    nb = json.loads(NOTEBOOK_PATH.read_text())
    cells = nb.get("cells", [])
    assert isinstance(cells, list) and len(cells) > 0
    return cells


def _code_source(cell):
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def test_performance_notebook_code_cells_compile_smoke():
    cells = _read_notebook_cells()
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert code_cells, "Expected code cells in PerformanceAnalysis.ipynb"
    for i, cell in enumerate(code_cells):
        source = _code_source(cell)
        compile(source, f"PerformanceAnalysis_cell_{i}", "exec")


def test_performance_notebook_core_defs_execute_smoke():
    cells = _read_notebook_cells()
    # Execute only setup/definition cells (skip benchmark/profiler runner cells).
    # Skip cell 16 in execution-only smoke:
    # it defines a numba kernel with cache=True, and numba caching requires
    # a real file-backed source locator (not an exec-compiled pseudo filename).
    # The JIT cell is still validated by the compile smoke test above.
    exec_indices = [1, 2, 4, 5, 7, 13, 14]

    namespace = {}
    for idx in exec_indices:
        source = _code_source(cells[idx])
        exec(compile(source, f"PerformanceAnalysis_cell_{idx}", "exec"), namespace)

    np.random.seed(123)

    # Encoder + nearest-cosine smoke
    baseline_encoder = namespace["baseline_create_multiscale_encoder"](
        1.2, n_modules=2, n_orientations=2, n_cells_per_orientation=2, base_freq=0.5
    )
    opt_encoder = namespace["opt_create_multiscale_encoder"](
        1.2, n_modules=2, n_orientations=2, n_cells_per_orientation=2, base_freq=0.5
    )
    opt_encoder.phases = np.array(baseline_encoder.phases, copy=True)

    points = np.random.uniform(-1, 1, (4, 2))
    enc_base = namespace["baseline_encode_points"](baseline_encoder, points)
    enc_opt = namespace["opt_encode_points"](opt_encoder, points)
    assert enc_base.shape == enc_opt.shape
    assert np.isfinite(enc_base).all()
    assert np.isfinite(enc_opt).all()
    # Optimized encoder may use a different feature column ordering while
    # preserving the same encoded values; avoid brittle strict column matching.
    assert np.allclose(
        np.sort(enc_base, axis=1), np.sort(enc_opt, axis=1), atol=1e-10, rtol=1e-10
    )

    query = enc_base[0] + np.random.normal(0, 0.01, enc_base.shape[1])
    idx_base = namespace["baseline_find_nearest_encoded"](query, enc_base)
    idx_opt = namespace["opt_find_nearest_encoded"](query, enc_base)
    assert isinstance(idx_base, int)
    assert isinstance(idx_opt, int)

    # Retrieval smoke on tiny settings
    steps = 12
    update_indices = np.random.randint(0, enc_base.shape[1], size=steps)

    dam_base = namespace["BaselineDAM"](enc_base, n=3, beta=0.01, alpha=0.5, lmbda=0.0)
    dam_opt = namespace["OptimizedDAM"](enc_base, n=3, beta=0.01, alpha=0.5, lmbda=0.0)
    state_base, _, _, _, best_idx_base = dam_base.retrieve_differential(
        query, steps=steps, update_indices=update_indices, trace_every=0
    )
    state_opt, _, _, _, best_idx_opt = dam_opt.retrieve_differential(
        query, steps=steps, update_indices=update_indices, trace_every=0
    )
    assert state_base.shape == state_opt.shape == (enc_base.shape[1],)
    assert isinstance(best_idx_base, int)
    assert isinstance(best_idx_opt, int)

    # Tiny end-to-end sweep smoke for notebook-local baseline/optimized implementations
    tiny_cfg = {
        "scale_factors": [1.2],
        "n_values": [2],
        "K_values": [2],
        "n_modules": 1,
        "n_orientations": 2,
        "n_cells_per_orientation": 1,
        "base_freq": 0.5,
        "noise_level": 0.01,
        "num_trials": 1,
        "num_test_points": 1,
        "steps_multiplier": 1,
    }

    np.random.seed(321)
    ident_base, gen_base = namespace["baseline_run_breakit_sweep"](**tiny_cfg)
    np.random.seed(321)
    ident_opt, gen_opt = namespace["opt_run_breakit_sweep"](**tiny_cfg)

    c, n, k = 1.2, 2, 2
    for res in (ident_base, gen_base, ident_opt, gen_opt):
        assert c in res and n in res[c] and k in res[c][n]
        cell = res[c][n][k]
        assert set(cell.keys()) == {"accuracy", "avg_sim", "std_sim"}
        assert np.isfinite(cell["accuracy"])
        assert np.isfinite(cell["avg_sim"])
        assert np.isfinite(cell["std_sim"])
