import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matplotlib.figure import Figure

from DAM import DenseAssociativeMemory, SilentDAM
from Encoder import GridEncoder
from experiments import (
    analyze_multiscale_tradeoff,
    get_generalization_optimum_c,
    generalization_experiment,
    generalization_experiment_multiscale,
    identification_experiment,
    identification_experiment_multiscale,
    run_breakit_sweep,
    run_multiscale_ident_gen_sweep,
    sample_biased,
    sample_clustered,
    sample_uniform,
    structure_preservation_analysis,
)
from plotting import (
    plot_dam_retrieval_traces,
    plot_pattern_correlations,
    plot_tradeoff_breakit,
    summarize_multiscale_tradeoff_table,
)
from utils import create_multiscale_encoder, create_single_scale_encoder, encode_points, encode_single_point


def test_dam_retrieve_differential_smoke():
    np.random.seed(7)
    patterns = np.random.uniform(-1, 1, (3, 8))
    dam = DenseAssociativeMemory(patterns, n=3, beta=0.05, alpha=0.4, lmbda=0.0, verbose=False)
    noisy = patterns[0] + np.random.normal(0, 0.01, size=patterns.shape[1])

    retrieved, best, energy_trace, similarity_trace = dam.retrieve_differential(noisy, steps=20)

    assert retrieved.shape == (8,)
    assert best.shape == (8,)
    assert isinstance(energy_trace, list)
    assert isinstance(similarity_trace, list)


def test_dam_retrieve_differential_default_contract_4tuple():
    np.random.seed(70)
    patterns = np.random.uniform(-1, 1, (4, 10))
    dam = DenseAssociativeMemory(patterns, n=3, beta=0.03, alpha=0.5, lmbda=0.0, verbose=False)
    noisy = patterns[1] + np.random.normal(0, 0.01, size=patterns.shape[1])
    out = dam.retrieve_differential(noisy, steps=25)
    assert isinstance(out, tuple)
    assert len(out) == 4
    state, best, energy_trace, similarity_trace = out
    assert state.shape == (10,)
    assert best.shape == (10,)
    assert isinstance(energy_trace, list)
    assert isinstance(similarity_trace, list)


def test_dam_retrieve_differential_optional_best_idx_5tuple():
    np.random.seed(71)
    patterns = np.random.uniform(-1, 1, (5, 12))
    dam = DenseAssociativeMemory(patterns, n=4, beta=0.02, alpha=0.4, lmbda=0.0, verbose=False)
    noisy = patterns[0] + np.random.normal(0, 0.02, size=patterns.shape[1])
    out = dam.retrieve_differential(
        noisy,
        steps=40,
        return_best_idx=True,
    )
    assert isinstance(out, tuple)
    assert len(out) == 5
    state, best, energy_trace, similarity_trace, best_idx = out
    assert state.shape == (12,)
    assert best.shape == (12,)
    assert isinstance(energy_trace, list)
    assert isinstance(similarity_trace, list)
    assert isinstance(best_idx, int)
    assert 0 <= best_idx < patterns.shape[0]


def test_dam_numpy_numba_backend_parity_with_fixed_update_indices():
    np.random.seed(72)
    patterns = np.random.uniform(-1, 1, (6, 14))
    dam = DenseAssociativeMemory(patterns, n=4, beta=0.02, alpha=0.5, lmbda=0.0, verbose=False)
    noisy = patterns[2] + np.random.normal(0, 0.01, size=patterns.shape[1])
    steps = 120
    update_indices = np.random.randint(0, patterns.shape[1], size=steps)

    state_np, _, _, _, best_np = dam.retrieve_differential(
        noisy,
        steps=steps,
        update_indices=update_indices,
        trace_every=0,
        backend="numpy",
        return_best_idx=True,
    )
    state_nb, _, _, _, best_nb = dam.retrieve_differential(
        noisy,
        steps=steps,
        update_indices=update_indices,
        trace_every=0,
        backend="numba",
        return_best_idx=True,
    )
    assert best_np == best_nb
    assert np.allclose(state_np, state_nb, rtol=1e-7, atol=1e-7)


def test_dam_retrieve_differential_invalid_backend_raises():
    np.random.seed(73)
    patterns = np.random.uniform(-1, 1, (3, 8))
    dam = DenseAssociativeMemory(patterns, n=3, beta=0.05, alpha=0.4, lmbda=0.0, verbose=False)
    noisy = patterns[0] + np.random.normal(0, 0.01, size=patterns.shape[1])
    try:
        dam.retrieve_differential(noisy, steps=10, backend="bogus")
        assert False, "Expected ValueError for invalid backend"
    except ValueError as exc:
        assert "Unknown backend" in str(exc)


def test_dam_numba_backend_requires_trace_every_zero():
    np.random.seed(74)
    patterns = np.random.uniform(-1, 1, (3, 8))
    dam = DenseAssociativeMemory(patterns, n=3, beta=0.05, alpha=0.4, lmbda=0.0, verbose=False)
    noisy = patterns[0] + np.random.normal(0, 0.01, size=patterns.shape[1])
    try:
        dam.retrieve_differential(noisy, steps=10, backend="numba", trace_every=10)
        assert False, "Expected ValueError when trace_every > 0 for numba backend"
    except ValueError as exc:
        assert "trace_every=0" in str(exc)


def test_silent_dam_energy_smoke():
    np.random.seed(8)
    patterns = np.random.uniform(-1, 1, (2, 6))
    dam = SilentDAM(patterns, n=2, beta=0.1, alpha=0.5, lmbda=0.01)
    state = np.random.uniform(-1, 1, 6)

    energy = dam.energy(state)

    assert np.isscalar(energy)


def test_grid_encoder_and_utils_shapes_smoke():
    np.random.seed(9)
    encoder = GridEncoder(n_modules=2, n_orientations=3, n_cells_per_orientation=2, scales=[0.5, 1.0])
    single = encoder.encode(np.array([0.2, -0.4]))
    batch = encoder.encode(np.array([[0.2, -0.4], [0.0, 0.1]]))

    assert single.shape == (24,)
    assert batch.shape == (2, 24)

    util_encoder = create_single_scale_encoder(scale=1.0, n_orientations=3, n_cells=2)
    util_single = encode_single_point(util_encoder, np.array([0.1, 0.2]))
    util_batch = encode_points(util_encoder, np.array([[0.1, 0.2], [0.2, 0.3]]))
    assert util_single.shape == (12,)
    assert util_batch.shape == (2, 12)

    multi_encoder = create_multiscale_encoder(scale_factor=np.sqrt(np.e), n_modules=3, base_freq=0.5)
    assert multi_encoder.scales.shape == (3,)


def test_experiment_entrypoints_smoke():
    np.random.seed(10)
    ident = identification_experiment(
        scales=[0.5],
        n_values=[2],
        K_values=[2],
        noise_level=0.01,
        num_trials=1,
        steps_multiplier=1,
        n_orientations=2,
        n_cells=2,
    )
    gen = generalization_experiment(
        scales=[0.5],
        n_values=[2],
        K_values=[2],
        num_test_points=2,
        num_trials=1,
        steps_multiplier=1,
        n_orientations=2,
        n_cells=2,
    )

    assert 0.5 in ident and 2 in ident[0.5] and 2 in ident[0.5][2]
    assert 0.5 in gen and 2 in gen[0.5] and 2 in gen[0.5][2]


def _assert_metric_cell(cell):
    assert set(cell.keys()) == {"accuracy", "avg_sim", "std_sim"}
    assert np.isfinite(cell["accuracy"])
    assert np.isfinite(cell["avg_sim"])
    assert np.isfinite(cell["std_sim"])


def test_multiscale_experiment_apis_smoke():
    np.random.seed(11)
    scale_factors = [1.1]
    n_values = [2]
    K_values = [2]
    ident = identification_experiment_multiscale(
        scale_factors,
        n_values,
        K_values,
        noise_level=0.01,
        num_trials=1,
        steps_multiplier=1,
        n_modules=2,
        n_orientations=2,
        n_cells_per_orientation=2,
        base_freq=0.5,
        verbose=False,
    )
    gen = generalization_experiment_multiscale(
        scale_factors,
        n_values,
        K_values,
        num_test_points=2,
        num_trials=1,
        steps_multiplier=1,
        n_modules=2,
        n_orientations=2,
        n_cells_per_orientation=2,
        base_freq=0.5,
        verbose=False,
    )
    c, n, K = scale_factors[0], n_values[0], K_values[0]
    assert c in ident and n in ident[c] and K in ident[c][n]
    assert c in gen and n in gen[c] and K in gen[c][n]
    _assert_metric_cell(ident[c][n][K])
    _assert_metric_cell(gen[c][n][K])

    ident2, gen2 = run_multiscale_ident_gen_sweep(
        scale_factors,
        n_values,
        K_values,
        noise_level=0.01,
        num_test_points=2,
        num_trials=1,
        steps_multiplier=1,
        n_modules=2,
        n_orientations=2,
        n_cells_per_orientation=2,
        base_freq=0.5,
        verbose=False,
    )
    assert set(ident2.keys()) == set(scale_factors)
    assert set(gen2.keys()) == set(scale_factors)
    _assert_metric_cell(ident2[c][n][K])
    _assert_metric_cell(gen2[c][n][K])


def test_multiscale_analysis_helpers_smoke():
    np.random.seed(12)
    scale_factors = [1.1, 1.5]
    n_values = [2, 4]
    k_values = [2, 3]

    ident, gen = run_multiscale_ident_gen_sweep(
        scale_factors,
        n_values,
        k_values,
        noise_level=0.01,
        num_test_points=2,
        num_trials=1,
        steps_multiplier=1,
        n_modules=2,
        n_orientations=2,
        n_cells_per_orientation=2,
        base_freq=0.5,
        verbose=False,
    )

    analysis = analyze_multiscale_tradeoff(
        ident, gen, scale_factors, n_values, k_values, print_summary=False
    )
    assert set(analysis.keys()) == {"rows", "trends"}
    assert len(analysis["rows"]) == len(scale_factors) * len(n_values) * len(k_values)
    assert len(analysis["trends"]) == len(n_values)

    c_opt, gen_acc, ident_acc = get_generalization_optimum_c(
        gen, ident, scale_factors, n_values, k_values, n_for_opt=4
    )
    assert c_opt in scale_factors
    assert np.isfinite(gen_acc)
    assert np.isfinite(ident_acc)

    structure = structure_preservation_analysis(
        scale_factors,
        n_points=20,
        n_samples=50,
        n_modules=2,
        n_orientations=2,
        n_cells_per_orientation=2,
        base_freq=0.5,
    )
    assert set(structure.keys()) == set(scale_factors)
    for cell in structure.values():
        assert set(cell.keys()) == {"correlation", "p_value"}
        assert np.isfinite(cell["correlation"])


def test_breakit_wrapper_with_sampling_smoke():
    np.random.seed(13)
    sample_fns = [
        sample_uniform,
        lambda K: sample_clustered(K, n_clusters=2, sigma=0.1),
        sample_biased,
    ]
    for sample_fn in sample_fns:
        ident, gen = run_breakit_sweep(
            scale_factors=[1.2],
            n_values=[2],
            K_values=[2],
            n_modules=2,
            n_orientations=2,
            n_cells_per_orientation=2,
            base_freq=0.5,
            noise_level=0.01,
            num_trials=1,
            num_test_points=2,
            steps_multiplier=1,
            sample_2d_fn=sample_fn,
            verbose=False,
        )
        _assert_metric_cell(ident[1.2][2][2])
        _assert_metric_cell(gen[1.2][2][2])


def _synthetic_ident_gen(scale_factors, n_values, K_values):
    ident, gen = {}, {}
    for c in scale_factors:
        ident[c], gen[c] = {}, {}
        for n in n_values:
            ident[c][n], gen[c][n] = {}, {}
            for K in K_values:
                ident[c][n][K] = {
                    "accuracy": float(40 + n + K),
                    "avg_sim": 0.5,
                    "std_sim": 0.05,
                }
                gen[c][n][K] = {
                    "accuracy": float(35 + n + K),
                    "avg_sim": 0.45,
                    "std_sim": 0.06,
                }
    return ident, gen


def test_plotting_helpers_smoke():
    import matplotlib.pyplot as plt

    scale_factors = (1.0, 1.5)
    n_values = (2, 4)
    K_values = (2, 3)
    ident, gen = _synthetic_ident_gen(scale_factors, n_values, K_values)

    summary = summarize_multiscale_tradeoff_table(
        ident, gen, scale_factors, n_values, K_values
    )
    assert set(summary.keys()) == {"rows", "averages_by_cn", "trends"}
    assert len(summary["rows"]) == len(n_values) * len(scale_factors) * len(K_values)
    assert len(summary["averages_by_cn"]) == len(n_values) * len(scale_factors)
    assert len(summary["trends"]) == len(n_values)
    for row in summary["rows"]:
        assert {"scale_c", "n", "K", "ident_accuracy", "gen_accuracy", "tradeoff_label"} <= row.keys()

    fig = plot_tradeoff_breakit(ident, gen, scale_factors, n_values, K_values)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)

    patterns = np.random.default_rng(0).normal(size=(4, 8))
    fig2 = plot_pattern_correlations(patterns)
    try:
        assert isinstance(fig2, Figure)
        assert len(fig2.axes) == 2
    finally:
        plt.close(fig2)

    energy = np.linspace(1.0, 0.2, 5)
    sim = np.random.default_rng(1).uniform(0, 1, size=(5, 3))
    fig3 = plot_dam_retrieval_traces(energy, sim, target_pattern_idx=0)
    try:
        assert isinstance(fig3, Figure)
        assert len(fig3.axes) == 2
    finally:
        plt.close(fig3)
