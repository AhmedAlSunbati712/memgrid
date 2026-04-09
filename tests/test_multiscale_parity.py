from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import generalization_experiment_multiscale


def _run_generalization_sweep(seed):
    np.random.seed(seed)
    scale_factors = [1.0, 1.3, float(np.sqrt(np.e)), 2.2, 3.0]
    n_order, k = 4, 6
    results = generalization_experiment_multiscale(
        scale_factors,
        n_values=[n_order],
        K_values=[k],
        num_test_points=12,
        num_trials=2,
        steps_multiplier=2,
        n_modules=4,
        n_orientations=3,
        n_cells_per_orientation=4,
        base_freq=0.5,
        beta=0.01,
        alpha=0.5,
        lmbda=0.0,
        verbose=False,
    )
    accuracies = {
        c: float(results[c][n_order][k]["accuracy"]) for c in scale_factors
    }
    best_c = max(accuracies, key=accuracies.get)
    return accuracies, best_c


def test_multiscale_generalization_reproducible():
    first, best_first = _run_generalization_sweep(seed=123)
    second, best_second = _run_generalization_sweep(seed=123)

    assert first == second
    assert best_first == best_second


def test_multiscale_generalization_sweep_outputs_valid_metrics():
    accuracies, best_c = _run_generalization_sweep(seed=123)
    assert best_c in accuracies
    assert len(accuracies) == 5
    assert all(np.isfinite(v) for v in accuracies.values())
