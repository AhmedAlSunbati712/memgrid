from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DAM import SilentDAM
from utils import create_multiscale_encoder, find_nearest_cosine, find_nearest_euclidean_2d


def _run_generalization_sweep(seed):
    np.random.seed(seed)
    scale_factors = [1.0, 1.3, float(np.sqrt(np.e)), 2.2, 3.0]
    n = 4
    k = 6
    num_trials = 2
    num_test_points = 12
    steps_multiplier = 2
    accuracies = {}

    for c in scale_factors:
        encoder = create_multiscale_encoder(c, n_modules=4, base_freq=0.5)
        enc_dim = encoder.encode(np.array([0.0, 0.0])).size
        successes = []
        for _ in range(num_trials):
            stored_points_2d = np.random.uniform(-1, 1, (k, 2))
            encoded_patterns = encoder.encode(stored_points_2d)
            dam = SilentDAM(encoded_patterns, n=n, beta=0.01, alpha=0.5, lmbda=0.0)
            test_points_2d = np.random.uniform(-1, 1, (num_test_points, 2))
            for test_point in test_points_2d:
                gt_idx = find_nearest_euclidean_2d(test_point, stored_points_2d)
                test_encoded = encoder.encode(test_point)
                retrieved, _, _, _ = dam.retrieve_differential(
                    test_encoded,
                    steps=steps_multiplier * enc_dim,
                )
                retrieved_idx = find_nearest_cosine(retrieved, encoded_patterns)
                successes.append(retrieved_idx == gt_idx)
        accuracies[c] = float(np.mean(successes) * 100.0)

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
