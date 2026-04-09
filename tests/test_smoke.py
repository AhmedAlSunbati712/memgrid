import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DAM import DenseAssociativeMemory, SilentDAM
from Encoder import GridEncoder
from experiments import generalization_experiment, identification_experiment
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
