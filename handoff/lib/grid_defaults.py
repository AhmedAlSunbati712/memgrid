from __future__ import annotations

import numpy as np

HANDOFF_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

# Part 4 multiscale defaults (DenseAM.ipynb)
FULL_SCALE_FACTORS = np.linspace(1.0, 3.2, 22)
QUICK_SCALE_FACTORS = np.linspace(1.0, 3.2, 3)

FULL_N_VALUES = [2, 3, 4, 5, 6]
FULL_K_VALUES = [5, 10, 20, 30]
QUICK_N_VALUES = [2, 4]
QUICK_K_VALUES = [5]

N_MODULES = 4
N_ORIENTATIONS = 3
N_CELLS_PER_ORIENTATION = 4
BASE_FREQ = 0.5

NOISE_LEVEL = 0.1
NUM_TRIALS_FULL = 3
NUM_TRIALS_QUICK = 1
NUM_TEST_POINTS = 50
STEPS_MULTIPLIER = 20

BETA = 0.01
ALPHA = 0.5
LMBDA = 0.0
RETRIEVAL_BACKEND = "numba"

# Part 4b break-it baseline
BASE_ENCODER = {
    "n_modules": N_MODULES,
    "n_orientations": N_ORIENTATIONS,
    "n_cells_per_orientation": N_CELLS_PER_ORIENTATION,
    "base_freq": BASE_FREQ,
}

BREAKIT_EXPERIMENTS: dict[str, dict[str, object]] = {
    "orientations": {
        "param": "n_orientations",
        "values": [2, 3, 4, 5, 6, 7],
        "quick_values": [3],
    },
    "base_freq": {
        "param": "base_freq",
        "values": [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7],
        "quick_values": [0.5],
    },
    "modules": {
        "param": "n_modules",
        "values": [3, 4, 5, 6, 7, 8, 9],
        "quick_values": [4],
    },
    "cells": {
        "param": "n_cells_per_orientation",
        "values": [1, 3, 5, 7, 9],
        "quick_values": [4],
    },
    "distributions": {
        "param": "sample_2d_fn",
        "values": ["uniform", "clustered", "biased"],
        "quick_values": ["uniform"],
    },
}

N_VALUES_4B = [2, 3, 4, 5, 6]
K_VALUES_4B = [5, 10, 20]
N_FOR_OPT = 4

DEFAULT_MULTISCALE_OUTPUT = HANDOFF_ROOT / "results" / "grid" / "multiscale"
DEFAULT_BREAKIT_OUTPUT = HANDOFF_ROOT / "results" / "grid" / "breakit"
