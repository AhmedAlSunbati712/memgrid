from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.synthetic_sweep import build_reduced_sweep_configs, build_sweep_configs, detect_tradeoff_candidates


def test_build_sweep_configs_contains_expected_axes():
    configs = build_sweep_configs()

    assert configs
    task_modes = {config.task_mode for config in configs}
    poolings = {config.pooling for config in configs}
    assert task_modes == {"mixed_color_position", "color_only", "position_only"}
    assert poolings == {"cls", "mean_tokens"}

    color_only_jitters = {
        config.color_only_jitter_std for config in configs if config.task_mode == "color_only"
    }
    assert color_only_jitters == {4.0, 8.0, 16.0}


def test_build_reduced_sweep_configs_is_subset_with_storage_curve():
    configs = build_reduced_sweep_configs()

    assert configs
    assert {config.n_per_color for config in configs} == {12}
    assert {config.square_size for config in configs} == {56}
    assert {config.ident_noise_std for config in configs} == {0.0, 5.0}
    assert {config.ident_max_shift for config in configs} == {0, 2}


def test_detect_tradeoff_candidates_accepts_interior_peak_with_positive_margin():
    rows = [
        {
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 2.0,
            "ident_max_shift": 1,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 2,
            "ident_accuracy": 20.0,
            "gen_accuracy": 40.0,
            "avg_margin": -0.01,
        },
        {
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 2.0,
            "ident_max_shift": 1,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 4,
            "ident_accuracy": 45.0,
            "gen_accuracy": 80.0,
            "avg_margin": 0.02,
        },
        {
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 2.0,
            "ident_max_shift": 1,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 6,
            "ident_accuracy": 70.0,
            "gen_accuracy": 55.0,
            "avg_margin": 0.01,
        },
    ]

    assessment = detect_tradeoff_candidates(rows)[0]
    assert assessment.qualifies is True
    assert assessment.reason == "qualifies"
    assert assessment.peak_index == 1


def test_detect_tradeoff_candidates_rejects_non_interior_peak():
    rows = [
        {
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 2.0,
            "ident_max_shift": 1,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 2,
            "ident_accuracy": 20.0,
            "gen_accuracy": 80.0,
            "avg_margin": 0.02,
        },
        {
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 2.0,
            "ident_max_shift": 1,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 4,
            "ident_accuracy": 45.0,
            "gen_accuracy": 60.0,
            "avg_margin": 0.01,
        },
        {
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 2.0,
            "ident_max_shift": 1,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 6,
            "ident_accuracy": 70.0,
            "gen_accuracy": 50.0,
            "avg_margin": 0.01,
        },
    ]

    assessment = detect_tradeoff_candidates(rows)[0]
    assert assessment.qualifies is False
    assert assessment.reason == "peak_not_interior"


def test_detect_tradeoff_candidates_rejects_no_positive_margin():
    rows = [
        {
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 2.0,
            "ident_max_shift": 1,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 2,
            "ident_accuracy": 20.0,
            "gen_accuracy": 40.0,
            "avg_margin": -0.02,
        },
        {
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 2.0,
            "ident_max_shift": 1,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 4,
            "ident_accuracy": 45.0,
            "gen_accuracy": 80.0,
            "avg_margin": -0.01,
        },
        {
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 2.0,
            "ident_max_shift": 1,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 6,
            "ident_accuracy": 70.0,
            "gen_accuracy": 55.0,
            "avg_margin": -0.01,
        },
    ]

    assessment = detect_tradeoff_candidates(rows)[0]
    assert assessment.qualifies is False
    assert assessment.reason == "no_positive_margin"
