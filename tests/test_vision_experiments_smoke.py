from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.vision_experiments import (
    DEFAULT_COLLAPSE_THRESHOLD,
    DEFAULT_OVERSHOOT_RATIO,
    compute_retrieved_norm_ratio,
    evaluate_layerwise_baseline,
    find_nearest_by_metric,
    is_retrieval_calibrated,
    is_retrieval_collapsed,
    is_retrieval_overshooting,
    rank_sweep_rows,
    run_layerwise_generalization,
    run_layerwise_identification,
    score_against_stored,
    sweep_layerwise_dam_configs,
)


def _tiny_feature_bundle():
    stored = {
        "layer_0": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "layer_6": np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32),
    }
    probes = {
        "layer_0": np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
        "layer_6": np.array([[0.75, 0.25], [0.25, 0.75]], dtype=np.float32),
    }
    return stored, probes


def test_run_layerwise_identification_smoke():
    stored, probes = _tiny_feature_bundle()
    results = run_layerwise_identification(
        stored,
        probes,
        np.array([0, 1]),
        n=2,
        beta=0.05,
        alpha=0.5,
        lmbda=0.0,
        steps_multiplier=5,
        retrieval_backend="numpy",
        trace_every=0,
    )

    assert set(results.keys()) == {"layer_0", "layer_6"}
    for metrics in results.values():
        for key in (
            "accuracy",
            "avg_sim",
            "std_sim",
            "baseline_accuracy",
            "avg_initial_sim",
            "avg_delta_sim",
            "avg_retrieved_norm",
            "winner_agreement",
        ):
            assert np.isfinite(metrics[key])
        assert metrics["metric"] == "cosine"
        assert len(metrics["baseline_winners"]) == 2
        assert len(metrics["dam_winners"]) == 2


def test_run_layerwise_generalization_smoke_is_deterministic():
    stored, probes = _tiny_feature_bundle()
    kwargs = dict(
        n=2,
        beta=0.05,
        alpha=0.5,
        lmbda=0.0,
        steps_multiplier=5,
        retrieval_backend="numpy",
        trace_every=0,
    )

    first = run_layerwise_generalization(stored, probes, np.array([0, 1]), **kwargs)
    second = run_layerwise_generalization(stored, probes, np.array([0, 1]), **kwargs)

    assert first == second


def test_metric_helpers_expected_winners():
    stored = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([0.8, 0.2], dtype=np.float32)

    assert find_nearest_by_metric(query, stored, metric="cosine") == 0
    assert find_nearest_by_metric(query, stored, metric="dot") == 0
    assert find_nearest_by_metric(query, stored, metric="euclidean") == 0
    scores = score_against_stored(query, stored, metric="euclidean")
    assert scores[0] > scores[1]


def test_cosine_and_euclidean_agree_for_normalized_vectors():
    stored = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)

    assert find_nearest_by_metric(query, stored, metric="cosine") == find_nearest_by_metric(
        query, stored, metric="euclidean"
    )


def test_evaluate_layerwise_baseline_smoke():
    stored, probes = _tiny_feature_bundle()
    results = evaluate_layerwise_baseline(stored, probes, np.array([0, 1]), metric="cosine")

    assert set(results.keys()) == {"layer_0", "layer_6"}
    for metrics in results.values():
        assert np.isfinite(metrics["accuracy"])
        assert np.isfinite(metrics["avg_sim"])
        assert np.isfinite(metrics["std_sim"])
        assert metrics["metric"] == "cosine"
        assert len(metrics["winners"]) == 2


def test_is_retrieval_collapsed_threshold():
    assert is_retrieval_collapsed(0.001, collapse_threshold=DEFAULT_COLLAPSE_THRESHOLD)
    assert not is_retrieval_collapsed(0.5, collapse_threshold=DEFAULT_COLLAPSE_THRESHOLD)


def test_norm_ratio_and_calibration_helpers():
    ratio = compute_retrieved_norm_ratio(3.0, 1.5)
    assert np.isclose(ratio, 2.0)
    assert not is_retrieval_overshooting(ratio, overshoot_ratio=DEFAULT_OVERSHOOT_RATIO)
    assert is_retrieval_overshooting(2.1, overshoot_ratio=DEFAULT_OVERSHOOT_RATIO)
    assert is_retrieval_calibrated(1.0, 1.0)
    assert not is_retrieval_calibrated(0.001, 1.0)
    assert not is_retrieval_calibrated(3.0, 1.0)


def test_rank_sweep_rows_prefers_calibrated_then_delta_then_accuracy():
    rows = [
        {
            "retrieval_calibrated": False,
            "retrieval_collapse": False,
            "retrieval_overshoot": True,
            "retrieved_norm_ratio": 12.0,
            "avg_delta_sim": 0.9,
            "dam_accuracy": 99.0,
        },
        {
            "retrieval_calibrated": False,
            "retrieval_collapse": True,
            "retrieval_overshoot": False,
            "retrieved_norm_ratio": 0.01,
            "avg_delta_sim": 0.9,
            "dam_accuracy": 99.0,
        },
        {
            "retrieval_calibrated": True,
            "retrieval_collapse": False,
            "retrieval_overshoot": False,
            "retrieved_norm_ratio": 1.4,
            "avg_delta_sim": 0.1,
            "dam_accuracy": 70.0,
        },
        {
            "retrieval_calibrated": True,
            "retrieval_collapse": False,
            "retrieval_overshoot": False,
            "retrieved_norm_ratio": 1.1,
            "avg_delta_sim": 0.1,
            "dam_accuracy": 90.0,
        },
    ]

    ranked = rank_sweep_rows(rows)
    assert ranked[0]["dam_accuracy"] == 90.0
    assert ranked[1]["dam_accuracy"] == 70.0
    assert ranked[-1]["retrieval_collapse"] is True


def test_sweep_layerwise_dam_configs_deterministic_and_complete():
    stored, probes = _tiny_feature_bundle()
    configs = [
        {
            "label": "cfg_a",
            "n": 2,
            "beta": 0.05,
            "alpha": 0.5,
            "lmbda": 0.0,
            "steps_multiplier": 5,
            "retrieval_backend": "numpy",
            "trace_every": 0,
            "seed": 123,
        },
        {
            "label": "cfg_b",
            "n": 2,
            "beta": 0.2,
            "alpha": 0.2,
            "lmbda": 0.0,
            "steps_multiplier": 5,
            "retrieval_backend": "numpy",
            "trace_every": 0,
            "seed": 123,
        },
    ]

    first = sweep_layerwise_dam_configs(
        stored,
        probes,
        np.array([0, 1]),
        task_name="identification",
        configs=configs,
    )
    second = sweep_layerwise_dam_configs(
        stored,
        probes,
        np.array([0, 1]),
        task_name="identification",
        configs=configs,
    )

    assert first == second
    assert len(first) == 4
    for row in first:
        for key in (
            "task_name",
            "config_label",
            "layer",
            "baseline_accuracy",
            "dam_accuracy",
            "baseline_delta_accuracy",
            "avg_initial_sim",
            "avg_final_sim",
            "avg_delta_sim",
            "avg_retrieved_norm",
            "retrieved_norm_ratio",
            "retrieval_collapse",
            "retrieval_overshoot",
            "retrieval_calibrated",
            "winner_agreement",
        ):
            assert key in row
