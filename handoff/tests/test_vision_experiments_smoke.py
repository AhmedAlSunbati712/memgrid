from pathlib import Path

import numpy as np
import pytest

from vision.vision_experiments import (
    evaluate_layerwise_baseline,
    find_nearest_by_metric,
    score_against_stored,
)
from vision.baseline_diagnostics import (
    evaluate_clean_reextract_sanity,
    evaluate_exact_image_sanity,
    summarize_perturbation_sweep,
)
from vision.feature_preprocess import LayerwisePreprocessor


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


def test_metric_helpers_expected_winners():
    stored = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([0.8, 0.2], dtype=np.float32)

    assert find_nearest_by_metric(query, stored, metric="cosine") == 0
    assert find_nearest_by_metric(query, stored, metric="dot") == 0
    assert find_nearest_by_metric(query, stored, metric="euclidean") == 0
    scores = score_against_stored(query, stored, metric="euclidean")
    assert scores[0] > scores[1]


def test_decision_noise_is_reproducible_and_can_flip_winner():
    stored = np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)
    query = np.array([0.995, 0.005], dtype=np.float32)
    no_noise = find_nearest_by_metric(query, stored, metric="cosine")
    with_noise_a = find_nearest_by_metric(
        query,
        stored,
        metric="cosine",
        decision_noise_std=0.01,
        decision_noise_rng=np.random.default_rng(7),
    )
    with_noise_b = find_nearest_by_metric(
        query,
        stored,
        metric="cosine",
        decision_noise_std=0.01,
        decision_noise_rng=np.random.default_rng(7),
    )

    assert no_noise in {0, 1}
    assert with_noise_a == with_noise_b


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
        assert np.isfinite(metrics["avg_target_sim"])
        assert np.isfinite(metrics["avg_best_wrong_sim"])
        assert np.isfinite(metrics["avg_margin"])
        assert metrics["avg_sim"] == metrics["avg_target_sim"]
        assert np.isfinite(metrics["std_sim"])
        assert np.isfinite(metrics["avg_probe_norm"])
        assert np.isfinite(metrics["avg_target_norm"])
        assert metrics["metric"] == "cosine"
        assert len(metrics["winners"]) == 2
        assert metrics["decision_noise_std"] == 0.0


def test_evaluate_layerwise_baseline_accepts_decision_noise():
    stored = {"layer_0": np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)}
    probes = {"layer_0": np.array([[0.995, 0.005]], dtype=np.float32)}
    results_a = evaluate_layerwise_baseline(
        stored,
        probes,
        np.array([0]),
        decision_noise_std=0.01,
        decision_noise_seed=123,
    )
    results_b = evaluate_layerwise_baseline(
        stored,
        probes,
        np.array([0]),
        decision_noise_std=0.01,
        decision_noise_seed=123,
    )
    assert results_a["layer_0"]["winners"] == results_b["layer_0"]["winners"]
    assert results_a["layer_0"]["decision_noise_std"] == 0.01


def test_baseline_diagnostics_exact_and_clean_reextract_sanity():
    stored, probes = _tiny_feature_bundle()

    exact = evaluate_exact_image_sanity(stored)
    clean = evaluate_clean_reextract_sanity(stored, stored)
    sweep = summarize_perturbation_sweep(stored, {"small_shift": probes}, np.array([0, 1]))

    for metrics in exact.values():
        assert metrics["accuracy"] == 100.0
        assert metrics["avg_target_sim"] == pytest.approx(1.0)
    for metrics in clean.values():
        assert metrics["accuracy"] == 100.0
        assert metrics["avg_target_sim"] == pytest.approx(1.0)
    assert set(sweep) == {"small_shift"}


def test_l2_only_preserves_target_similarity_better_than_zscore_on_noisy_probe():
    raw_features = {
        "layer_0": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.001],
                [1.0, -1.0, -0.001],
            ],
            dtype=np.float32,
        )
    }
    noisy_probe_raw = {
        "layer_0": np.array(
            [
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.1],
            ],
            dtype=np.float32,
        )
    }
    targets = np.array([0, 1], dtype=np.int64)

    l2_prep = LayerwisePreprocessor(use_zscore=False, l2_normalize=True)
    l2_all = l2_prep.fit_transform(raw_features)
    l2_probe = l2_prep.transform(noisy_probe_raw)
    l2_results = evaluate_layerwise_baseline(
        {"layer_0": l2_all["layer_0"][:2]},
        l2_probe,
        targets,
    )

    zscore_prep = LayerwisePreprocessor(use_zscore=True, l2_normalize=True)
    zscore_all = zscore_prep.fit_transform(raw_features)
    zscore_probe = zscore_prep.transform(noisy_probe_raw)
    zscore_results = evaluate_layerwise_baseline(
        {"layer_0": zscore_all["layer_0"][:2]},
        zscore_probe,
        targets,
    )

    assert l2_results["layer_0"]["avg_target_sim"] > 0.9
    assert zscore_results["layer_0"]["avg_target_sim"] < l2_results["layer_0"]["avg_target_sim"]
