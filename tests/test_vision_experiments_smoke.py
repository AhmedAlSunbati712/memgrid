from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.vision_experiments import (
    evaluate_layerwise_baseline,
    find_nearest_by_metric,
    score_against_stored,
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
        assert np.isfinite(metrics["avg_probe_norm"])
        assert np.isfinite(metrics["avg_target_norm"])
        assert metrics["metric"] == "cosine"
        assert len(metrics["winners"]) == 2
