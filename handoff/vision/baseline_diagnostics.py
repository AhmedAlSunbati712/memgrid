from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from vision.vision_experiments import evaluate_layerwise_baseline


FeatureBundle = dict[str, np.ndarray]


def evaluate_exact_image_sanity(
    stored_features: FeatureBundle,
    *,
    metric: str = "cosine",
) -> dict[str, dict[str, object]]:
    if not stored_features:
        raise ValueError("stored_features must be non-empty")

    first_layer = next(iter(stored_features.values()))
    targets = np.arange(first_layer.shape[0], dtype=np.int64)
    return evaluate_layerwise_baseline(stored_features, stored_features, targets, metric=metric)


def evaluate_clean_reextract_sanity(
    stored_features: FeatureBundle,
    clean_probe_features: FeatureBundle,
    ground_truth_idx: np.ndarray | None = None,
    *,
    metric: str = "cosine",
) -> dict[str, dict[str, object]]:
    if not stored_features:
        raise ValueError("stored_features must be non-empty")

    if ground_truth_idx is None:
        first_layer = next(iter(clean_probe_features.values()))
        ground_truth_idx = np.arange(first_layer.shape[0], dtype=np.int64)
    return evaluate_layerwise_baseline(stored_features, clean_probe_features, ground_truth_idx, metric=metric)


def summarize_perturbation_sweep(
    stored_features: FeatureBundle,
    probe_features_by_condition: Mapping[str, FeatureBundle],
    ground_truth_idx: np.ndarray,
    *,
    metric: str = "cosine",
) -> dict[str, dict[str, dict[str, object]]]:
    if not probe_features_by_condition:
        raise ValueError("probe_features_by_condition must be non-empty")

    return {
        condition: evaluate_layerwise_baseline(
            stored_features,
            probe_features,
            ground_truth_idx,
            metric=metric,
        )
        for condition, probe_features in probe_features_by_condition.items()
    }
