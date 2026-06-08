from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils import cosine


VALID_METRICS = {"cosine", "dot", "euclidean"}


def load_feature_cache(npz_path: str | Path) -> dict[str, Any]:
    payload = np.load(npz_path, allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def extract_feature_bundle(cache_payload: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key.removeprefix("features_"): np.asarray(value, dtype=np.float32)
        for key, value in cache_payload.items()
        if key.startswith("features_")
    }


def subset_feature_bundle(
    features_by_layer: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        layer: np.asarray(feats[indices], dtype=np.float32)
        for layer, feats in features_by_layer.items()
    }


def _normalize_vector(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return vec / norm


def score_against_stored(
    query: np.ndarray,
    stored: np.ndarray,
    *,
    metric: str = "cosine",
    decision_noise_std: float = 0.0,
    decision_noise_rng: np.random.Generator | None = None,
) -> np.ndarray:
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}")

    query = np.asarray(query, dtype=np.float64).reshape(-1)
    stored = np.asarray(stored, dtype=np.float64)
    if stored.ndim != 2:
        raise ValueError("stored must be a 2D matrix")
    if stored.shape[1] != query.shape[0]:
        raise ValueError("query and stored feature dimensions must match")

    if metric == "dot":
        scores = stored @ query
    elif metric == "euclidean":
        scores = -np.linalg.norm(stored - query, axis=1)
    else:
        query_norm = np.linalg.norm(query) + 1e-12
        stored_norms = np.linalg.norm(stored, axis=1) + 1e-12
        scores = (stored @ query) / (stored_norms * query_norm)

    if decision_noise_std > 0.0:
        rng = decision_noise_rng or np.random.default_rng()
        scores = np.asarray(scores, dtype=np.float64) + rng.normal(
            0.0,
            decision_noise_std,
            size=np.asarray(scores).shape,
        )
    return scores


def find_nearest_by_metric(
    query: np.ndarray,
    stored: np.ndarray,
    *,
    metric: str = "cosine",
    decision_noise_std: float = 0.0,
    decision_noise_rng: np.random.Generator | None = None,
) -> int:
    return int(
        np.argmax(
            score_against_stored(
                query,
                stored,
                metric=metric,
                decision_noise_std=decision_noise_std,
                decision_noise_rng=decision_noise_rng,
            )
        )
    )


def _target_cosines(
    queries: np.ndarray,
    stored: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [cosine(query, stored[int(target_idx)]) for query, target_idx in zip(queries, targets)],
        dtype=np.float64,
    )


def _best_wrong_cosines(
    queries: np.ndarray,
    stored: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    for query, target_idx in zip(queries, targets):
        wrong_indices = [idx for idx in range(stored.shape[0]) if idx != int(target_idx)]
        if not wrong_indices:
            values.append(float("nan"))
            continue
        wrong_cos = [cosine(query, stored[idx]) for idx in wrong_indices]
        values.append(float(np.max(wrong_cos)))
    return np.asarray(values, dtype=np.float64)


def _nanmean_or_nan(values: np.ndarray) -> float:
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def evaluate_layerwise_baseline(
    stored_features: dict[str, np.ndarray],
    probe_features: dict[str, np.ndarray],
    ground_truth_idx: np.ndarray,
    *,
    metric: str = "cosine",
    normalize_query: bool = False,
    decision_noise_std: float = 0.0,
    decision_noise_seed: int | None = None,
) -> dict[str, dict[str, object]]:
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}")
    if not stored_features or not probe_features:
        raise ValueError("stored_features and probe_features must be non-empty")
    if set(stored_features) != set(probe_features):
        raise ValueError("stored_features and probe_features must have identical layer keys")

    targets = np.asarray(ground_truth_idx, dtype=np.int64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("ground_truth_idx must be a non-empty 1D array")

    results: dict[str, dict[str, object]] = {}
    for layer in sorted(stored_features):
        stored = np.asarray(stored_features[layer], dtype=np.float32)
        probes = np.asarray(probe_features[layer], dtype=np.float32)
        if stored.ndim != 2 or probes.ndim != 2:
            raise ValueError(f"{layer} features must be 2D matrices")
        if probes.shape[0] != targets.shape[0]:
            raise ValueError(f"{layer} probe count must match ground_truth_idx length")
        if stored.shape[1] != probes.shape[1]:
            raise ValueError(f"{layer} stored/probe feature dims must match")
        if np.any(targets < 0) or np.any(targets >= stored.shape[0]):
            raise ValueError(f"{layer} ground_truth_idx contains out-of-range values")

        winners: list[int] = []
        scored_queries = []
        for probe_idx, probe_vec in enumerate(probes):
            score_vec = _normalize_vector(probe_vec) if normalize_query else probe_vec
            scored_queries.append(score_vec)
            noise_rng = None
            if decision_noise_std > 0.0:
                seed_offset = 0 if decision_noise_seed is None else int(decision_noise_seed)
                noise_rng = np.random.default_rng(seed_offset + probe_idx)
            winners.append(
                find_nearest_by_metric(
                    score_vec,
                    stored,
                    metric=metric,
                    decision_noise_std=decision_noise_std,
                    decision_noise_rng=noise_rng,
                )
            )

        scored_queries_arr = np.asarray(scored_queries, dtype=np.float32)
        target_cos = _target_cosines(scored_queries_arr, stored, targets)
        best_wrong_cos = _best_wrong_cosines(scored_queries_arr, stored, targets)
        margins = target_cos - best_wrong_cos
        successes = np.asarray(winners, dtype=np.int64) == targets
        results[layer] = {
            "accuracy": float(np.mean(successes) * 100.0),
            "avg_target_sim": float(np.mean(target_cos)),
            "avg_best_wrong_sim": _nanmean_or_nan(best_wrong_cos),
            "avg_margin": _nanmean_or_nan(margins),
            "avg_sim": float(np.mean(target_cos)),
            "std_sim": float(np.std(target_cos)),
            "avg_probe_norm": float(np.mean(np.linalg.norm(probes, axis=1))),
            "avg_target_norm": float(np.mean(np.linalg.norm(stored[targets], axis=1))),
            "metric": metric,
            "decision_noise_std": float(decision_noise_std),
            "winners": winners,
        }

    return results
