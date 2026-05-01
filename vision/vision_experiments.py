from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from DAM import SilentDAM
from utils import cosine


VALID_METRICS = {"cosine", "dot", "euclidean"}
DEFAULT_COLLAPSE_THRESHOLD = 0.05
DEFAULT_OVERSHOOT_RATIO = 2.0


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
        return stored @ query
    if metric == "euclidean":
        return -np.linalg.norm(stored - query, axis=1)

    query_norm = np.linalg.norm(query) + 1e-12
    stored_norms = np.linalg.norm(stored, axis=1) + 1e-12
    return (stored @ query) / (stored_norms * query_norm)


def find_nearest_by_metric(
    query: np.ndarray,
    stored: np.ndarray,
    *,
    metric: str = "cosine",
) -> int:
    return int(np.argmax(score_against_stored(query, stored, metric=metric)))


def _target_cosines(
    queries: np.ndarray,
    stored: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [cosine(query, stored[int(target_idx)]) for query, target_idx in zip(queries, targets)],
        dtype=np.float64,
    )


def evaluate_layerwise_baseline(
    stored_features: dict[str, np.ndarray],
    probe_features: dict[str, np.ndarray],
    ground_truth_idx: np.ndarray,
    *,
    metric: str = "cosine",
    normalize_query: bool = False,
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
        for probe_vec in probes:
            score_vec = _normalize_vector(probe_vec) if normalize_query else probe_vec
            scored_queries.append(score_vec)
            winners.append(find_nearest_by_metric(score_vec, stored, metric=metric))

        scored_queries_arr = np.asarray(scored_queries, dtype=np.float32)
        target_cos = _target_cosines(scored_queries_arr, stored, targets)
        successes = np.asarray(winners, dtype=np.int64) == targets
        results[layer] = {
            "accuracy": float(np.mean(successes) * 100.0),
            "avg_sim": float(np.mean(target_cos)),
            "std_sim": float(np.std(target_cos)),
            "avg_probe_norm": float(np.mean(np.linalg.norm(probes, axis=1))),
            "avg_target_norm": float(np.mean(np.linalg.norm(stored[targets], axis=1))),
            "metric": metric,
            "winners": winners,
        }

    return results


def _run_layerwise_task(
    stored_features: dict[str, np.ndarray],
    probe_features: dict[str, np.ndarray],
    ground_truth_idx: np.ndarray,
    *,
    n: int,
    beta: float,
    alpha: float,
    lmbda: float,
    steps_multiplier: int,
    retrieval_backend: str,
    trace_every: int,
    seed: int,
    metric: str,
    normalize_retrieved: bool,
) -> dict[str, dict[str, object]]:
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}")
    if not stored_features or not probe_features:
        raise ValueError("stored_features and probe_features must be non-empty")

    stored_layers = set(stored_features)
    probe_layers = set(probe_features)
    if stored_layers != probe_layers:
        raise ValueError("stored_features and probe_features must have identical layer keys")

    targets = np.asarray(ground_truth_idx, dtype=np.int64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("ground_truth_idx must be a non-empty 1D array")

    results: dict[str, dict[str, object]] = {}
    for layer in sorted(stored_layers):
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

        dam = SilentDAM(stored, n=n, beta=beta, alpha=alpha, lmbda=lmbda)
        steps = int(steps_multiplier * stored.shape[1])
        rng = np.random.default_rng(seed)
        successes: list[bool] = []
        final_similarities: list[float] = []
        initial_similarities: list[float] = []
        retrieved_norms: list[float] = []
        probe_norms: list[float] = []
        target_norms: list[float] = []
        baseline_winners: list[int] = []
        dam_winners: list[int] = []

        for probe_vec, target_idx in zip(probes, targets):
            target_vec = stored[int(target_idx)]
            initial_similarities.append(float(cosine(probe_vec, target_vec)))
            probe_norms.append(float(np.linalg.norm(probe_vec)))
            target_norms.append(float(np.linalg.norm(target_vec)))
            baseline_winner = find_nearest_by_metric(probe_vec, stored, metric=metric)
            baseline_winners.append(baseline_winner)

            update_indices = rng.integers(0, stored.shape[1], size=steps, dtype=np.int64)
            retrieved, _, _, _ = dam.retrieve_differential(
                probe_vec,
                steps=steps,
                update_indices=update_indices,
                backend=retrieval_backend,
                trace_every=trace_every,
            )
            scored_retrieved = _normalize_vector(retrieved) if normalize_retrieved else retrieved
            retrieved_idx = find_nearest_by_metric(scored_retrieved, stored, metric=metric)
            dam_winners.append(retrieved_idx)
            successes.append(bool(retrieved_idx == int(target_idx)))
            final_similarities.append(float(cosine(scored_retrieved, target_vec)))
            retrieved_norms.append(float(np.linalg.norm(retrieved)))

        initial_arr = np.asarray(initial_similarities, dtype=np.float64)
        final_arr = np.asarray(final_similarities, dtype=np.float64)
        baseline_successes = np.asarray(baseline_winners, dtype=np.int64) == targets
        dam_winners_arr = np.asarray(dam_winners, dtype=np.int64)
        results[layer] = {
            "accuracy": float(np.mean(successes) * 100.0),
            "avg_sim": float(np.mean(final_arr)),
            "std_sim": float(np.std(final_arr)),
            "baseline_accuracy": float(np.mean(baseline_successes) * 100.0),
            "avg_initial_sim": float(np.mean(initial_arr)),
            "std_initial_sim": float(np.std(initial_arr)),
            "avg_delta_sim": float(np.mean(final_arr - initial_arr)),
            "avg_retrieved_norm": float(np.mean(retrieved_norms)),
            "avg_probe_norm": float(np.mean(probe_norms)),
            "avg_target_norm": float(np.mean(target_norms)),
            "winner_agreement": float(np.mean(dam_winners_arr == np.asarray(baseline_winners)) * 100.0),
            "metric": metric,
            "baseline_winners": baseline_winners,
            "dam_winners": dam_winners,
        }

    return results


def run_layerwise_identification(
    stored_features: dict[str, np.ndarray],
    probe_features: dict[str, np.ndarray],
    ground_truth_idx: np.ndarray,
    *,
    n: int = 4,
    beta: float = 0.01,
    alpha: float = 0.5,
    lmbda: float = 0.0,
    steps_multiplier: int = 20,
    retrieval_backend: str = "numba",
    trace_every: int = 0,
    seed: int = 0,
    metric: str = "cosine",
    normalize_retrieved: bool = False,
) -> dict[str, dict[str, object]]:
    return _run_layerwise_task(
        stored_features,
        probe_features,
        ground_truth_idx,
        n=n,
        beta=beta,
        alpha=alpha,
        lmbda=lmbda,
        steps_multiplier=steps_multiplier,
        retrieval_backend=retrieval_backend,
        trace_every=trace_every,
        seed=seed,
        metric=metric,
        normalize_retrieved=normalize_retrieved,
    )


def run_layerwise_generalization(
    stored_features: dict[str, np.ndarray],
    probe_features: dict[str, np.ndarray],
    ground_truth_idx: np.ndarray,
    *,
    n: int = 4,
    beta: float = 0.01,
    alpha: float = 0.5,
    lmbda: float = 0.0,
    steps_multiplier: int = 20,
    retrieval_backend: str = "numba",
    trace_every: int = 0,
    seed: int = 0,
    metric: str = "cosine",
    normalize_retrieved: bool = False,
) -> dict[str, dict[str, object]]:
    return _run_layerwise_task(
        stored_features,
        probe_features,
        ground_truth_idx,
        n=n,
        beta=beta,
        alpha=alpha,
        lmbda=lmbda,
        steps_multiplier=steps_multiplier,
        retrieval_backend=retrieval_backend,
        trace_every=trace_every,
        seed=seed,
        metric=metric,
        normalize_retrieved=normalize_retrieved,
    )


def is_retrieval_collapsed(
    avg_retrieved_norm: float,
    *,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> bool:
    return float(avg_retrieved_norm) < float(collapse_threshold)


def compute_retrieved_norm_ratio(
    avg_retrieved_norm: float,
    avg_target_norm: float,
    *,
    eps: float = 1e-12,
) -> float:
    return float(avg_retrieved_norm) / max(float(avg_target_norm), eps)


def is_retrieval_overshooting(
    retrieved_norm_ratio: float,
    *,
    overshoot_ratio: float = DEFAULT_OVERSHOOT_RATIO,
) -> bool:
    return float(retrieved_norm_ratio) > float(overshoot_ratio)


def is_retrieval_calibrated(
    avg_retrieved_norm: float,
    avg_target_norm: float,
    *,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
    overshoot_ratio: float = DEFAULT_OVERSHOOT_RATIO,
) -> bool:
    norm_ratio = compute_retrieved_norm_ratio(avg_retrieved_norm, avg_target_norm)
    return not is_retrieval_collapsed(avg_retrieved_norm, collapse_threshold=collapse_threshold) and not is_retrieval_overshooting(
        norm_ratio,
        overshoot_ratio=overshoot_ratio,
    )


def rank_sweep_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            not bool(row["retrieval_calibrated"]),
            bool(row["retrieval_collapse"]),
            bool(row["retrieval_overshoot"]),
            -float(row["avg_delta_sim"]),
            -float(row["dam_accuracy"]),
            abs(float(row["retrieved_norm_ratio"]) - 1.0),
        ),
    )


def sweep_layerwise_dam_configs(
    stored_features: dict[str, np.ndarray],
    probe_features: dict[str, np.ndarray],
    ground_truth_idx: np.ndarray,
    *,
    task_name: str,
    configs: list[dict[str, object]],
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
    overshoot_ratio: float = DEFAULT_OVERSHOOT_RATIO,
    normalize_query: bool = False,
) -> list[dict[str, object]]:
    if task_name not in {"identification", "generalization"}:
        raise ValueError("task_name must be 'identification' or 'generalization'")
    if not configs:
        raise ValueError("configs must be non-empty")

    runner = (
        run_layerwise_identification
        if task_name == "identification"
        else run_layerwise_generalization
    )

    rows: list[dict[str, object]] = []
    baseline_cache: dict[tuple[str, bool], dict[str, dict[str, object]]] = {}

    for idx, config in enumerate(configs):
        metric = str(config.get("metric", "cosine"))
        if metric not in VALID_METRICS:
            raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}")

        cache_key = (metric, bool(normalize_query))
        if cache_key not in baseline_cache:
            baseline_cache[cache_key] = evaluate_layerwise_baseline(
                stored_features,
                probe_features,
                ground_truth_idx,
                metric=metric,
                normalize_query=normalize_query,
            )

        run_results = runner(
            stored_features,
            probe_features,
            ground_truth_idx,
            n=int(config.get("n", 4)),
            beta=float(config.get("beta", 0.01)),
            alpha=float(config.get("alpha", 0.5)),
            lmbda=float(config.get("lmbda", 0.0)),
            steps_multiplier=int(config.get("steps_multiplier", 20)),
            retrieval_backend=str(config.get("retrieval_backend", "numba")),
            trace_every=int(config.get("trace_every", 0)),
            seed=int(config.get("seed", 0)),
            metric=metric,
            normalize_retrieved=bool(config.get("normalize_retrieved", False)),
        )

        baseline_results = baseline_cache[cache_key]
        for layer, metrics in run_results.items():
            avg_retrieved_norm = float(metrics["avg_retrieved_norm"])
            avg_target_norm = float(metrics["avg_target_norm"])
            retrieved_norm_ratio = compute_retrieved_norm_ratio(
                avg_retrieved_norm,
                avg_target_norm,
            )
            retrieval_collapse = is_retrieval_collapsed(
                avg_retrieved_norm,
                collapse_threshold=collapse_threshold,
            )
            retrieval_overshoot = is_retrieval_overshooting(
                retrieved_norm_ratio,
                overshoot_ratio=overshoot_ratio,
            )
            row = {
                "task_name": task_name,
                "config_index": idx,
                "config_label": str(config.get("label", f"{task_name}_{idx}")),
                "layer": layer,
                "n": int(config.get("n", 4)),
                "beta": float(config.get("beta", 0.01)),
                "alpha": float(config.get("alpha", 0.5)),
                "lmbda": float(config.get("lmbda", 0.0)),
                "steps_multiplier": int(config.get("steps_multiplier", 20)),
                "retrieval_backend": str(config.get("retrieval_backend", "numba")),
                "trace_every": int(config.get("trace_every", 0)),
                "seed": int(config.get("seed", 0)),
                "metric": metric,
                "normalize_retrieved": bool(config.get("normalize_retrieved", False)),
                "collapse_threshold": float(collapse_threshold),
                "overshoot_ratio": float(overshoot_ratio),
                "retrieval_collapse": retrieval_collapse,
                "retrieval_overshoot": retrieval_overshoot,
                "retrieval_calibrated": not retrieval_collapse and not retrieval_overshoot,
                "baseline_accuracy": float(baseline_results[layer]["accuracy"]),
                "baseline_avg_sim": float(baseline_results[layer]["avg_sim"]),
                "dam_accuracy": float(metrics["accuracy"]),
                "baseline_delta_accuracy": float(metrics["accuracy"]) - float(baseline_results[layer]["accuracy"]),
                "avg_initial_sim": float(metrics["avg_initial_sim"]),
                "avg_final_sim": float(metrics["avg_sim"]),
                "avg_delta_sim": float(metrics["avg_delta_sim"]),
                "avg_retrieved_norm": avg_retrieved_norm,
                "avg_probe_norm": float(metrics["avg_probe_norm"]),
                "avg_target_norm": avg_target_norm,
                "retrieved_norm_ratio": retrieved_norm_ratio,
                "winner_agreement": float(metrics["winner_agreement"]),
            }
            rows.append(row)

    return rank_sweep_rows(rows)
