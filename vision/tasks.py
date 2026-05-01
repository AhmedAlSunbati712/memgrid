from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from vision.image_generator import StimulusRecord


@dataclass(frozen=True)
class TaskBatch:
    stored_indices: np.ndarray
    probe_indices: np.ndarray
    ground_truth_idx: np.ndarray
    stored_ids: tuple[str, ...]
    probe_ids: tuple[str, ...]


def _normalize_indices(indices: Sequence[int] | None, total: int) -> np.ndarray:
    if indices is None:
        return np.arange(total, dtype=np.int64)

    out = np.asarray(indices, dtype=np.int64)
    if out.ndim != 1:
        raise ValueError("indices must be a 1D sequence")
    if out.size == 0:
        raise ValueError("indices must be non-empty")
    if np.any(out < 0) or np.any(out >= total):
        raise ValueError("indices out of bounds")
    if len(np.unique(out)) != len(out):
        raise ValueError("indices must be unique")
    return out


def build_balanced_splits_by_color(
    metadata: Sequence[StimulusRecord],
    n_stored_per_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_stored_per_color <= 0:
        raise ValueError("n_stored_per_color must be positive")
    if not metadata:
        raise ValueError("metadata must be non-empty")

    color_to_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(metadata):
        color_to_indices.setdefault(record.color_name, []).append(idx)

    stored: list[int] = []
    probes: list[int] = []
    for color_name in sorted(color_to_indices):
        color_indices = color_to_indices[color_name]
        if len(color_indices) <= n_stored_per_color:
            raise ValueError(
                f"Color '{color_name}' needs more than {n_stored_per_color} samples "
                "to leave at least one probe item."
            )
        stored.extend(color_indices[:n_stored_per_color])
        probes.extend(color_indices[n_stored_per_color:])

    return np.asarray(stored, dtype=np.int64), np.asarray(probes, dtype=np.int64)


def build_identification_task(
    metadata: Sequence[StimulusRecord],
    stored_indices: Sequence[int] | None = None,
    probe_indices: Sequence[int] | None = None,
) -> TaskBatch:
    if not metadata:
        raise ValueError("metadata must be non-empty")

    stored = _normalize_indices(stored_indices, len(metadata))
    probe = _normalize_indices(probe_indices, len(metadata)) if probe_indices is not None else stored.copy()

    stored_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(stored)}
    if any(int(idx) not in stored_lookup for idx in probe):
        raise ValueError("Identification probes must reference items present in the stored set.")

    ground_truth = np.asarray([stored_lookup[int(idx)] for idx in probe], dtype=np.int64)
    stored_ids = tuple(metadata[int(idx)].stimulus_id for idx in stored)
    probe_ids = tuple(metadata[int(idx)].stimulus_id for idx in probe)
    return TaskBatch(
        stored_indices=stored,
        probe_indices=probe,
        ground_truth_idx=ground_truth,
        stored_ids=stored_ids,
        probe_ids=probe_ids,
    )


def synthetic_metadata_distance(
    query: StimulusRecord,
    candidate: StimulusRecord,
    *,
    color_weight: float = 1.0,
    position_weight: float = 1.0,
) -> float:
    if color_weight < 0 or position_weight < 0:
        raise ValueError("weights must be non-negative")

    query_max = max(query.image_size - query.square_size, 1)
    candidate_max = max(candidate.image_size - candidate.square_size, 1)
    norm = float(max(query_max, candidate_max))

    color_penalty = 0.0 if query.color_name == candidate.color_name else float(color_weight)
    dx = (query.x - candidate.x) / norm
    dy = (query.y - candidate.y) / norm
    position_penalty = float(position_weight) * float(np.hypot(dx, dy))
    return color_penalty + position_penalty


def build_generalization_task_synthetic(
    metadata: Sequence[StimulusRecord],
    stored_indices: Sequence[int] | None = None,
    probe_indices: Sequence[int] | None = None,
    *,
    color_weight: float = 1.0,
    position_weight: float = 1.0,
) -> TaskBatch:
    if not metadata:
        raise ValueError("metadata must be non-empty")

    if stored_indices is None and probe_indices is None:
        stored, probe = build_balanced_splits_by_color(metadata, n_stored_per_color=1)
    else:
        stored = _normalize_indices(stored_indices, len(metadata))
        if probe_indices is None:
            stored_set = set(int(idx) for idx in stored)
            probe = np.asarray(
                [idx for idx in range(len(metadata)) if idx not in stored_set],
                dtype=np.int64,
            )
            if probe.size == 0:
                raise ValueError("No probe indices remain after excluding the stored set.")
        else:
            probe = _normalize_indices(probe_indices, len(metadata))

    stored_set = set(int(idx) for idx in stored)
    if any(int(idx) in stored_set for idx in probe):
        raise ValueError("Generalization probes must be disjoint from the stored set.")

    ground_truth = []
    for probe_idx in probe:
        probe_record = metadata[int(probe_idx)]
        distances = [
            synthetic_metadata_distance(
                probe_record,
                metadata[int(stored_idx)],
                color_weight=color_weight,
                position_weight=position_weight,
            )
            for stored_idx in stored
        ]
        ground_truth.append(int(np.argmin(distances)))

    stored_ids = tuple(metadata[int(idx)].stimulus_id for idx in stored)
    probe_ids = tuple(metadata[int(idx)].stimulus_id for idx in probe)
    return TaskBatch(
        stored_indices=stored,
        probe_indices=probe,
        ground_truth_idx=np.asarray(ground_truth, dtype=np.int64),
        stored_ids=stored_ids,
        probe_ids=probe_ids,
    )
