from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.image_generator import StimulusRecord
from vision.tasks import (
    build_balanced_splits_by_color,
    build_generalization_task_synthetic,
    build_identification_task,
    synthetic_metadata_distance,
)


def _record(stimulus_id, color_name, x, y):
    return StimulusRecord(
        stimulus_id=stimulus_id,
        color_name=color_name,
        color_rgb=(0, 0, 0),
        x=x,
        y=y,
        image_size=32,
        square_size=8,
    )


def test_build_balanced_splits_by_color_groups_deterministically():
    metadata = [
        _record("red_0", "red", 0, 0),
        _record("red_1", "red", 1, 0),
        _record("red_2", "red", 2, 0),
        _record("blue_0", "blue", 0, 1),
        _record("blue_1", "blue", 1, 1),
        _record("blue_2", "blue", 2, 1),
    ]
    stored, probes = build_balanced_splits_by_color(metadata, n_stored_per_color=1)

    assert np.array_equal(stored, np.array([3, 0]))
    assert np.array_equal(probes, np.array([4, 5, 1, 2]))


def test_build_identification_task_aligns_probe_targets():
    metadata = [
        _record("a", "red", 0, 0),
        _record("b", "red", 1, 0),
        _record("c", "blue", 0, 1),
    ]
    batch = build_identification_task(metadata, stored_indices=[2, 0], probe_indices=[0, 2])

    assert tuple(batch.stored_ids) == ("c", "a")
    assert tuple(batch.probe_ids) == ("a", "c")
    assert np.array_equal(batch.ground_truth_idx, np.array([1, 0]))


def test_synthetic_metadata_distance_prefers_same_color_when_positions_are_close():
    query = _record("q", "red", 10, 10)
    same_color = _record("same", "red", 11, 10)
    other_color = _record("other", "blue", 10, 10)

    assert synthetic_metadata_distance(query, same_color) < synthetic_metadata_distance(query, other_color)


def test_build_generalization_task_uses_metadata_distance_ground_truth():
    metadata = [
        _record("red_store", "red", 2, 2),
        _record("blue_store", "blue", 22, 22),
        _record("red_probe", "red", 3, 2),
        _record("blue_probe", "blue", 21, 22),
    ]
    batch = build_generalization_task_synthetic(
        metadata,
        stored_indices=[0, 1],
        probe_indices=[2, 3],
    )

    assert tuple(batch.stored_ids) == ("red_store", "blue_store")
    assert tuple(batch.probe_ids) == ("red_probe", "blue_probe")
    assert np.array_equal(batch.ground_truth_idx, np.array([0, 1]))


def test_position_only_generalization_ignores_color_when_color_weight_zero():
    metadata = [
        _record("red_far", "red", 0, 0),
        _record("blue_near", "blue", 20, 20),
        _record("red_probe", "red", 19, 20),
    ]
    batch = build_generalization_task_synthetic(
        metadata,
        stored_indices=[0, 1],
        probe_indices=[2],
        color_weight=0.0,
        position_weight=1.0,
    )

    assert np.array_equal(batch.ground_truth_idx, np.array([1]))
