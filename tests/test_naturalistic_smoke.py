from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.io_naturalistic import build_leave_one_exemplar_out_folds, load_naturalistic_category
from vision.tasks import build_generalization_task_human_similarity


def test_load_naturalistic_category_fruits_integrity():
    category = load_naturalistic_category("fruits")

    assert category.category == "fruits"
    assert len(category.filenames) == 120
    assert category.similarity_matrix.shape == (120, 120)
    assert category.integrity["is_symmetric"] is True
    assert category.integrity["diag_mean"] == 1.0
    assert category.integrity["n_concepts"] == 40


def test_load_naturalistic_category_vegetables_trimmed_to_three_exemplars():
    category = load_naturalistic_category("vegetables", required_exemplars=3)

    assert len(category.filenames) == 117
    assert category.similarity_matrix.shape == (117, 117)
    assert category.integrity["n_concepts"] == 39
    assert category.integrity["required_exemplars"] == 3
    assert category.integrity["dropped_concepts"] == ["taisoi", "tatsoi"]


def test_leave_one_exemplar_out_folds_are_balanced_for_fruits():
    category = load_naturalistic_category("fruits")
    folds = build_leave_one_exemplar_out_folds(category.concepts)

    assert len(folds) == 3
    held_out_all = set()
    for stored, probes in folds:
        assert len(stored) == 80
        assert len(probes) == 40
        assert set(stored).isdisjoint(set(probes))
        held_out_all.update(int(idx) for idx in probes)
    assert held_out_all == set(range(120))


def test_leave_one_exemplar_out_folds_are_balanced_for_trimmed_vegetables():
    category = load_naturalistic_category("vegetables", required_exemplars=3)
    folds = build_leave_one_exemplar_out_folds(category.concepts)

    assert len(folds) == 3
    held_out_all = set()
    for stored, probes in folds:
        assert len(stored) == 78
        assert len(probes) == 39
        assert set(stored).isdisjoint(set(probes))
        held_out_all.update(int(idx) for idx in probes)
    assert held_out_all == set(range(117))


def test_build_generalization_task_human_similarity_uses_max_similarity_stored_target():
    item_ids = ("a1", "a2", "b1", "probe")
    sim = np.array(
        [
            [1.0, 0.9, 0.1, 0.8],
            [0.9, 1.0, 0.2, 0.95],
            [0.1, 0.2, 1.0, 0.3],
            [0.8, 0.95, 0.3, 1.0],
        ],
        dtype=np.float64,
    )
    batch = build_generalization_task_human_similarity(item_ids, sim, stored_indices=[0, 1, 2], probe_indices=[3])

    assert tuple(batch.stored_ids) == ("a1", "a2", "b1")
    assert tuple(batch.probe_ids) == ("probe",)
    assert np.array_equal(batch.ground_truth_idx, np.array([1]))
