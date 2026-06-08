from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.run_animals_image_transformer_similarity_ordered import (
    DAM_ORDERS,
    SETTING_SPECS,
    _classify_tradeoff,
    _mean_pairwise_cosine_stats,
)
from vision.run_animals_image_transformer_similarity_ordered_fixed import _distance_ranked


def test_setting_specs_cover_expected_animals_similarity_regimes():
    names = [setting.name for setting in SETTING_SPECS]
    assert names == [
        "easy_s40",
        "easy_s80",
        "easy_s100",
        "occ50_s40",
        "occ50_s80",
        "occ50_s100",
        "occ50_dnoise001_s40",
        "occ50_dnoise001_s80",
        "occ50_dnoise001_s100",
    ]


def test_dam_orders_match_requested_energy_powers():
    assert DAM_ORDERS == (2, 4, 6, 8)


def test_pairwise_cosine_stats_smoke():
    stored = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
        ],
        dtype=np.float64,
    )
    sim_mean, dist_mean, sim_std, dist_std = _mean_pairwise_cosine_stats(stored)
    assert np.isclose(sim_mean, (0.0 + (1.0 / np.sqrt(2.0)) + (1.0 / np.sqrt(2.0))) / 3.0)
    assert np.isclose(dist_mean, 1.0 - sim_mean)
    assert sim_std >= 0.0
    assert dist_std >= 0.0


def test_tradeoff_classification_smoke():
    assert _classify_tradeoff(0.5, "middle") == "tradeoff_supported"
    assert _classify_tradeoff(0.5, "high") == "tradeoff_partial"
    assert _classify_tradeoff(0.0, "middle") == "tradeoff_partial"
    assert _classify_tradeoff(0.0, "high") == "tradeoff_not_supported"


def test_distance_ranking_uses_pairwise_distance_then_layer_index():
    rows = [
        {"setting_name": "s", "model_name": "m", "layer": "layer_4", "mean_pairwise_cosine_distance_mean": "0.2"},
        {"setting_name": "s", "model_name": "m", "layer": "layer_1", "mean_pairwise_cosine_distance_mean": "0.1"},
        {"setting_name": "s", "model_name": "m", "layer": "layer_3", "mean_pairwise_cosine_distance_mean": "0.2"},
    ]
    ranked = _distance_ranked(rows, with_dam_n=False)
    ranked.sort(key=lambda row: int(row["distance_rank"]))
    assert [row["layer"] for row in ranked] == ["layer_1", "layer_3", "layer_4"]
    assert [int(row["distance_rank"]) for row in ranked] == [0, 1, 2]
