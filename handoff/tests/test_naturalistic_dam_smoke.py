from pathlib import Path
import csv

import numpy as np

from vision.naturalistic_dam import (
    STAGE_A_CONFIGS,
    STAGE_B_CONFIGS,
    build_branch_pairs,
    classify_comparison_outcome,
    classify_frontier_shape,
    FrontierBranchSpec,
    get_dam_model_specs,
    get_animals_frontier_configs,
    get_hard_ident_tradeoff_configs,
    get_frontier_branch_specs,
    pareto_frontier_rows,
    probe_rdm_spearman,
    rowwise_cosine,
    select_anchor_layers,
    select_best_row,
    select_stage_b_focus_rows,
    should_trigger_stage_b,
    summarize_encoder_head_to_head,
)


def test_get_dam_model_specs_includes_clip():
    specs = get_dam_model_specs(include_clip=True)
    assert ("vit_base_patch16_224", "cls") in specs
    assert ("vit_base_patch16_clip_224.openai", "cls") in specs


def test_select_anchor_layers_picks_retrieval_and_alignment():
    rows = [
        {
            "model_name": "vit_base_patch16_224",
            "pooling": "cls",
            "layer": "layer_0",
            "gen_accuracy": "40.0",
            "human_rdm_spearman": "0.20",
        },
        {
            "model_name": "vit_base_patch16_224",
            "pooling": "cls",
            "layer": "layer_11",
            "gen_accuracy": "35.0",
            "human_rdm_spearman": "0.55",
        },
    ]
    anchors = select_anchor_layers(rows, (("vit_base_patch16_224", "cls"),))
    picked = anchors[("vit_base_patch16_224", "cls")]
    assert [anchor.anchor_kind for anchor in picked] == ["retrieval", "alignment"]
    assert [anchor.layer for anchor in picked] == ["layer_0", "layer_11"]


def test_probe_rdm_spearman_returns_perfect_alignment_for_matching_structure():
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.7, 0.3],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    similarity = np.asarray(
        [
            [1.0, 0.7, 0.0],
            [0.7, 1.0, 0.3],
            [0.0, 0.3, 1.0],
        ],
        dtype=np.float64,
    )
    corr = probe_rdm_spearman(features, similarity, np.asarray([0, 1, 2], dtype=np.int64))
    assert corr > 0.9


def test_rowwise_cosine_matches_expected_values():
    a = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    b = np.asarray([[1.0, 0.0], [1.0, -1.0]], dtype=np.float64)
    vals = rowwise_cosine(a, b)
    assert np.isclose(vals[0], 1.0)
    assert np.isclose(vals[1], 0.0)


def test_stage_b_triggers_only_without_qualifying_win():
    assert should_trigger_stage_b([{"qualifying_win": False}, {"qualifying_win": False}]) is True
    assert should_trigger_stage_b([{"qualifying_win": False}, {"qualifying_win": True}]) is False


def test_select_stage_b_focus_rows_prefers_top_retrieval_and_best_clip():
    rows = [
        {
            "category": "animals",
            "model_name": "vit_base_patch16_224",
            "pooling": "cls",
            "layer": "layer_11",
            "gen_accuracy_baseline": 55.0,
            "human_rdm_spearman_baseline": 0.30,
        },
        {
            "category": "animals",
            "model_name": "vit_base_patch16_224",
            "pooling": "mean_tokens",
            "layer": "layer_6",
            "gen_accuracy_baseline": 50.0,
            "human_rdm_spearman_baseline": 0.40,
        },
        {
            "category": "animals",
            "model_name": "vit_base_patch16_clip_224.openai",
            "pooling": "cls",
            "layer": "layer_11",
            "gen_accuracy_baseline": 48.0,
            "human_rdm_spearman_baseline": 0.60,
        },
    ]
    picked = select_stage_b_focus_rows(rows)
    keys = {(row["model_name"], row["pooling"]) for row in picked}
    assert ("vit_base_patch16_224", "cls") in keys
    assert ("vit_base_patch16_224", "mean_tokens") in keys
    assert ("vit_base_patch16_clip_224.openai", "cls") in keys


def test_stage_config_sets_are_non_empty_and_disjoint():
    assert len(STAGE_A_CONFIGS) > 0
    assert len(STAGE_B_CONFIGS) > 0
    stage_a = {tuple(sorted(config.items())) for config in STAGE_A_CONFIGS}
    stage_b = {tuple(sorted(config.items())) for config in STAGE_B_CONFIGS}
    assert stage_a.isdisjoint(stage_b)


def test_select_best_row_prefers_clean_win():
    rows = [
        {
            "qualifying_win": "False",
            "gen_accuracy_delta": "10.0",
            "gen_avg_margin_delta": "0.1",
            "probe_rdm_spearman_delta": "0.1",
            "gen_avg_human_similarity_regret_delta": "-0.1",
        },
        {
            "qualifying_win": "True",
            "gen_accuracy_delta": "2.0",
            "gen_avg_margin_delta": "0.2",
            "probe_rdm_spearman_delta": "0.2",
            "gen_avg_human_similarity_regret_delta": "-0.02",
        },
    ]
    picked = select_best_row(rows)
    assert picked["qualifying_win"] == "True"


def test_classify_comparison_outcome_labels_clean_and_retrieval_only():
    vit = {"qualifying_win": "True", "gen_accuracy_delta": "2.0", "probe_rdm_spearman_delta": "0.1"}
    clip = {"qualifying_win": "False", "gen_accuracy_delta": "1.0", "probe_rdm_spearman_delta": "-0.2"}
    assert classify_comparison_outcome(vit, clip) == "vit_clean_win"
    assert classify_comparison_outcome(None, clip) == "retrieval_only_gain"


def test_build_branch_pairs_matches_vit_and_clip_roles():
    rows = [
        {"category": "animals", "pooling": "cls", "anchor_kind": "retrieval", "model_name": "vit_base_patch16_224", "layer": "layer_11"},
        {"category": "animals", "pooling": "cls", "anchor_kind": "retrieval", "model_name": "vit_base_patch16_clip_224.openai", "layer": "layer_11"},
        {"category": "animals", "pooling": "cls", "anchor_kind": "alignment", "model_name": "vit_base_patch16_224", "layer": "layer_6"},
        {"category": "animals", "pooling": "cls", "anchor_kind": "alignment", "model_name": "vit_base_patch16_clip_224.openai", "layer": "layer_11"},
    ]
    pairs = build_branch_pairs(rows)
    assert len(pairs) == 2
    assert {(pair.anchor_kind, pair.vit_layer, pair.clip_layer) for pair in pairs} == {
        ("retrieval", "layer_11", "layer_11"),
        ("alignment", "layer_6", "layer_11"),
    }


def test_summarize_encoder_head_to_head_writes_artifacts(tmp_path: Path):
    fieldnames = [
        "category",
        "pooling",
        "anchor_kind",
        "model_name",
        "layer",
        "qualifying_win",
        "gen_accuracy_delta",
        "gen_avg_margin_delta",
        "probe_rdm_spearman_delta",
        "gen_avg_human_similarity_regret_delta",
        "dam_n",
        "dam_beta",
        "dam_alpha",
        "dam_steps_multiplier",
    ]
    rows = [
        {
            "category": "animals",
            "pooling": "cls",
            "anchor_kind": "retrieval",
            "model_name": "vit_base_patch16_224",
            "layer": "layer_11",
            "qualifying_win": "True",
            "gen_accuracy_delta": "5.0",
            "gen_avg_margin_delta": "0.1",
            "probe_rdm_spearman_delta": "0.2",
            "gen_avg_human_similarity_regret_delta": "-0.02",
            "dam_n": "2",
            "dam_beta": "0.05",
            "dam_alpha": "0.05",
            "dam_steps_multiplier": "1",
        },
        {
            "category": "animals",
            "pooling": "cls",
            "anchor_kind": "retrieval",
            "model_name": "vit_base_patch16_clip_224.openai",
            "layer": "layer_11",
            "qualifying_win": "False",
            "gen_accuracy_delta": "3.0",
            "gen_avg_margin_delta": "0.05",
            "probe_rdm_spearman_delta": "0.05",
            "gen_avg_human_similarity_regret_delta": "-0.01",
            "dam_n": "2",
            "dam_beta": "0.1",
            "dam_alpha": "0.05",
            "dam_steps_multiplier": "1",
        },
    ]
    with (tmp_path / "animals_combined.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_encoder_head_to_head(tmp_path, ["animals"])
    assert summary["categories"] == ["animals"]
    assert len(summary["comparison_rows"]) == 1
    assert summary["comparison_rows"][0]["outcome"] == "vit_clean_win"
    assert (tmp_path / "vit_vs_clip_head_to_head.json").exists()


def test_get_animals_frontier_configs_has_expected_range():
    configs = get_animals_frontier_configs()
    assert len(configs) == 180
    assert configs[0]["lmbda"] == 0.0
    assert {config["steps_multiplier"] for config in configs} == {1, 2, 3, 5, 8}


def test_get_hard_ident_tradeoff_configs_has_expected_range():
    configs = get_hard_ident_tradeoff_configs()
    assert len(configs) == 480
    assert {config["n"] for config in configs} == {2, 4}
    assert {config["beta"] for config in configs} == {0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0}


def test_get_frontier_branch_specs_targets_animals_and_fruits_control():
    specs = get_frontier_branch_specs()
    assert specs == (
        FrontierBranchSpec(
            category="animals",
            model_name="vit_base_patch16_224",
            pooling="cls",
            layer="layer_11",
            split_mode="random_storage_curve",
            storage_sizes=(40, 80),
            n_seeds=5,
        ),
        FrontierBranchSpec(
            category="fruits",
            model_name="vit_base_patch16_224",
            pooling="cls",
            layer="layer_11",
            split_mode="balanced_exemplar_folds",
            storage_sizes=None,
            n_seeds=3,
        ),
    )


def test_pareto_frontier_rows_filters_dominated_points():
    rows = [
        {"gen_accuracy_delta": 1.0, "probe_rdm_spearman_delta": 0.1, "gen_avg_human_similarity_regret_delta": -0.01},
        {"gen_accuracy_delta": 2.0, "probe_rdm_spearman_delta": 0.2, "gen_avg_human_similarity_regret_delta": -0.02},
        {"gen_accuracy_delta": 1.5, "probe_rdm_spearman_delta": 0.05, "gen_avg_human_similarity_regret_delta": 0.0},
    ]
    frontier = pareto_frontier_rows(
        rows,
        x_key="gen_accuracy_delta",
        y_key="probe_rdm_spearman_delta",
        z_key="gen_avg_human_similarity_regret_delta",
        maximize_z=False,
    )
    assert rows[1] in frontier
    assert rows[0] not in frontier


def test_classify_frontier_shape_covers_main_labels():
    no_useful = classify_frontier_shape([{"qualifying_win": False}])
    assert no_useful == "no_useful_dam_regime"

    hard = classify_frontier_shape(
        [
            {"qualifying_win": True, "gen_accuracy_delta": 3.0, "probe_rdm_spearman_delta": -0.1},
        ]
    )
    assert hard == "hard_tradeoff"

    no_tradeoff = classify_frontier_shape(
        [
            {"qualifying_win": True, "gen_accuracy_delta": 2.0, "probe_rdm_spearman_delta": 0.1},
            {"qualifying_win": True, "gen_accuracy_delta": 3.0, "probe_rdm_spearman_delta": 0.2},
        ]
    )
    assert no_tradeoff == "no_tradeoff"
