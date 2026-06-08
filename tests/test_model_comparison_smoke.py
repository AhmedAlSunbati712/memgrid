from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.model_comparison import (
    DEFAULT_MODEL_NAMES,
    build_model_comparison_configs,
    model_poolings,
    summarize_model_rows,
)


def test_model_pooling_policy_matches_family():
    assert model_poolings("vit_base_patch16_224") == ("cls", "mean_tokens")
    assert model_poolings("vit_base_patch16_clip_224.openai") == ("cls", "mean_tokens")
    assert model_poolings("convnext_tiny") == ("auto",)


def test_build_model_comparison_configs_covers_all_models_and_storage_levels():
    configs = build_model_comparison_configs()

    assert configs
    assert {config.model_name for config in configs} == set(DEFAULT_MODEL_NAMES)
    assert "vit_base_patch16_clip_224.openai" in DEFAULT_MODEL_NAMES
    assert {config.n_stored_per_color for config in configs} == {2, 4, 6}
    assert {config.task_mode for config in configs} == {"mixed_color_position", "color_only", "position_only"}


def test_summarize_model_rows_smoke():
    rows = [
        {
            "model_name": "vit_base_patch16_224",
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 5.0,
            "ident_max_shift": 2,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 2,
            "ident_accuracy": 55.0,
            "ident_avg_margin": 0.02,
            "gen_accuracy": 45.0,
            "gen_avg_margin": 0.01,
            "avg_margin": 0.015,
        },
        {
            "model_name": "vit_base_patch16_224",
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 5.0,
            "ident_max_shift": 2,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 4,
            "ident_accuracy": 60.0,
            "ident_avg_margin": 0.03,
            "gen_accuracy": 65.0,
            "gen_avg_margin": 0.02,
            "avg_margin": 0.025,
        },
        {
            "model_name": "vit_base_patch16_224",
            "task_mode": "mixed_color_position",
            "pooling": "cls",
            "square_size": 56,
            "ident_noise_std": 5.0,
            "ident_max_shift": 2,
            "color_only_jitter_std": 0.0,
            "layer": "layer_0",
            "n_stored_per_color": 6,
            "ident_accuracy": 75.0,
            "ident_avg_margin": 0.04,
            "gen_accuracy": 50.0,
            "gen_avg_margin": 0.01,
            "avg_margin": 0.025,
        },
    ]

    summary = summarize_model_rows(rows)
    assert summary["models"] == ["vit_base_patch16_224"]
    assert summary["n_rows"] == 3
    assert summary["best_identification"]["vit_base_patch16_224"]["ident_accuracy"] == 75.0
    assert summary["best_generalization"]["vit_base_patch16_224"]["gen_accuracy"] == 65.0
