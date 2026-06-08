from pathlib import Path
import sys
import csv
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.run_naturalistic_baseline import (
    BASELINE_MODEL_SPECS,
    CLIP_MODEL_SPECS,
    _random_storage_splits,
    _corruption_label,
    _summarize_integrity,
    get_model_specs,
    summarize_cross_category,
)
from vision.run_naturalistic_corruption_sweep import summarize_corruption_sweep


def test_naturalistic_integrity_summary_has_expected_categories():
    summary = _summarize_integrity("datasets_peterson.pkl", ".")

    assert set(summary) == {"animals", "fruits", "vegetables"}
    assert summary["fruits"]["n_images"] == 120
    assert summary["fruits"]["matrix_shape"] == [120, 120]
    assert summary["fruits"]["missing_on_disk"] == []


def test_cross_category_summary_smoke(tmp_path: Path):
    fieldnames = [
        "category",
        "model_name",
        "pooling",
        "layer",
        "ident_accuracy",
        "gen_accuracy",
        "gen_avg_margin",
        "gen_avg_retrieved_human_similarity",
        "gen_avg_human_similarity_regret",
        "gen_same_concept_accuracy",
        "human_rdm_spearman",
        "chance_accuracy",
    ]
    rows = [
        {
            "category": "fruits",
            "model_name": "vit_base_patch16_224",
            "pooling": "cls",
            "layer": "layer_11",
            "ident_accuracy": 100.0,
            "gen_accuracy": 40.0,
            "gen_avg_margin": -0.01,
            "gen_avg_retrieved_human_similarity": 0.7,
            "gen_avg_human_similarity_regret": 0.1,
            "gen_same_concept_accuracy": 80.0,
            "human_rdm_spearman": 0.3,
            "chance_accuracy": 1.25,
        },
        {
            "category": "vegetables",
            "model_name": "vit_base_patch16_224",
            "pooling": "cls",
            "layer": "layer_11",
            "ident_accuracy": 100.0,
            "gen_accuracy": 35.0,
            "gen_avg_margin": -0.02,
            "gen_avg_retrieved_human_similarity": 0.65,
            "gen_avg_human_similarity_regret": 0.12,
            "gen_same_concept_accuracy": 75.0,
            "human_rdm_spearman": 0.28,
            "chance_accuracy": 1.2820512820512822,
        },
    ]
    for category in ("fruits", "vegetables"):
        with (tmp_path / f"{category}_combined.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([row for row in rows if row["category"] == category])

    summary = summarize_cross_category(tmp_path, ["fruits", "vegetables"])
    assert summary["categories"] == ["fruits", "vegetables"]
    assert summary["best_generalization"]["fruits"]["gen_accuracy"] == 40.0
    assert summary["best_generalization"]["vegetables"]["gen_accuracy"] == 35.0
    assert len(summary["category_deltas"]) == 1
    payload = json.loads((tmp_path / "fruits_vs_vegetables_summary.json").read_text())
    assert payload["categories"] == ["fruits", "vegetables"]


def test_random_storage_splits_are_deterministic_and_disjoint():
    splits_a = _random_storage_splits(120, storage_sizes=[40, 80], n_seeds=2, seed=0)
    splits_b = _random_storage_splits(120, storage_sizes=[40, 80], n_seeds=2, seed=0)

    assert len(splits_a) == 4
    assert len(splits_b) == 4
    for (label_a, split_seed_a, stored_a, probes_a), (label_b, split_seed_b, stored_b, probes_b) in zip(splits_a, splits_b):
        assert label_a == label_b
        assert split_seed_a == split_seed_b
        assert set(stored_a).isdisjoint(set(probes_a))
        assert len(stored_a) in {40, 80}
        assert len(probes_a) == 120 - len(stored_a)
        assert list(stored_a) == list(stored_b)
        assert list(probes_a) == list(probes_b)


def test_model_specs_can_include_clip():
    assert get_model_specs(include_clip=False) == BASELINE_MODEL_SPECS
    assert get_model_specs(include_clip=True) == CLIP_MODEL_SPECS
    assert ("vit_base_patch16_clip_224.openai", "cls") in CLIP_MODEL_SPECS


def test_corruption_label_smoke():
    label = _corruption_label(
        corruption_mode="noise_shift",
        noise_std=15.0,
        max_shift=8,
        occlusion_frac=0.3,
        mask_frac=0.3,
        n_chunks=6,
        affine_strength=0.05,
        decision_noise_std=0.01,
    )
    assert "noise15" in label
    assert "dnoise0.01" in label


def test_corruption_sweep_summary_finds_useful_regimes(tmp_path: Path):
    rows = [
        {
            "category": "animals",
            "corruption_label": "occ0.3_dnoise0",
            "corruption_mode": "occlusion",
            "decision_noise_std": 0.0,
            "ident_accuracy": "72.0",
            "gen_accuracy": "33.0",
            "human_rdm_spearman": "0.4",
        },
        {
            "category": "animals",
            "corruption_label": "occ0.3_dnoise0",
            "corruption_mode": "occlusion",
            "decision_noise_std": 0.0,
            "ident_accuracy": "68.0",
            "gen_accuracy": "31.0",
            "human_rdm_spearman": "0.35",
        },
        {
            "category": "fruits",
            "corruption_label": "noise30_shift16_dnoise0",
            "corruption_mode": "noise_shift",
            "decision_noise_std": 0.0,
            "ident_accuracy": "99.0",
            "gen_accuracy": "12.0",
            "human_rdm_spearman": "0.2",
        },
    ]
    summary = summarize_corruption_sweep(rows, tmp_path)
    assert summary["n_rows"] == 3
    assert summary["n_regimes"] == 2
    assert summary["useful_regimes"][0]["category"] == "animals"
