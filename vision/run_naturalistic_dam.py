from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import (
    LayerwisePreprocessor,
    VisionEmbeddingWrapper,
    build_generalization_task_human_similarity,
    build_identification_task,
    evaluate_layerwise_baseline,
    load_naturalistic_category,
    subset_feature_bundle,
)
from vision.ident_corruption import IDENT_DECISION_NOISE_STD, corrupt_ident_probe
from vision.naturalistic_dam import (
    STAGE_A_CONFIGS,
    STAGE_B_CONFIGS,
    build_category_split_specs,
    evaluate_dam_trial,
    get_dam_model_specs,
    load_baseline_rows,
    select_anchor_layers,
    select_stage_b_focus_rows,
    should_trigger_stage_b,
    summarize_encoder_head_to_head,
    summarize_dam_cross_category,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DAM on the naturalistic benchmark.")
    parser.add_argument("--category", choices=("fruits", "vegetables", "animals"), default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/vision/naturalistic_dam")
    parser.add_argument("--baseline-dir", default="results/vision/naturalistic")
    parser.add_argument("--dataset-pkl", default="datasets_peterson.pkl")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--include-clip", action="store_true", default=True)
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numba")
    parser.add_argument("--model-spec", nargs="*", default=None, help="Optional model::pooling filters.")
    parser.add_argument("--max-stage-a-configs", type=int, default=None)
    parser.add_argument("--skip-stage-b", action="store_true")
    return parser.parse_args()


def _load_images(paths: tuple[Path, ...]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _rdm_spearman(features: np.ndarray, similarity_matrix: np.ndarray) -> float:
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    normalized = features / norms
    cosine_sim = normalized @ normalized.T
    feature_rdm = 1.0 - cosine_sim
    human_rdm = 1.0 - np.asarray(similarity_matrix, dtype=np.float64)
    tri = np.triu_indices_from(feature_rdm, k=1)
    corr = spearmanr(feature_rdm[tri], human_rdm[tri]).correlation
    return float(corr)


def _category_settings(category: str) -> dict[str, object]:
    if category == "fruits":
        return {
            "required_exemplars": None,
            "split_mode": "balanced_exemplar_folds",
            "storage_sizes": None,
            "n_seeds": 5,
        }
    if category == "vegetables":
        return {
            "required_exemplars": 3,
            "split_mode": "balanced_exemplar_folds",
            "storage_sizes": None,
            "n_seeds": 5,
        }
    return {
        "required_exemplars": None,
        "split_mode": "random_storage_curve",
        "storage_sizes": [40, 80],
        "n_seeds": 5,
    }


def _qualifying_win(row: dict[str, object]) -> bool:
    return (
        float(row["exact_accuracy_dam"]) == 100.0
        and float(row["clean_reextract_accuracy_dam"]) == 100.0
        and float(row["gen_accuracy_delta"]) > 0.0
        and float(row["gen_avg_margin_delta"]) > 0.0
        and float(row["ident_accuracy_delta"]) >= -2.0
        and float(row["gen_avg_human_similarity_regret_delta"]) <= 0.01
        and float(row["probe_rdm_spearman_delta"]) >= -0.03
    )


def _append_config_fields(row: dict[str, object], config: dict[str, float | int], stage: str) -> None:
    row["dam_stage"] = stage
    row["dam_n"] = int(config["n"])
    row["dam_beta"] = float(config["beta"])
    row["dam_alpha"] = float(config["alpha"])
    row["dam_lmbda"] = float(config["lmbda"])
    row["dam_steps_multiplier"] = int(config["steps_multiplier"])


def _baseline_generalization_extras(
    *,
    baseline_row: dict[str, object],
    task,
    similarity_matrix: np.ndarray,
    concepts: tuple[str, ...],
) -> dict[str, float]:
    winners = np.asarray(baseline_row["winners"], dtype=np.int64)
    stored_global = np.asarray(task.stored_indices, dtype=np.int64)
    probe_global = np.asarray(task.probe_indices, dtype=np.int64)
    retrieved_global = stored_global[winners]
    target_global = stored_global[np.asarray(task.ground_truth_idx, dtype=np.int64)]
    retrieved_sims = np.asarray(
        [similarity_matrix[int(probe), int(ret)] for probe, ret in zip(probe_global, retrieved_global)],
        dtype=np.float64,
    )
    target_sims = np.asarray(
        [similarity_matrix[int(probe), int(tgt)] for probe, tgt in zip(probe_global, target_global)],
        dtype=np.float64,
    )
    same_concept = np.asarray(
        [concepts[int(probe)] == concepts[int(ret)] for probe, ret in zip(probe_global, retrieved_global)],
        dtype=np.float64,
    )
    return {
        "gen_avg_retrieved_human_similarity_baseline": float(np.mean(retrieved_sims)),
        "gen_avg_human_similarity_regret_baseline": float(np.mean(target_sims - retrieved_sims)),
        "gen_same_concept_accuracy_baseline": float(np.mean(same_concept) * 100.0) if len(set(concepts)) > 1 else float("nan"),
    }


def run_naturalistic_dam(
    *,
    category: str,
    device: str,
    batch_size: int,
    seed: int,
    output_dir: Path,
    baseline_dir: Path,
    dataset_pkl: str,
    image_root: str,
    include_clip: bool,
    backend: str = "numba",
    model_specs_override: tuple[tuple[str, str], ...] | None = None,
    max_stage_a_configs: int | None = None,
    skip_stage_b: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = _category_settings(category)
    data = load_naturalistic_category(
        category,
        dataset_pkl=dataset_pkl,
        image_root=image_root,
        required_exemplars=settings["required_exemplars"],
    )
    images = _load_images(data.image_paths)
    model_specs = model_specs_override or get_dam_model_specs(include_clip=include_clip)
    stage_a_configs = STAGE_A_CONFIGS[: max_stage_a_configs or len(STAGE_A_CONFIGS)]
    baseline_rows = load_baseline_rows(baseline_dir / f"{category}_combined.csv")
    anchors = select_anchor_layers(baseline_rows, model_specs)
    split_specs = build_category_split_specs(
        category=category,
        concepts=data.concepts,
        n_items=len(data.filenames),
        split_mode=str(settings["split_mode"]),
        storage_sizes=settings["storage_sizes"],
        n_seeds=int(settings["n_seeds"]),
        seed=seed,
    )

    wrappers: dict[tuple[str, str], VisionEmbeddingWrapper] = {}
    rows: list[dict[str, object]] = []
    branch_rows: dict[tuple[str, str, str], list[dict[str, object]]] = {}

    for model_name, pooling in model_specs:
        if (model_name, pooling) not in anchors:
            continue
        wrapper = wrappers.setdefault(
            (model_name, pooling),
            VisionEmbeddingWrapper(model_name=model_name, pretrained=True, device=device, pooling=pooling),
        )
        raw_features = wrapper.extract(images, batch_size=batch_size)
        prep = LayerwisePreprocessor(use_zscore=False, l2_normalize=True)
        clean_features = prep.fit_transform(raw_features)
        clean_reextract_features = prep.transform(wrapper.extract(images, batch_size=batch_size))
        noisy_images = [
            corrupt_ident_probe(image, seed=seed + 1000 + idx)
            for idx, image in enumerate(images)
        ]
        noisy_features = prep.transform(wrapper.extract(noisy_images, batch_size=batch_size))
        rdms = {
            layer: _rdm_spearman(layer_features, data.similarity_matrix)
            for layer, layer_features in clean_features.items()
        }
        layer_names = {anchor.layer for anchor in anchors[(model_name, pooling)]}

        for split_label, split_seed, stored_indices, probe_indices in split_specs:
            ident_task = build_identification_task(
                list(data.filenames),
                stored_indices=stored_indices,
                probe_indices=stored_indices,
            )
            gen_task = build_generalization_task_human_similarity(
                data.filenames,
                data.similarity_matrix,
                stored_indices=stored_indices,
                probe_indices=probe_indices,
            )
            stored_feature_bundle = subset_feature_bundle(clean_features, ident_task.stored_indices)
            clean_probe_bundle = subset_feature_bundle(clean_reextract_features, ident_task.probe_indices)
            noisy_probe_bundle = subset_feature_bundle(noisy_features, ident_task.probe_indices)
            gen_probe_bundle = subset_feature_bundle(clean_features, gen_task.probe_indices)
            ident_baseline = evaluate_layerwise_baseline(
                stored_feature_bundle,
                noisy_probe_bundle,
                ident_task.ground_truth_idx,
                metric="cosine",
                decision_noise_std=IDENT_DECISION_NOISE_STD,
                decision_noise_seed=seed + split_seed + 8000,
            )
            gen_baseline = evaluate_layerwise_baseline(
                stored_feature_bundle,
                gen_probe_bundle,
                gen_task.ground_truth_idx,
                metric="cosine",
            )
            for layer in sorted(layer_names):
                gen_extra = _baseline_generalization_extras(
                    baseline_row=gen_baseline[layer],
                    task=gen_task,
                    similarity_matrix=data.similarity_matrix,
                    concepts=data.concepts,
                )
                branch_key = (model_name, pooling, layer)
                base_row = {
                    "category": category,
                    "split_label": split_label,
                    "split_seed": split_seed,
                    "split_mode": settings["split_mode"],
                    "model_name": model_name,
                    "pooling": pooling,
                    "layer": layer,
                    "anchor_kind": next(
                        anchor.anchor_kind for anchor in anchors[(model_name, pooling)] if anchor.layer == layer
                    ),
                    "n_stored": int(len(stored_indices)),
                    "n_probe": int(len(probe_indices)),
                    "chance_accuracy": 100.0 / len(stored_indices),
                    "ident_accuracy_baseline": ident_baseline[layer]["accuracy"],
                    "ident_avg_margin_baseline": ident_baseline[layer]["avg_margin"],
                    "gen_accuracy_baseline": gen_baseline[layer]["accuracy"],
                    "gen_avg_margin_baseline": gen_baseline[layer]["avg_margin"],
                    "gen_avg_retrieved_human_similarity_baseline": gen_extra["gen_avg_retrieved_human_similarity_baseline"],
                    "gen_avg_human_similarity_regret_baseline": gen_extra["gen_avg_human_similarity_regret_baseline"],
                    "gen_same_concept_accuracy_baseline": gen_extra["gen_same_concept_accuracy_baseline"],
                    "human_rdm_spearman_baseline": rdms[layer],
                }
                for config in stage_a_configs:
                    dam = evaluate_dam_trial(
                        layer=layer,
                        stored_feature_bundle=stored_feature_bundle,
                        clean_probe_bundle=clean_probe_bundle,
                        noisy_probe_bundle=noisy_probe_bundle,
                        gen_probe_bundle=gen_probe_bundle,
                        ident_ground_truth_idx=ident_task.ground_truth_idx,
                        gen_ground_truth_idx=gen_task.ground_truth_idx,
                        similarity_matrix=data.similarity_matrix,
                        gen_probe_indices=gen_task.probe_indices,
                        gen_stored_indices=gen_task.stored_indices,
                        concepts=data.concepts,
                        config=config,
                        seed=seed + split_seed * 10000,
                        decision_noise_std=IDENT_DECISION_NOISE_STD,
                    )
                    row = dict(base_row)
                    row.update(dam)
                    row["ident_accuracy_delta"] = float(row["ident_accuracy_dam"]) - float(row["ident_accuracy_baseline"])
                    row["gen_accuracy_delta"] = float(row["gen_accuracy_dam"]) - float(row["gen_accuracy_baseline"])
                    row["gen_avg_margin_delta"] = float(row["gen_avg_margin_dam"]) - float(row["gen_avg_margin_baseline"])
                    row["gen_avg_human_similarity_regret_delta"] = (
                        float(row["gen_avg_human_similarity_regret_dam"]) - float(row["gen_avg_human_similarity_regret_baseline"])
                    )
                    row["qualifying_win"] = _qualifying_win(row)
                    _append_config_fields(row, config, "A")
                    rows.append(row)
                    branch_rows.setdefault(branch_key, []).append(row)

    if not skip_stage_b and should_trigger_stage_b(rows):
        focus = select_stage_b_focus_rows(rows)
        focus_keys = {(str(row["category"]), str(row["model_name"]), str(row["pooling"]), str(row["layer"])) for row in focus}
        for focus_row in focus:
            model_name = str(focus_row["model_name"])
            pooling = str(focus_row["pooling"])
            layer = str(focus_row["layer"])
            key = (category, model_name, pooling, layer)
            if key not in focus_keys:
                continue
            wrapper = wrappers[(model_name, pooling)]
            raw_features = wrapper.extract(images, batch_size=batch_size)
            prep = LayerwisePreprocessor(use_zscore=False, l2_normalize=True)
            clean_features = prep.fit_transform(raw_features)
            clean_reextract_features = prep.transform(wrapper.extract(images, batch_size=batch_size))
            noisy_images = [
                corrupt_ident_probe(image, seed=seed + 1000 + idx)
                for idx, image in enumerate(images)
            ]
            noisy_features = prep.transform(wrapper.extract(noisy_images, batch_size=batch_size))
            for split_label, split_seed, stored_indices, probe_indices in split_specs:
                ident_task = build_identification_task(list(data.filenames), stored_indices=stored_indices, probe_indices=stored_indices)
                gen_task = build_generalization_task_human_similarity(
                    data.filenames,
                    data.similarity_matrix,
                    stored_indices=stored_indices,
                    probe_indices=probe_indices,
                )
                stored_feature_bundle = subset_feature_bundle(clean_features, ident_task.stored_indices)
                clean_probe_bundle = subset_feature_bundle(clean_reextract_features, ident_task.probe_indices)
                noisy_probe_bundle = subset_feature_bundle(noisy_features, ident_task.probe_indices)
                gen_probe_bundle = subset_feature_bundle(clean_features, gen_task.probe_indices)
                ident_baseline = evaluate_layerwise_baseline(
                    stored_feature_bundle,
                    noisy_probe_bundle,
                    ident_task.ground_truth_idx,
                    metric="cosine",
                    decision_noise_std=IDENT_DECISION_NOISE_STD,
                    decision_noise_seed=seed + split_seed + 8000,
                )
                gen_baseline = evaluate_layerwise_baseline(stored_feature_bundle, gen_probe_bundle, gen_task.ground_truth_idx, metric="cosine")
                gen_extra = _baseline_generalization_extras(
                    baseline_row=gen_baseline[layer],
                    task=gen_task,
                    similarity_matrix=data.similarity_matrix,
                    concepts=data.concepts,
                )
                for config in STAGE_B_CONFIGS:
                    dam = evaluate_dam_trial(
                        layer=layer,
                        stored_feature_bundle=stored_feature_bundle,
                        clean_probe_bundle=clean_probe_bundle,
                        noisy_probe_bundle=noisy_probe_bundle,
                        gen_probe_bundle=gen_probe_bundle,
                        ident_ground_truth_idx=ident_task.ground_truth_idx,
                        gen_ground_truth_idx=gen_task.ground_truth_idx,
                        similarity_matrix=data.similarity_matrix,
                        gen_probe_indices=gen_task.probe_indices,
                        gen_stored_indices=gen_task.stored_indices,
                        concepts=data.concepts,
                        config=config,
                        seed=seed + split_seed * 10000 + 500000,
                        decision_noise_std=IDENT_DECISION_NOISE_STD,
                    )
                    row = {
                        "category": category,
                        "split_label": split_label,
                        "split_seed": split_seed,
                        "split_mode": settings["split_mode"],
                        "model_name": model_name,
                        "pooling": pooling,
                        "layer": layer,
                        "anchor_kind": str(focus_row["anchor_kind"]),
                        "n_stored": int(len(stored_indices)),
                        "n_probe": int(len(probe_indices)),
                        "chance_accuracy": 100.0 / len(stored_indices),
                        "ident_accuracy_baseline": ident_baseline[layer]["accuracy"],
                        "ident_avg_margin_baseline": ident_baseline[layer]["avg_margin"],
                        "gen_accuracy_baseline": gen_baseline[layer]["accuracy"],
                        "gen_avg_margin_baseline": gen_baseline[layer]["avg_margin"],
                        "gen_avg_retrieved_human_similarity_baseline": gen_extra["gen_avg_retrieved_human_similarity_baseline"],
                        "gen_avg_human_similarity_regret_baseline": gen_extra["gen_avg_human_similarity_regret_baseline"],
                        "gen_same_concept_accuracy_baseline": gen_extra["gen_same_concept_accuracy_baseline"],
                        "human_rdm_spearman_baseline": float(focus_row["human_rdm_spearman_baseline"]),
                    }
                    row.update(dam)
                    row["ident_accuracy_delta"] = float(row["ident_accuracy_dam"]) - float(row["ident_accuracy_baseline"])
                    row["gen_accuracy_delta"] = float(row["gen_accuracy_dam"]) - float(row["gen_accuracy_baseline"])
                    row["gen_avg_margin_delta"] = float(row["gen_avg_margin_dam"]) - float(row["gen_avg_margin_baseline"])
                    row["gen_avg_human_similarity_regret_delta"] = (
                        float(row["gen_avg_human_similarity_regret_dam"]) - float(row["gen_avg_human_similarity_regret_baseline"])
                    )
                    row["qualifying_win"] = _qualifying_win(row)
                    _append_config_fields(row, config, "B")
                    rows.append(row)

    csv_path = output_dir / f"{category}_combined.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "category": category,
        "models": [f"{model_name}::{pooling}" for model_name, pooling in model_specs if (model_name, pooling) in anchors],
        "n_rows": len(rows),
        "sanity_all_pass": all(
            float(row["exact_accuracy_dam"]) == 100.0
            and float(row["clean_reextract_accuracy_dam"]) == 100.0
            and abs(float(row["exact_avg_target_sim_dam"]) - 1.0) < 1e-6
            and abs(float(row["clean_reextract_avg_target_sim_dam"]) - 1.0) < 1e-6
            for row in rows
        ),
        "n_qualifying_wins": int(sum(bool(row["qualifying_win"]) for row in rows)),
        "best_retrieval_gain": max(rows, key=lambda row: float(row["gen_accuracy_delta"])),
        "best_alignment_preserving_gain": max(
            rows,
            key=lambda row: float(row["gen_accuracy_delta"]) - max(0.0, -float(row["probe_rdm_spearman_delta"])),
        ),
    }
    wins = [row for row in rows if bool(row["qualifying_win"])]
    summary["best_qualifying_win"] = max(wins, key=lambda row: float(row["gen_accuracy_delta"])) if wins else None
    with (output_dir / f"{category}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    categories = [args.category] if args.category else ["fruits", "vegetables", "animals"]
    output_dir = Path(args.output_dir)
    summaries = []
    for category in categories:
        summaries.append(
            run_naturalistic_dam(
                category=category,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed,
                output_dir=output_dir,
                baseline_dir=Path(args.baseline_dir),
                dataset_pkl=args.dataset_pkl,
                image_root=args.image_root,
                include_clip=args.include_clip,
                backend=args.backend,
                model_specs_override=(
                    tuple(tuple(spec.split("::", 1)) for spec in args.model_spec) if args.model_spec else None
                ),
                max_stage_a_configs=args.max_stage_a_configs,
                skip_stage_b=args.skip_stage_b,
            )
        )
    print(json.dumps({"summaries": summaries}, indent=2))
    if len(categories) > 1:
        summarize_dam_cross_category(output_dir, categories)
        summarize_encoder_head_to_head(output_dir, categories)


if __name__ == "__main__":
    main()
