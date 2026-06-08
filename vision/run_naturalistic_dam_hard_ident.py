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
    corrupt_image,
    evaluate_layerwise_baseline,
    load_naturalistic_category,
    subset_feature_bundle,
)
from vision.naturalistic_dam import (
    STAGE_A_CONFIGS,
    build_category_split_specs,
    evaluate_dam_trial,
    get_hard_ident_tradeoff_configs,
)


HARD_IDENT_BRANCHES: tuple[dict[str, object], ...] = (
    {
        "category": "animals",
        "model_name": "vit_base_patch16_224",
        "pooling": "cls",
        "layer": "layer_11",
        "split_mode": "random_storage_curve",
        "storage_sizes": [40, 80],
        "n_seeds": 5,
        "corruption_mode": "occlusion",
        "occlusion_frac": 0.5,
        "decision_noise_levels": (0.0, 0.01),
        "required_exemplars": None,
    },
    {
        "category": "fruits",
        "model_name": "vit_base_patch16_224",
        "pooling": "cls",
        "layer": "layer_11",
        "split_mode": "balanced_exemplar_folds",
        "storage_sizes": None,
        "n_seeds": 3,
        "corruption_mode": "occlusion",
        "occlusion_frac": 0.5,
        "decision_noise_levels": (0.0, 0.01),
        "required_exemplars": None,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a narrow DAM comparison on the hard-identification regime.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/vision/naturalistic_dam_hard_ident")
    parser.add_argument("--dataset-pkl", default="datasets_peterson.pkl")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numba")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--config-set", choices=("stage_a", "tradeoff"), default="tradeoff")
    return parser.parse_args()


def _load_images(paths: tuple[Path, ...]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


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
    has_meaningful_concepts = len(set(concepts)) < len(concepts)
    return {
        "gen_avg_retrieved_human_similarity_baseline": float(np.mean(retrieved_sims)),
        "gen_avg_human_similarity_regret_baseline": float(np.mean(target_sims - retrieved_sims)),
        "gen_same_concept_accuracy_baseline": float(np.mean(same_concept) * 100.0) if has_meaningful_concepts else float("nan"),
    }


def _rdm_spearman(features: np.ndarray, similarity_matrix: np.ndarray) -> float:
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    normalized = features / norms
    cosine_sim = normalized @ normalized.T
    feature_rdm = 1.0 - cosine_sim
    human_rdm = 1.0 - np.asarray(similarity_matrix, dtype=np.float64)
    tri = np.triu_indices_from(feature_rdm, k=1)
    corr = spearmanr(feature_rdm[tri], human_rdm[tri]).correlation
    return float(corr)


def _qualifying_hard_ident_win(row: dict[str, object]) -> bool:
    return (
        float(row["exact_accuracy_dam"]) == 100.0
        and float(row["clean_reextract_accuracy_dam"]) == 100.0
        and float(row["ident_accuracy_baseline"]) < 95.0
        and float(row["gen_accuracy_delta"]) > 0.0
        and float(row["gen_avg_margin_delta"]) > 0.0
        and float(row["probe_rdm_spearman_delta"]) >= -0.03
    )


def _config_signature(row: dict[str, object]) -> tuple[object, ...]:
    return (
        str(row["category"]),
        float(row["decision_noise_std"]),
        int(row["dam_n"]),
        float(row["dam_beta"]),
        float(row["dam_alpha"]),
        float(row["dam_lmbda"]),
        int(row["dam_steps_multiplier"]),
    )


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def _aggregate_config_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_config_signature(row), []).append(row)

    out: list[dict[str, object]] = []
    for _, group_rows in grouped.items():
        sample = group_rows[0]
        agg: dict[str, object] = {
            "category": sample["category"],
            "model_name": sample["model_name"],
            "pooling": sample["pooling"],
            "layer": sample["layer"],
            "corruption_mode": sample["corruption_mode"],
            "occlusion_frac": sample["occlusion_frac"],
            "decision_noise_std": sample["decision_noise_std"],
            "dam_n": sample["dam_n"],
            "dam_beta": sample["dam_beta"],
            "dam_alpha": sample["dam_alpha"],
            "dam_lmbda": sample["dam_lmbda"],
            "dam_steps_multiplier": sample["dam_steps_multiplier"],
            "n_splits": len(group_rows),
            "n_qualifying_splits": int(sum(bool(row["qualifying_hard_ident_win"]) for row in group_rows)),
            "mean_qualifying_fraction": float(np.mean([float(bool(row["qualifying_hard_ident_win"])) for row in group_rows])),
        }
        metric_map = {
            "ident_accuracy_baseline": "mean_ident_accuracy_baseline",
            "ident_accuracy_dam": "mean_ident_accuracy_dam",
            "ident_accuracy_delta": "mean_ident_accuracy_delta",
            "gen_accuracy_baseline": "mean_gen_accuracy_baseline",
            "gen_accuracy_dam": "mean_gen_accuracy_dam",
            "gen_accuracy_delta": "mean_gen_accuracy_delta",
            "gen_avg_margin_delta": "mean_gen_avg_margin_delta",
            "gen_avg_human_similarity_regret_delta": "mean_gen_avg_human_similarity_regret_delta",
            "probe_rdm_spearman_delta": "mean_probe_rdm_spearman_delta",
            "cue_recovered_cosine_mean": "mean_cue_recovered_cosine",
            "cue_displacement_mean": "mean_cue_displacement",
            "target_pull_gain_mean": "mean_target_pull_gain",
        }
        for src, dst in metric_map.items():
            mean_val, std_val = _mean_std([float(row[src]) for row in group_rows])
            agg[dst] = mean_val
            agg[dst.replace("mean_", "std_")] = std_val
        out.append(agg)
    return out


def _score_tradeoff_row(row: dict[str, object]) -> float:
    return (
        float(row["mean_gen_accuracy_delta"])
        + float(row["mean_gen_avg_margin_delta"]) * 50.0
        - max(0.0, float(row["mean_cue_displacement"]) - 0.3) * 10.0
        + float(row["mean_probe_rdm_spearman_delta"]) * 5.0
    )


def _best_row(rows: list[dict[str, object]], key_fn) -> dict[str, object] | None:
    if not rows:
        return None
    return max(rows, key=key_fn)


def _summarize_recoverability(raw_rows: list[dict[str, object]]) -> dict[str, object]:
    split_groups: dict[tuple[str, float, str], list[dict[str, object]]] = {}
    for row in raw_rows:
        split_groups.setdefault(
            (str(row["category"]), float(row["decision_noise_std"]), str(row["split_label"])),
            [],
        ).append(row)
    per_split: list[dict[str, object]] = []
    for (category, decision_noise_std, split_label), group_rows in split_groups.items():
        best_gain_row = max(group_rows, key=lambda row: float(row["gen_accuracy_delta"]))
        per_split.append(
            {
                "category": category,
                "decision_noise_std": decision_noise_std,
                "split_label": split_label,
                "gen_accuracy_delta_best": float(best_gain_row["gen_accuracy_delta"]),
                "gen_avg_margin_baseline": float(best_gain_row["gen_avg_margin_baseline"]),
                "ident_avg_margin_baseline": float(best_gain_row["ident_avg_margin_baseline"]),
                "gen_avg_human_similarity_regret_baseline": float(best_gain_row["gen_avg_human_similarity_regret_baseline"]),
                "gen_accuracy_baseline": float(best_gain_row["gen_accuracy_baseline"]),
            }
        )

    summary: dict[str, object] = {"per_split_rows": per_split, "by_category": {}}
    for category in sorted({row["category"] for row in per_split}):
        subset = [row for row in per_split if row["category"] == category]
        if len(subset) < 2:
            continue
        x_margin = np.asarray([row["gen_avg_margin_baseline"] for row in subset], dtype=np.float64)
        x_ident = np.asarray([row["ident_avg_margin_baseline"] for row in subset], dtype=np.float64)
        x_regret = np.asarray([row["gen_avg_human_similarity_regret_baseline"] for row in subset], dtype=np.float64)
        x_acc = np.asarray([row["gen_accuracy_baseline"] for row in subset], dtype=np.float64)
        y_gain = np.asarray([row["gen_accuracy_delta_best"] for row in subset], dtype=np.float64)
        summary["by_category"][category] = {
            "n_splits": len(subset),
            "spearman_gain_vs_gen_margin": float(spearmanr(x_margin, y_gain).correlation),
            "spearman_gain_vs_ident_margin": float(spearmanr(x_ident, y_gain).correlation),
            "spearman_gain_vs_gen_regret": float(spearmanr(x_regret, y_gain).correlation),
            "spearman_gain_vs_gen_accuracy": float(spearmanr(x_acc, y_gain).correlation),
        }
    return summary


def _summarize_tradeoff_hypothesis(agg_rows: list[dict[str, object]], recoverability: dict[str, object]) -> dict[str, object]:
    animals = [row for row in agg_rows if row["category"] == "animals"]
    fruits = [row for row in agg_rows if row["category"] == "fruits"]
    animals_sorted = sorted(animals, key=lambda row: float(row["mean_gen_accuracy_delta"]))
    fruits_sorted = sorted(fruits, key=lambda row: float(row["mean_gen_accuracy_delta"]))
    animals_best = animals_sorted[-1] if animals_sorted else None
    animals_mid = animals_sorted[len(animals_sorted) // 2] if animals_sorted else None

    tradeoff_supported = False
    if animals_best and animals_mid:
        tradeoff_supported = (
            float(animals_best["mean_gen_accuracy_delta"]) > float(animals_mid["mean_gen_accuracy_delta"])
            and float(animals_best["mean_cue_displacement"]) >= float(animals_mid["mean_cue_displacement"]) + 0.05
            and float(animals_best["mean_target_pull_gain"]) > float(animals_mid["mean_target_pull_gain"])
        )

    animals_dnoise = [row for row in animals if float(row["decision_noise_std"]) == 0.01]
    displacement_corr = float("nan")
    target_pull_corr = float("nan")
    if len(animals_dnoise) >= 3:
        displacement_corr = float(
            spearmanr(
                [float(row["mean_gen_accuracy_delta"]) for row in animals_dnoise],
                [float(row["mean_cue_displacement"]) for row in animals_dnoise],
            ).correlation
        )
        target_pull_corr = float(
            spearmanr(
                [float(row["mean_gen_accuracy_delta"]) for row in animals_dnoise],
                [float(row["mean_target_pull_gain"]) for row in animals_dnoise],
            ).correlation
        )

    gentle_stabilization_supported = (
        bool(animals_best)
        and float(animals_best["mean_gen_accuracy_delta"]) > 0.0
        and float(animals_best["mean_cue_displacement"]) < 0.02
        and displacement_corr < -0.2
    )

    rec = recoverability.get("by_category", {})
    recoverability_supported = False
    if "animals" in rec and "fruits" in rec:
        animals_corr = float(rec["animals"]["spearman_gain_vs_gen_margin"])
        fruits_corr = float(rec["fruits"]["spearman_gain_vs_gen_margin"])
        recoverability_supported = animals_corr > 0.3 and animals_corr >= fruits_corr

    if tradeoff_supported:
        supported = "cue_fidelity_tradeoff"
    elif gentle_stabilization_supported:
        supported = "gentle_stabilization"
    else:
        supported = "recoverability_gate"
    return {
        "supported_hypothesis": supported,
        "tradeoff_supported": tradeoff_supported,
        "gentle_stabilization_supported": gentle_stabilization_supported,
        "recoverability_supported": recoverability_supported,
        "animals_gain_vs_displacement_spearman": displacement_corr,
        "animals_gain_vs_target_pull_spearman": target_pull_corr,
        "animals_best_retrieval_config": animals_best,
        "animals_mid_retrieval_config": animals_mid,
        "fruits_best_retrieval_config": fruits_sorted[-1] if fruits_sorted else None,
        "key_findings": {
            "animals_mean_best_gen_delta": float(animals_best["mean_gen_accuracy_delta"]) if animals_best else float("nan"),
            "animals_mean_best_cue_displacement": float(animals_best["mean_cue_displacement"]) if animals_best else float("nan"),
            "animals_mean_best_probe_rdm_delta": float(animals_best["mean_probe_rdm_spearman_delta"]) if animals_best else float("nan"),
            "fruits_mean_best_gen_delta": float(fruits_sorted[-1]["mean_gen_accuracy_delta"]) if fruits_sorted else float("nan"),
            "fruits_mean_best_margin_delta": float(fruits_sorted[-1]["mean_gen_avg_margin_delta"]) if fruits_sorted else float("nan"),
        },
    }


def _corruption_label(branch: dict[str, object], decision_noise_std: float) -> str:
    if str(branch["corruption_mode"]) == "occlusion":
        return f"occ{float(branch['occlusion_frac']):g}_dnoise{decision_noise_std:g}"
    return f"{branch['corruption_mode']}_dnoise{decision_noise_std:g}"


def run_hard_ident_dam(
    *,
    device: str,
    batch_size: int,
    seed: int,
    output_dir: Path,
    dataset_pkl: str,
    image_root: str,
    backend: str,
    max_configs: int | None = None,
    config_set: str = "tradeoff",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_pool = STAGE_A_CONFIGS if config_set == "stage_a" else get_hard_ident_tradeoff_configs()
    configs = config_pool[: max_configs or len(config_pool)]
    rows: list[dict[str, object]] = []

    for branch in HARD_IDENT_BRANCHES:
        data = load_naturalistic_category(
            str(branch["category"]),
            dataset_pkl=dataset_pkl,
            image_root=image_root,
            required_exemplars=branch["required_exemplars"],
        )
        images = _load_images(data.image_paths)
        wrapper = VisionEmbeddingWrapper(
            model_name=str(branch["model_name"]),
            pretrained=True,
            device=device,
            pooling=str(branch["pooling"]),
        )
        raw_features = wrapper.extract(images, batch_size=batch_size)
        prep = LayerwisePreprocessor(use_zscore=False, l2_normalize=True)
        clean_features = prep.fit_transform(raw_features)
        clean_reextract_features = prep.transform(wrapper.extract(images, batch_size=batch_size))
        rdms = {
            layer: _rdm_spearman(layer_features, data.similarity_matrix)
            for layer, layer_features in clean_features.items()
        }
        split_specs = build_category_split_specs(
            category=str(branch["category"]),
            concepts=data.concepts,
            n_items=len(data.filenames),
            split_mode=str(branch["split_mode"]),
            storage_sizes=branch["storage_sizes"],
            n_seeds=int(branch["n_seeds"]),
            seed=seed,
        )

        for decision_noise_std in branch["decision_noise_levels"]:
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
                corrupted_images = [
                    corrupt_image(
                        images[int(idx)],
                        mode=str(branch["corruption_mode"]),
                        seed=seed + 1000 + int(idx),
                        occlusion_frac=float(branch["occlusion_frac"]),
                    )
                    for idx in ident_task.probe_indices
                ]
                corrupted_probe_bundle = prep.transform(wrapper.extract(corrupted_images, batch_size=batch_size))
                gen_probe_bundle = subset_feature_bundle(clean_features, gen_task.probe_indices)

                ident_baseline = evaluate_layerwise_baseline(
                    stored_feature_bundle,
                    corrupted_probe_bundle,
                    ident_task.ground_truth_idx,
                    metric="cosine",
                    decision_noise_std=float(decision_noise_std),
                    decision_noise_seed=seed + split_seed,
                )
                gen_baseline = evaluate_layerwise_baseline(
                    stored_feature_bundle,
                    gen_probe_bundle,
                    gen_task.ground_truth_idx,
                    metric="cosine",
                    decision_noise_std=float(decision_noise_std),
                    decision_noise_seed=seed + 50000 + split_seed,
                )
                layer = str(branch["layer"])
                gen_extra = _baseline_generalization_extras(
                    baseline_row=gen_baseline[layer],
                    task=gen_task,
                    similarity_matrix=data.similarity_matrix,
                    concepts=data.concepts,
                )
                base_row = {
                    "category": branch["category"],
                    "model_name": branch["model_name"],
                    "pooling": branch["pooling"],
                    "layer": layer,
                    "split_label": split_label,
                    "split_seed": split_seed,
                    "split_mode": branch["split_mode"],
                    "n_stored": int(len(stored_indices)),
                    "n_probe": int(len(probe_indices)),
                    "chance_accuracy": 100.0 / len(stored_indices),
                    "corruption_mode": branch["corruption_mode"],
                    "occlusion_frac": float(branch["occlusion_frac"]),
                    "decision_noise_std": float(decision_noise_std),
                    "corruption_label": _corruption_label(branch, float(decision_noise_std)),
                    "ident_accuracy_baseline": ident_baseline[layer]["accuracy"],
                    "ident_avg_margin_baseline": ident_baseline[layer]["avg_margin"],
                    "gen_accuracy_baseline": gen_baseline[layer]["accuracy"],
                    "gen_avg_margin_baseline": gen_baseline[layer]["avg_margin"],
                    "gen_avg_retrieved_human_similarity_baseline": gen_extra["gen_avg_retrieved_human_similarity_baseline"],
                    "gen_avg_human_similarity_regret_baseline": gen_extra["gen_avg_human_similarity_regret_baseline"],
                    "gen_same_concept_accuracy_baseline": gen_extra["gen_same_concept_accuracy_baseline"],
                    "human_rdm_spearman_baseline": rdms[layer],
                }
                for config in configs:
                    dam = evaluate_dam_trial(
                        layer=layer,
                        stored_feature_bundle=stored_feature_bundle,
                        clean_probe_bundle=clean_probe_bundle,
                        noisy_probe_bundle=corrupted_probe_bundle,
                        gen_probe_bundle=gen_probe_bundle,
                        ident_ground_truth_idx=ident_task.ground_truth_idx,
                        gen_ground_truth_idx=gen_task.ground_truth_idx,
                        similarity_matrix=data.similarity_matrix,
                        gen_probe_indices=gen_task.probe_indices,
                        gen_stored_indices=gen_task.stored_indices,
                        concepts=data.concepts,
                        config=config,
                        seed=seed + split_seed * 10000,
                        decision_noise_std=float(decision_noise_std),
                    )
                    row = dict(base_row)
                    row.update(dam)
                    row["ident_accuracy_delta"] = float(row["ident_accuracy_dam"]) - float(row["ident_accuracy_baseline"])
                    row["gen_accuracy_delta"] = float(row["gen_accuracy_dam"]) - float(row["gen_accuracy_baseline"])
                    row["gen_avg_margin_delta"] = float(row["gen_avg_margin_dam"]) - float(row["gen_avg_margin_baseline"])
                    row["gen_avg_human_similarity_regret_delta"] = (
                        float(row["gen_avg_human_similarity_regret_dam"]) - float(row["gen_avg_human_similarity_regret_baseline"])
                    )
                    row["qualifying_hard_ident_win"] = _qualifying_hard_ident_win(row)
                    row["dam_n"] = int(config["n"])
                    row["dam_beta"] = float(config["beta"])
                    row["dam_alpha"] = float(config["alpha"])
                    row["dam_lmbda"] = float(config["lmbda"])
                    row["dam_steps_multiplier"] = int(config["steps_multiplier"])
                    rows.append(row)

    csv_path = output_dir / "hard_ident_combined.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregated_rows = _aggregate_config_rows(rows)
    agg_csv_path = output_dir / "hard_ident_aggregated.csv"
    with agg_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregated_rows)

    recoverability = _summarize_recoverability(rows)
    hypothesis = _summarize_tradeoff_hypothesis(aggregated_rows, recoverability)

    summary = {
        "n_rows": len(rows),
        "n_aggregated_rows": len(aggregated_rows),
        "n_qualifying_hard_ident_wins": int(sum(bool(row["qualifying_hard_ident_win"]) for row in rows)),
        "best_by_branch": {},
        "best_aggregated_by_branch": {},
        "recoverability_analysis": recoverability,
        "hypothesis_summary": hypothesis,
    }
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["category"]), float(row["decision_noise_std"])), []).append(row)
    for key, group_rows in grouped.items():
        summary["best_by_branch"][f"{key[0]}::dnoise{key[1]:g}"] = max(
            group_rows,
            key=lambda row: (
                float(row["gen_accuracy_delta"]),
                float(row["gen_avg_margin_delta"]),
                float(row["probe_rdm_spearman_delta"]),
            ),
        )
    agg_grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in aggregated_rows:
        agg_grouped.setdefault((str(row["category"]), float(row["decision_noise_std"])), []).append(row)
    for key, group_rows in agg_grouped.items():
        summary["best_aggregated_by_branch"][f"{key[0]}::dnoise{key[1]:g}"] = {
            "best_retrieval_gain": _best_row(group_rows, lambda row: float(row["mean_gen_accuracy_delta"])),
            "best_fidelity_preserving": _best_row(
                group_rows,
                lambda row: (
                    float(row["mean_probe_rdm_spearman_delta"]),
                    -float(row["mean_cue_displacement"]),
                    float(row["mean_gen_accuracy_delta"]),
                ),
            ),
            "best_tradeoff": _best_row(group_rows, _score_tradeoff_row),
        }
    with (output_dir / "hard_ident_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_hard_ident_dam(
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        dataset_pkl=args.dataset_pkl,
        image_root=args.image_root,
        backend=args.backend,
        max_configs=args.max_configs,
        config_set=args.config_set,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
