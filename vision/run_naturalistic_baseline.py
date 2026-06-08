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
    build_preview_grid,
    build_leave_one_exemplar_out_folds,
    corrupt_image,
    evaluate_clean_reextract_sanity,
    evaluate_exact_image_sanity,
    evaluate_layerwise_baseline,
    load_naturalistic_category,
    perturb_image,
    subset_feature_bundle,
)


BASELINE_MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("vit_base_patch16_224", "cls"),
    ("vit_base_patch16_224", "mean_tokens"),
    ("convnext_tiny", "auto"),
)

CLIP_MODEL_SPECS: tuple[tuple[str, str], ...] = BASELINE_MODEL_SPECS + (
    ("vit_base_patch16_clip_224.openai", "cls"),
    ("vit_base_patch16_clip_224.openai", "mean_tokens"),
)


def get_model_specs(include_clip: bool = False) -> tuple[tuple[str, str], ...]:
    return CLIP_MODEL_SPECS if include_clip else BASELINE_MODEL_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the naturalistic fruits baseline with Peterson similarity.")
    parser.add_argument("--category", default="fruits")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/vision/naturalistic")
    parser.add_argument("--dataset-pkl", default="datasets_peterson.pkl")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--required-exemplars", type=int, default=None)
    parser.add_argument(
        "--split-mode",
        choices=("balanced_exemplar_folds", "random_storage_curve"),
        default=None,
    )
    parser.add_argument("--storage-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--include-clip", action="store_true")
    parser.add_argument("--corruption-mode", default="noise_shift")
    parser.add_argument("--noise-std", type=float, default=5.0)
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--occlusion-frac", type=float, default=0.3)
    parser.add_argument("--mask-frac", type=float, default=0.3)
    parser.add_argument("--n-chunks", type=int, default=6)
    parser.add_argument("--affine-strength", type=float, default=0.05)
    parser.add_argument("--decision-noise-std", type=float, default=0.0)
    parser.add_argument("--save-preview", action="store_true")
    return parser.parse_args()


def _load_images(paths: tuple[Path, ...]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _feature_rdm(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    normalized = features / norms
    cosine_sim = normalized @ normalized.T
    return 1.0 - cosine_sim


def _human_rdm(similarity_matrix: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(similarity_matrix, dtype=np.float64)


def _rdm_spearman(features: np.ndarray, similarity_matrix: np.ndarray) -> float:
    feature_rdm = _feature_rdm(features)
    human_rdm = _human_rdm(similarity_matrix)
    tri = np.triu_indices_from(feature_rdm, k=1)
    corr = spearmanr(feature_rdm[tri], human_rdm[tri]).correlation
    return float(corr)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _summarize_integrity(dataset_pkl: str, image_root: str) -> dict[str, object]:
    categories = {}
    for category in ("animals", "fruits", "vegetables"):
        categories[category] = load_naturalistic_category(
            category,
            dataset_pkl=dataset_pkl,
            image_root=image_root,
        ).integrity
    return categories


def _random_storage_splits(
    n_items: int,
    *,
    storage_sizes: list[int],
    n_seeds: int,
    seed: int,
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    splits: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")
    for n_stored in storage_sizes:
        if n_stored <= 0 or n_stored >= n_items:
            raise ValueError(f"n_stored must be between 1 and {n_items - 1}")
        for split_seed in range(n_seeds):
            rng = np.random.default_rng(seed + split_seed + 10000 * n_stored)
            perm = rng.permutation(n_items)
            stored = np.asarray(perm[:n_stored], dtype=np.int64)
            probes = np.asarray(perm[n_stored:], dtype=np.int64)
            splits.append((f"stored_{n_stored}_seed_{split_seed}", split_seed, stored, probes))
    return splits


def _corruption_label(
    *,
    corruption_mode: str,
    noise_std: float,
    max_shift: int,
    occlusion_frac: float,
    mask_frac: float,
    n_chunks: int,
    affine_strength: float,
    decision_noise_std: float,
) -> str:
    if corruption_mode == "noise_shift":
        base = f"noise{noise_std:g}_shift{max_shift}"
    elif corruption_mode == "occlusion":
        base = f"occ{occlusion_frac:g}"
    elif corruption_mode == "multi_cutout":
        base = f"cutout{mask_frac:g}_chunks{n_chunks}"
    elif corruption_mode == "warp":
        base = f"warp{affine_strength:g}"
    else:
        base = corruption_mode
    return f"{base}_dnoise{decision_noise_std:g}"


def _build_corrupted_identification_images(
    images: list[Image.Image],
    probe_indices: np.ndarray,
    *,
    corruption_mode: str,
    noise_std: float,
    max_shift: int,
    occlusion_frac: float,
    mask_frac: float,
    n_chunks: int,
    affine_strength: float,
    seed: int,
) -> list[Image.Image]:
    return [
        corrupt_image(
            images[int(idx)],
            mode=corruption_mode,
            seed=seed + 1000 + int(idx),
            noise_std=noise_std,
            max_shift=max_shift,
            occlusion_frac=occlusion_frac,
            mask_frac=mask_frac,
            n_chunks=n_chunks,
            affine_strength=affine_strength,
        )
        for idx in probe_indices
    ]


def _save_corruption_preview(
    *,
    output_dir: Path,
    category: str,
    images: list[Image.Image],
    probe_indices: np.ndarray,
    corruption_mode: str,
    noise_std: float,
    max_shift: int,
    occlusion_frac: float,
    mask_frac: float,
    n_chunks: int,
    affine_strength: float,
    decision_noise_std: float,
    seed: int,
) -> Path | None:
    if probe_indices.size == 0:
        return None
    sample_indices = probe_indices[:3]
    preview_images: list[Image.Image] = []
    for idx in sample_indices:
        clean = images[int(idx)]
        corrupted = corrupt_image(
            clean,
            mode=corruption_mode,
            seed=seed + 1000 + int(idx),
            noise_std=noise_std,
            max_shift=max_shift,
            occlusion_frac=occlusion_frac,
            mask_frac=mask_frac,
            n_chunks=n_chunks,
            affine_strength=affine_strength,
        )
        preview_images.extend([clean, corrupted])
    grid = build_preview_grid(preview_images, ncols=2, pad=4)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    label = _corruption_label(
        corruption_mode=corruption_mode,
        noise_std=noise_std,
        max_shift=max_shift,
        occlusion_frac=occlusion_frac,
        mask_frac=mask_frac,
        n_chunks=n_chunks,
        affine_strength=affine_strength,
        decision_noise_std=decision_noise_std,
    )
    path = preview_dir / f"{category}_{label}.png"
    grid.save(path)
    return path


def summarize_cross_category(output_dir: Path, categories: list[str]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for category in categories:
        csv_path = output_dir / f"{category}_combined.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["category"], row["model_name"], row["pooling"], row["layer"])
        grouped.setdefault(key, []).append(row)

    averaged: list[dict[str, object]] = []
    metrics = [
        "ident_accuracy",
        "gen_accuracy",
        "gen_avg_margin",
        "gen_avg_retrieved_human_similarity",
        "gen_avg_human_similarity_regret",
        "gen_same_concept_accuracy",
        "human_rdm_spearman",
        "chance_accuracy",
    ]
    for key, group_rows in grouped.items():
        out = {
            "category": key[0],
            "model_name": key[1],
            "pooling": key[2],
            "layer": key[3],
        }
        for metric in metrics:
            out[metric] = _nanmean([float(row[metric]) for row in group_rows])
        averaged.append(out)

    summary = {
        "categories": sorted({row["category"] for row in averaged}),
        "best_generalization": {},
        "best_alignment": {},
        "best_same_concept": {},
        "category_deltas": {},
    }
    for category in summary["categories"]:
        subset = [row for row in averaged if row["category"] == category]
        summary["best_generalization"][category] = max(subset, key=lambda row: row["gen_accuracy"])
        summary["best_alignment"][category] = max(subset, key=lambda row: row["human_rdm_spearman"])
        same_concept_subset = [row for row in subset if not np.isnan(row["gen_same_concept_accuracy"])]
        summary["best_same_concept"][category] = (
            max(same_concept_subset, key=lambda row: row["gen_same_concept_accuracy"]) if same_concept_subset else None
        )

    if len(summary["categories"]) >= 2:
        delta_rows = []
        base_category = summary["categories"][0]
        base_lookup = {
            (row["model_name"], row["pooling"], row["layer"]): row
            for row in averaged
            if row["category"] == base_category
        }
        for other_category in summary["categories"][1:]:
            other_lookup = {
                (row["model_name"], row["pooling"], row["layer"]): row
                for row in averaged
                if row["category"] == other_category
            }
            shared_keys = sorted(set(base_lookup) & set(other_lookup))
            for key in shared_keys:
                base = base_lookup[key]
                other = other_lookup[key]
                delta_rows.append(
                    {
                        "base_category": base_category,
                        "other_category": other_category,
                        "model_name": key[0],
                        "pooling": key[1],
                        "layer": key[2],
                        "gen_accuracy_delta_other_minus_base": other["gen_accuracy"] - base["gen_accuracy"],
                        "human_rdm_spearman_delta_other_minus_base": other["human_rdm_spearman"] - base["human_rdm_spearman"],
                        "gen_same_concept_delta_other_minus_base": other["gen_same_concept_accuracy"] - base["gen_same_concept_accuracy"],
                        "gen_regret_delta_other_minus_base": other["gen_avg_human_similarity_regret"] - base["gen_avg_human_similarity_regret"],
                    }
                )
        summary["category_deltas"] = delta_rows

    comparison_name = "_vs_".join(summary["categories"]) if summary["categories"] else "cross_category"
    csv_path = output_dir / f"{comparison_name}_summary.csv"
    if averaged:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(averaged[0].keys()))
            writer.writeheader()
            writer.writerows(averaged)
    with (output_dir / f"{comparison_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_naturalistic_baseline(
    *,
    category: str,
    device: str,
    batch_size: int,
    seed: int,
    output_dir: Path,
    dataset_pkl: str,
    image_root: str,
    required_exemplars: int | None = None,
    split_mode: str | None = None,
    storage_sizes: list[int] | None = None,
    n_seeds: int = 5,
    model_specs: tuple[tuple[str, str], ...] | None = None,
    corruption_mode: str = "noise_shift",
    noise_std: float = 5.0,
    max_shift: int = 2,
    occlusion_frac: float = 0.3,
    mask_frac: float = 0.3,
    n_chunks: int = 6,
    affine_strength: float = 0.05,
    decision_noise_std: float = 0.0,
    save_preview: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    integrity_summary = _summarize_integrity(dataset_pkl, image_root)
    with (output_dir / "dataset_integrity.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity_summary, handle, indent=2)

    data = load_naturalistic_category(
        category,
        dataset_pkl=dataset_pkl,
        image_root=image_root,
        required_exemplars=required_exemplars,
    )
    images = _load_images(data.image_paths)
    has_meaningful_concepts = len(set(data.concepts)) > 1
    effective_split_mode = split_mode
    if effective_split_mode is None:
        effective_split_mode = "balanced_exemplar_folds" if required_exemplars is not None or category in {"fruits", "vegetables"} else "random_storage_curve"

    if effective_split_mode == "balanced_exemplar_folds":
        split_specs = [
            (f"fold_{fold_index}", fold_index, stored_indices, probe_indices)
            for fold_index, (stored_indices, probe_indices) in enumerate(build_leave_one_exemplar_out_folds(data.concepts))
        ]
    elif effective_split_mode == "random_storage_curve":
        if storage_sizes is None:
            storage_sizes = [40, 80]
        split_specs = _random_storage_splits(
            len(data.filenames),
            storage_sizes=storage_sizes,
            n_seeds=n_seeds,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown split mode: {effective_split_mode}")

    corruption_label = _corruption_label(
        corruption_mode=corruption_mode,
        noise_std=noise_std,
        max_shift=max_shift,
        occlusion_frac=occlusion_frac,
        mask_frac=mask_frac,
        n_chunks=n_chunks,
        affine_strength=affine_strength,
        decision_noise_std=decision_noise_std,
    )
    preview_path: Path | None = None

    rows: list[dict[str, object]] = []
    wrappers: dict[tuple[str, str], VisionEmbeddingWrapper] = {}
    active_model_specs = model_specs or BASELINE_MODEL_SPECS

    for model_name, pooling in active_model_specs:
        wrapper = wrappers.setdefault(
            (model_name, pooling),
            VisionEmbeddingWrapper(model_name=model_name, pretrained=True, device=device, pooling=pooling),
        )
        raw_features = wrapper.extract(images, batch_size=batch_size)
        prep = LayerwisePreprocessor(use_zscore=False, l2_normalize=True)
        clean_features = prep.fit_transform(raw_features)
        clean_reextract_features = prep.transform(wrapper.extract(images, batch_size=batch_size))
        rdms = {
            layer: _rdm_spearman(layer_features, data.similarity_matrix)
            for layer, layer_features in clean_features.items()
        }

        fold_rows: list[dict[str, object]] = []
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
            noisy_images = _build_corrupted_identification_images(
                images,
                ident_task.probe_indices,
                corruption_mode=corruption_mode,
                noise_std=noise_std,
                max_shift=max_shift,
                occlusion_frac=occlusion_frac,
                mask_frac=mask_frac,
                n_chunks=n_chunks,
                affine_strength=affine_strength,
                seed=seed,
            )
            noisy_features = prep.transform(wrapper.extract(noisy_images, batch_size=batch_size))
            gen_probe_bundle = subset_feature_bundle(clean_features, gen_task.probe_indices)

            if save_preview and preview_path is None:
                preview_path = _save_corruption_preview(
                    output_dir=output_dir,
                    category=category,
                    images=images,
                    probe_indices=ident_task.probe_indices,
                    corruption_mode=corruption_mode,
                    noise_std=noise_std,
                    max_shift=max_shift,
                    occlusion_frac=occlusion_frac,
                    mask_frac=mask_frac,
                    n_chunks=n_chunks,
                    affine_strength=affine_strength,
                    decision_noise_std=decision_noise_std,
                    seed=seed,
                )

            exact_sanity = evaluate_exact_image_sanity(stored_feature_bundle, metric="cosine")
            clean_reextract_sanity = evaluate_clean_reextract_sanity(
                stored_feature_bundle,
                clean_probe_bundle,
                ident_task.ground_truth_idx,
                metric="cosine",
            )
            ident_results = evaluate_layerwise_baseline(
                stored_feature_bundle,
                noisy_features,
                ident_task.ground_truth_idx,
                metric="cosine",
                decision_noise_std=decision_noise_std,
                decision_noise_seed=seed + split_seed,
            )
            gen_results = evaluate_layerwise_baseline(
                stored_feature_bundle,
                gen_probe_bundle,
                gen_task.ground_truth_idx,
                metric="cosine",
                decision_noise_std=decision_noise_std,
                decision_noise_seed=seed + 50000 + split_seed,
            )

            for layer in sorted(ident_results):
                winners = np.asarray(gen_results[layer]["winners"], dtype=np.int64)
                stored_global = np.asarray(gen_task.stored_indices, dtype=np.int64)
                probe_global = np.asarray(gen_task.probe_indices, dtype=np.int64)
                retrieved_global = stored_global[winners]
                target_global = stored_global[np.asarray(gen_task.ground_truth_idx, dtype=np.int64)]

                retrieved_sims = np.asarray(
                    [data.similarity_matrix[int(probe), int(ret)] for probe, ret in zip(probe_global, retrieved_global)],
                    dtype=np.float64,
                )
                target_sims = np.asarray(
                    [data.similarity_matrix[int(probe), int(tgt)] for probe, tgt in zip(probe_global, target_global)],
                    dtype=np.float64,
                )
                same_concept = np.asarray(
                    [
                        data.concepts[int(probe)] == data.concepts[int(ret)]
                        for probe, ret in zip(probe_global, retrieved_global)
                    ],
                    dtype=np.float64,
                )

                row = {
                    "category": category,
                    "split_label": split_label,
                    "split_seed": split_seed,
                    "split_mode": effective_split_mode,
                    "model_name": model_name,
                    "pooling": pooling,
                    "layer": layer,
                    "n_stored": int(len(stored_indices)),
                    "n_probe": int(len(probe_indices)),
                    "chance_accuracy": 100.0 / len(stored_indices),
                    "corruption_mode": corruption_mode,
                    "corruption_label": corruption_label,
                    "noise_std": float(noise_std),
                    "max_shift": int(max_shift),
                    "occlusion_frac": float(occlusion_frac),
                    "mask_frac": float(mask_frac),
                    "n_chunks": int(n_chunks),
                    "affine_strength": float(affine_strength),
                    "decision_noise_std": float(decision_noise_std),
                    "ident_accuracy": ident_results[layer]["accuracy"],
                    "ident_avg_target_sim": ident_results[layer]["avg_target_sim"],
                    "ident_avg_margin": ident_results[layer]["avg_margin"],
                    "gen_accuracy": gen_results[layer]["accuracy"],
                    "gen_avg_target_sim": gen_results[layer]["avg_target_sim"],
                    "gen_avg_margin": gen_results[layer]["avg_margin"],
                    "gen_avg_retrieved_human_similarity": float(np.mean(retrieved_sims)),
                    "gen_avg_human_similarity_regret": float(np.mean(target_sims - retrieved_sims)),
                    "gen_same_concept_accuracy": (
                        float(np.mean(same_concept) * 100.0) if has_meaningful_concepts else float("nan")
                    ),
                    "human_rdm_spearman": rdms[layer],
                    "exact_accuracy": exact_sanity[layer]["accuracy"],
                    "clean_reextract_accuracy": clean_reextract_sanity[layer]["accuracy"],
                    "exact_avg_target_sim": exact_sanity[layer]["avg_target_sim"],
                    "clean_reextract_avg_target_sim": clean_reextract_sanity[layer]["avg_target_sim"],
                }
                rows.append(row)
                fold_rows.append(row)

        model_slug = f"{category}_{model_name}_{pooling}".replace(".", "_")
        with (output_dir / f"{model_slug}.json").open("w", encoding="utf-8") as handle:
            json.dump(fold_rows, handle, indent=2)

    csv_path = output_dir / f"{category}_combined.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "category": category,
        "chance_accuracy_values": sorted({float(row["chance_accuracy"]) for row in rows}),
        "required_exemplars": required_exemplars,
        "split_mode": effective_split_mode,
        "storage_sizes": sorted({int(row["n_stored"]) for row in rows}),
        "n_seeds": n_seeds if effective_split_mode == "random_storage_curve" else None,
        "models": [f"{model_name}::{pooling}" for model_name, pooling in active_model_specs],
        "corruption_mode": corruption_mode,
        "corruption_label": corruption_label,
        "decision_noise_std": float(decision_noise_std),
        "preview_path": str(preview_path) if preview_path is not None else None,
        "sanity_all_pass": all(
            row["exact_accuracy"] == 100.0
            and row["clean_reextract_accuracy"] == 100.0
            and abs(float(row["exact_avg_target_sim"]) - 1.0) < 1e-6
            and abs(float(row["clean_reextract_avg_target_sim"]) - 1.0) < 1e-6
            for row in rows
        ),
        "best_generalization": {},
        "best_alignment": {},
        "best_margin": {},
    }
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(f"{row['model_name']}::{row['pooling']}", []).append(row)
    for key, model_rows in by_model.items():
        summary["best_generalization"][key] = max(
            model_rows,
            key=lambda row: (float(row["gen_accuracy"]), -float(row["gen_avg_human_similarity_regret"])),
        )
        summary["best_alignment"][key] = max(model_rows, key=lambda row: float(row["human_rdm_spearman"]))
        summary["best_margin"][key] = max(model_rows, key=lambda row: float(row["gen_avg_margin"]))

    with (output_dir / f"{category}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_naturalistic_baseline(
        category=args.category,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        dataset_pkl=args.dataset_pkl,
        image_root=args.image_root,
        required_exemplars=args.required_exemplars,
        split_mode=args.split_mode,
        storage_sizes=args.storage_sizes,
        n_seeds=args.n_seeds,
        model_specs=get_model_specs(include_clip=args.include_clip),
        corruption_mode=args.corruption_mode,
        noise_std=args.noise_std,
        max_shift=args.max_shift,
        occlusion_frac=args.occlusion_frac,
        mask_frac=args.mask_frac,
        n_chunks=args.n_chunks,
        affine_strength=args.affine_strength,
        decision_noise_std=args.decision_noise_std,
        save_preview=args.save_preview,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
