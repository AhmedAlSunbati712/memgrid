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
    perturb_image,
    subset_feature_bundle,
)
from vision.naturalistic_dam import (
    FrontierBranchSpec,
    build_category_split_specs,
    evaluate_dam_trial,
    get_animals_frontier_configs,
    get_frontier_branch_specs,
    summarize_frontier_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the focused animals/fruits DAM frontier benchmark.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/vision/naturalistic_dam/frontier")
    parser.add_argument("--baseline-dir", default="results/vision/naturalistic")
    parser.add_argument("--dataset-pkl", default="datasets_peterson.pkl")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numba")
    parser.add_argument("--max-configs", type=int, default=None)
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
        "gen_same_concept_accuracy_baseline": float(np.mean(same_concept) * 100.0) if len(set(concepts)) < len(concepts) else float("nan"),
    }


def _required_exemplars(category: str) -> int | None:
    if category == "vegetables":
        return 3
    return None


def run_frontier(
    *,
    device: str,
    batch_size: int,
    seed: int,
    output_dir: Path,
    dataset_pkl: str,
    image_root: str,
    backend: str,
    max_configs: int | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = get_animals_frontier_configs()
    if max_configs is not None:
        configs = configs[:max_configs]

    rows: list[dict[str, object]] = []
    for branch in get_frontier_branch_specs():
        data = load_naturalistic_category(
            branch.category,
            dataset_pkl=dataset_pkl,
            image_root=image_root,
            required_exemplars=_required_exemplars(branch.category),
        )
        images = _load_images(data.image_paths)
        wrapper = VisionEmbeddingWrapper(
            model_name=branch.model_name,
            pretrained=True,
            device=device,
            pooling=branch.pooling,
        )
        raw_features = wrapper.extract(images, batch_size=batch_size)
        prep = LayerwisePreprocessor(use_zscore=False, l2_normalize=True)
        clean_features = prep.fit_transform(raw_features)
        clean_reextract_features = prep.transform(wrapper.extract(images, batch_size=batch_size))
        noisy_images = [
            perturb_image(image, noise_std=5.0, max_shift=2, seed=seed + 1000 + idx)
            for idx, image in enumerate(images)
        ]
        noisy_features = prep.transform(wrapper.extract(noisy_images, batch_size=batch_size))
        rdms = {
            layer: _rdm_spearman(layer_features, data.similarity_matrix)
            for layer, layer_features in clean_features.items()
        }

        split_specs = build_category_split_specs(
            category=branch.category,
            concepts=data.concepts,
            n_items=len(data.filenames),
            split_mode=branch.split_mode,
            storage_sizes=list(branch.storage_sizes) if branch.storage_sizes is not None else None,
            n_seeds=branch.n_seeds,
            seed=seed,
        )

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
            )
            gen_baseline = evaluate_layerwise_baseline(
                stored_feature_bundle,
                gen_probe_bundle,
                gen_task.ground_truth_idx,
                metric="cosine",
            )

            gen_extra = _baseline_generalization_extras(
                baseline_row=gen_baseline[branch.layer],
                task=gen_task,
                similarity_matrix=data.similarity_matrix,
                concepts=data.concepts,
            )
            base_row = {
                "category": branch.category,
                "model_name": branch.model_name,
                "pooling": branch.pooling,
                "layer": branch.layer,
                "split_label": split_label,
                "split_seed": split_seed,
                "split_mode": branch.split_mode,
                "n_stored": int(len(stored_indices)),
                "n_probe": int(len(probe_indices)),
                "chance_accuracy": 100.0 / len(stored_indices),
                "ident_accuracy_baseline": ident_baseline[branch.layer]["accuracy"],
                "ident_avg_margin_baseline": ident_baseline[branch.layer]["avg_margin"],
                "gen_accuracy_baseline": gen_baseline[branch.layer]["accuracy"],
                "gen_avg_margin_baseline": gen_baseline[branch.layer]["avg_margin"],
                "gen_avg_retrieved_human_similarity_baseline": gen_extra["gen_avg_retrieved_human_similarity_baseline"],
                "gen_avg_human_similarity_regret_baseline": gen_extra["gen_avg_human_similarity_regret_baseline"],
                "gen_same_concept_accuracy_baseline": gen_extra["gen_same_concept_accuracy_baseline"],
                "human_rdm_spearman_baseline": rdms[branch.layer],
            }

            for config in configs:
                dam = evaluate_dam_trial(
                    layer=branch.layer,
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
                row["dam_n"] = int(config["n"])
                row["dam_beta"] = float(config["beta"])
                row["dam_alpha"] = float(config["alpha"])
                row["dam_lmbda"] = float(config["lmbda"])
                row["dam_steps_multiplier"] = int(config["steps_multiplier"])
                rows.append(row)

    csv_path = output_dir / "frontier_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_frontier_runs(output_dir, rows)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_frontier(
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        dataset_pkl=args.dataset_pkl,
        image_root=args.image_root,
        backend=args.backend,
        max_configs=args.max_configs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
