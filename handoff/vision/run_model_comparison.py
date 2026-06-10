from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import (
    LayerwisePreprocessor,
    VisionEmbeddingWrapper,
    build_balanced_splits_by_color,
    build_generalization_task_synthetic,
    build_identification_task,
    evaluate_clean_reextract_sanity,
    evaluate_exact_image_sanity,
    evaluate_layerwise_baseline,
    generate_square_stimuli,
    subset_feature_bundle,
)
from vision.ident_corruption import corrupt_ident_probe
from vision.model_comparison import DEFAULT_MODEL_NAMES, build_model_comparison_configs, summarize_model_rows
from vision.synthetic_sweep import POSITION_ONLY_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic multi-model baseline comparison.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/vision/model_comparison")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODEL_NAMES))
    return parser.parse_args()


def _stimulus_parameters(config: dict[str, object], *, image_size: int, seed: int) -> tuple[dict[str, object], int]:
    if config["task_mode"] == "position_only":
        run_n_per_color = int(config["n_per_color"]) * 6
        colors = POSITION_ONLY_COLORS
    else:
        run_n_per_color = int(config["n_per_color"])
        colors = None

    kwargs: dict[str, object] = {
        "n_per_color": run_n_per_color,
        "image_size": image_size,
        "square_size": int(config["square_size"]),
        "fixed_position": config["task_mode"] == "color_only",
        "color_jitter_std": float(config["color_only_jitter_std"]) if config["task_mode"] == "color_only" else 0.0,
        "seed": seed,
    }
    if colors is not None:
        kwargs["colors"] = colors
    return kwargs, run_n_per_color


def _wrapper_key(model_name: str, pooling: str) -> tuple[str, str]:
    return (model_name, pooling)


def run_model_comparison(
    *,
    model_names: list[str],
    device: str,
    batch_size: int,
    image_size: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    wrappers: dict[tuple[str, str], VisionEmbeddingWrapper] = {}
    clean_cache: dict[tuple[object, ...], dict[str, object]] = {}
    noisy_cache: dict[tuple[object, ...], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []

    configs = [asdict(config) for config in build_model_comparison_configs(model_names)]
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config}")
        stimulus_kwargs, run_n_per_color = _stimulus_parameters(config, image_size=image_size, seed=seed)
        clean_key = (
            config["model_name"],
            config["pooling"],
            config["task_mode"],
            config["n_per_color"],
            config["square_size"],
            config["color_only_jitter_std"],
        )
        if clean_key not in clean_cache:
            wrapper_id = _wrapper_key(str(config["model_name"]), str(config["pooling"]))
            if wrapper_id not in wrappers:
                wrappers[wrapper_id] = VisionEmbeddingWrapper(
                    model_name=str(config["model_name"]),
                    pretrained=True,
                    device=device,
                    pooling=str(config["pooling"]),
                )
            wrapper = wrappers[wrapper_id]
            images, metadata = generate_square_stimuli(**stimulus_kwargs)
            raw_features = wrapper.extract(images, batch_size=batch_size)
            prep = LayerwisePreprocessor(use_zscore=False, l2_normalize=True)
            clean_features = prep.fit_transform(raw_features)
            clean_reextract_raw = wrapper.extract(images, batch_size=batch_size)
            clean_reextract_features = prep.transform(clean_reextract_raw)
            clean_cache[clean_key] = {
                "images": images,
                "metadata": metadata,
                "prep": prep,
                "features": clean_features,
                "clean_reextract_features": clean_reextract_features,
                "run_n_per_color": run_n_per_color,
            }

        clean_bundle = clean_cache[clean_key]
        images = clean_bundle["images"]
        metadata = clean_bundle["metadata"]
        prep = clean_bundle["prep"]
        clean_features = clean_bundle["features"]
        clean_reextract_features = clean_bundle["clean_reextract_features"]

        noisy_key = clean_key + (
            config["ident_noise_std"],
            config["ident_max_shift"],
            config["ident_occlusion_frac"],
        )
        if noisy_key not in noisy_cache:
            wrapper = wrappers[_wrapper_key(str(config["model_name"]), str(config["pooling"]))]
            noisy_images = [
                corrupt_ident_probe(
                    image,
                    seed=seed + 5000 + local_idx,
                    noise_std=float(config["ident_noise_std"]),
                    max_shift=int(config["ident_max_shift"]),
                    occlusion_frac=float(config["ident_occlusion_frac"]),
                )
                for local_idx, image in enumerate(images)
            ]
            noisy_raw = wrapper.extract(noisy_images, batch_size=batch_size)
            noisy_cache[noisy_key] = {"features": prep.transform(noisy_raw)}
        noisy_features = noisy_cache[noisy_key]["features"]

        stored_indices, novel_probe_indices = build_balanced_splits_by_color(
            metadata,
            n_stored_per_color=int(config["n_stored_per_color"]),
        )
        ident_task = build_identification_task(
            metadata,
            stored_indices=stored_indices,
            probe_indices=stored_indices,
        )
        gen_task = build_generalization_task_synthetic(
            metadata,
            stored_indices=stored_indices,
            probe_indices=novel_probe_indices,
            color_weight=0.0 if config["task_mode"] == "position_only" else 1.0,
            position_weight=0.0 if config["task_mode"] == "color_only" else 1.0,
        )

        stored_feature_bundle = subset_feature_bundle(clean_features, ident_task.stored_indices)
        ident_probe_bundle = subset_feature_bundle(noisy_features, ident_task.probe_indices)
        clean_probe_bundle = subset_feature_bundle(clean_reextract_features, ident_task.probe_indices)
        gen_probe_bundle = subset_feature_bundle(clean_features, gen_task.probe_indices)

        exact_sanity = evaluate_exact_image_sanity(stored_feature_bundle, metric="cosine")
        clean_reextract_sanity = evaluate_clean_reextract_sanity(
            stored_feature_bundle,
            clean_probe_bundle,
            ident_task.ground_truth_idx,
            metric="cosine",
        )
        ident_results = evaluate_layerwise_baseline(
            stored_feature_bundle,
            ident_probe_bundle,
            ident_task.ground_truth_idx,
            metric="cosine",
            decision_noise_std=float(config["ident_decision_noise_std"]),
            decision_noise_seed=seed + 7000,
        )
        gen_results = evaluate_layerwise_baseline(
            stored_feature_bundle,
            gen_probe_bundle,
            gen_task.ground_truth_idx,
            metric="cosine",
        )

        for layer in sorted(ident_results):
            row = {
                "model_name": config["model_name"],
                "seed": seed,
                "metric": "cosine",
                "preprocess_mode": "l2_only",
                "task_mode": config["task_mode"],
                "pooling": config["pooling"],
                "n_per_color": config["n_per_color"],
                "run_n_per_color": clean_bundle["run_n_per_color"],
                "n_stored_per_color": config["n_stored_per_color"],
                "square_size": config["square_size"],
                "ident_noise_std": config["ident_noise_std"],
                "ident_max_shift": config["ident_max_shift"],
                "ident_occlusion_frac": config["ident_occlusion_frac"],
                "ident_decision_noise_std": config["ident_decision_noise_std"],
                "color_only_jitter_std": config["color_only_jitter_std"],
                "layer": layer,
                "ident_accuracy": ident_results[layer]["accuracy"],
                "ident_avg_target_sim": ident_results[layer]["avg_target_sim"],
                "ident_avg_best_wrong_sim": ident_results[layer]["avg_best_wrong_sim"],
                "ident_avg_margin": ident_results[layer]["avg_margin"],
                "gen_accuracy": gen_results[layer]["accuracy"],
                "gen_avg_target_sim": gen_results[layer]["avg_target_sim"],
                "gen_avg_best_wrong_sim": gen_results[layer]["avg_best_wrong_sim"],
                "gen_avg_margin": gen_results[layer]["avg_margin"],
                "avg_margin": 0.5 * (ident_results[layer]["avg_margin"] + gen_results[layer]["avg_margin"]),
            }
            rows.append(row)
            sanity_rows.append(
                {
                    **row,
                    "exact_accuracy": exact_sanity[layer]["accuracy"],
                    "exact_avg_target_sim": exact_sanity[layer]["avg_target_sim"],
                    "clean_reextract_accuracy": clean_reextract_sanity[layer]["accuracy"],
                    "clean_reextract_avg_target_sim": clean_reextract_sanity[layer]["avg_target_sim"],
                }
            )

    fieldnames = list(rows[0].keys()) if rows else []
    combined_csv = output_dir / "model_comparison_combined.csv"
    with combined_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_model_rows(rows)
    summary["sanity_all_pass"] = all(
        row["exact_accuracy"] == 100.0
        and row["clean_reextract_accuracy"] == 100.0
        and abs(float(row["exact_avg_target_sim"]) - 1.0) < 1e-6
        and abs(float(row["clean_reextract_avg_target_sim"]) - 1.0) < 1e-6
        for row in sanity_rows
    )

    with (output_dir / "model_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "model_comparison_sanity.json").open("w", encoding="utf-8") as handle:
        json.dump(sanity_rows, handle, indent=2)

    per_model_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        per_model_rows.setdefault(str(row["model_name"]), []).append(row)
    for model_name, model_rows in per_model_rows.items():
        model_slug = model_name.replace(".", "_")
        with (output_dir / f"{model_slug}.json").open("w", encoding="utf-8") as handle:
            json.dump(model_rows, handle, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    summary = run_model_comparison(
        model_names=list(args.models),
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
