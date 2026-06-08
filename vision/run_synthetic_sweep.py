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
    perturb_image,
    subset_feature_bundle,
)
from vision.synthetic_sweep import (
    POSITION_ONLY_COLORS,
    SweepConfig,
    build_reduced_sweep_configs,
    build_sweep_configs,
    detect_tradeoff_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic ViT sweep and summarize tradeoff candidates.")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/vision/sweeps")
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Run the reduced CPU-friendly sweep instead of the full grid.",
    )
    return parser.parse_args()


def _stimulus_parameters(config: SweepConfig, *, image_size: int, seed: int) -> tuple[dict[str, object], int]:
    if config.task_mode == "position_only":
        run_n_per_color = config.n_per_color * 6
        colors = POSITION_ONLY_COLORS
    else:
        run_n_per_color = config.n_per_color
        colors = None

    kwargs: dict[str, object] = {
        "n_per_color": run_n_per_color,
        "image_size": image_size,
        "square_size": config.square_size,
        "fixed_position": config.task_mode == "color_only",
        "color_jitter_std": config.color_only_jitter_std if config.task_mode == "color_only" else 0.0,
        "seed": seed,
    }
    if colors is not None:
        kwargs["colors"] = colors
    return kwargs, run_n_per_color


def _get_wrapper(
    wrappers: dict[str, VisionEmbeddingWrapper],
    *,
    model_name: str,
    device: str,
    pooling: str,
) -> VisionEmbeddingWrapper:
    key = pooling
    if key not in wrappers:
        wrappers[key] = VisionEmbeddingWrapper(
            model_name=model_name,
            pretrained=True,
            device=device,
            pooling=pooling,
        )
    return wrappers[key]


def _sanitize(value: object) -> str:
    return str(value).replace(".", "p")


def _run() -> dict[str, object]:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_reduced_sweep_configs() if args.reduced else build_sweep_configs()
    wrappers: dict[str, VisionEmbeddingWrapper] = {}
    clean_cache: dict[tuple[object, ...], dict[str, object]] = {}
    noisy_cache: dict[tuple[object, ...], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []

    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {asdict(config)}")
        stimulus_kwargs, run_n_per_color = _stimulus_parameters(config, image_size=args.image_size, seed=args.seed)
        clean_key = (
            config.task_mode,
            config.pooling,
            config.n_per_color,
            config.square_size,
            config.color_only_jitter_std,
        )

        if clean_key not in clean_cache:
            wrapper = _get_wrapper(
                wrappers,
                model_name=args.model_name,
                device=args.device,
                pooling=config.pooling,
            )
            images, metadata = generate_square_stimuli(**stimulus_kwargs)
            raw_features = wrapper.extract(images, batch_size=args.batch_size)
            prep = LayerwisePreprocessor(use_zscore=False, l2_normalize=True)
            clean_features = prep.fit_transform(raw_features)
            clean_reextract_raw = wrapper.extract(images, batch_size=args.batch_size)
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

        noisy_key = clean_key + (config.ident_noise_std, config.ident_max_shift)
        if noisy_key not in noisy_cache:
            wrapper = _get_wrapper(
                wrappers,
                model_name=args.model_name,
                device=args.device,
                pooling=config.pooling,
            )
            noisy_images = [
                perturb_image(
                    image,
                    noise_std=config.ident_noise_std,
                    max_shift=config.ident_max_shift,
                    seed=args.seed + 5000 + local_idx,
                )
                for local_idx, image in enumerate(images)
            ]
            noisy_raw = wrapper.extract(noisy_images, batch_size=args.batch_size)
            noisy_cache[noisy_key] = {
                "features": prep.transform(noisy_raw),
            }

        noisy_features = noisy_cache[noisy_key]["features"]
        stored_indices, novel_probe_indices = build_balanced_splits_by_color(
            metadata,
            n_stored_per_color=config.n_stored_per_color,
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
            color_weight=0.0 if config.task_mode == "position_only" else 1.0,
            position_weight=0.0 if config.task_mode == "color_only" else 1.0,
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
        )
        gen_results = evaluate_layerwise_baseline(
            stored_feature_bundle,
            gen_probe_bundle,
            gen_task.ground_truth_idx,
            metric="cosine",
        )

        for layer in sorted(ident_results):
            row = {
                "model_name": args.model_name,
                "seed": args.seed,
                "metric": "cosine",
                "preprocess_mode": "l2_only",
                "task_mode": config.task_mode,
                "pooling": config.pooling,
                "n_per_color": config.n_per_color,
                "run_n_per_color": clean_bundle["run_n_per_color"],
                "n_stored_per_color": config.n_stored_per_color,
                "square_size": config.square_size,
                "ident_noise_std": config.ident_noise_std,
                "ident_max_shift": config.ident_max_shift,
                "color_only_jitter_std": config.color_only_jitter_std,
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

    tradeoff_assessments = detect_tradeoff_candidates(rows)
    tradeoff_candidates = [asdict(item) for item in tradeoff_assessments if item.qualifies]

    best_identification = max(rows, key=lambda row: (row["ident_accuracy"], row["ident_avg_margin"]))
    best_generalization = max(rows, key=lambda row: (row["gen_accuracy"], row["gen_avg_margin"]))
    best_margin = max(rows, key=lambda row: row["avg_margin"])
    best_tradeoff = tradeoff_candidates[0] if tradeoff_candidates else None

    all_results_path = output_dir / f"synthetic_sweep_{args.model_name}.json"
    summary_path = output_dir / f"synthetic_sweep_summary_{args.model_name}.json"
    csv_path = output_dir / f"synthetic_sweep_{args.model_name}.csv"

    payload = {
        "model_name": args.model_name,
        "seed": args.seed,
        "image_size": args.image_size,
        "preprocess_mode": "l2_only",
        "metric": "cosine",
        "configs": [asdict(config) for config in configs],
        "rows": rows,
        "sanity_rows": sanity_rows,
        "tradeoff_assessments": [asdict(item) for item in tradeoff_assessments],
    }
    all_results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_name": args.model_name,
        "seed": args.seed,
        "preprocess_mode": "l2_only",
        "metric": "cosine",
        "n_runs": len(rows),
        "n_tradeoff_candidates": len(tradeoff_candidates),
        "best_identification": best_identification,
        "best_generalization": best_generalization,
        "best_margin": best_margin,
        "best_tradeoff": best_tradeoff,
        "top_tradeoff_candidates": tradeoff_candidates[:10],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved: {all_results_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {csv_path}")
    return summary


if __name__ == "__main__":
    _run()
