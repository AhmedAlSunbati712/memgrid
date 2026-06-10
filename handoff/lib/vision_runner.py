from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from handoff.lib.paths import (
    DATA_ROOT,
    HANDOFF_ROOT,
    RESULTS_ROOT,
    ensure_handoff_paths,
)

ensure_handoff_paths()

DEFAULT_VISION_OUTPUT = RESULTS_ROOT / "vision"
GRAPH_RESULTS = RESULTS_ROOT / "image_transformers"
SIMILARITY_ORDERED_RESULTS = RESULTS_ROOT / "image_transformers_similarity_ordered"
SIMILARITY_FIXED_RESULTS = RESULTS_ROOT / "image_transformers_similarity_ordered_fixed"
DEFAULT_GRAPH_OUTPUT = GRAPH_RESULTS
DEFAULT_SIMILARITY_OUTPUT = SIMILARITY_ORDERED_RESULTS
DEFAULT_SIMILARITY_FIXED_OUTPUT = SIMILARITY_FIXED_RESULTS
DEFAULT_DATASET_PKL = DATA_ROOT / "datasets_peterson.pkl"
DEFAULT_IMAGE_ROOT = DATA_ROOT


def handoff_output(subpath: str) -> Path:
    return HANDOFF_ROOT / "results" / "vision" / subpath


def vision_results(subpath: str) -> Path:
    return handoff_output(subpath)


def run_synthetic_baseline(
    *,
    model_name: str = "vit_base_patch16_224",
    device: str = "cpu",
    batch_size: int = 8,
    image_size: int = 224,
    square_size: int = 56,
    n_per_color: int = 4,
    n_stored_per_color: int = 2,
    task_mode: str = "mixed_color_position",
    ident_noise_std: float | None = None,
    ident_max_shift: int | None = None,
    ident_occlusion_frac: float | None = None,
    decision_noise_std: float | None = None,
    seed: int = 0,
    output_dir: Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    import numpy as np

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
    from vision.ident_corruption import (
        IDENT_DECISION_NOISE_STD,
        IDENT_MAX_SHIFT,
        IDENT_NOISE_STD,
        IDENT_OCCLUSION_FRAC,
        corrupt_ident_probe,
    )
    from vision.synthetic_sweep import POSITION_ONLY_COLORS

    noise_std = IDENT_NOISE_STD if ident_noise_std is None else ident_noise_std
    max_shift = IDENT_MAX_SHIFT if ident_max_shift is None else ident_max_shift
    occlusion_frac = IDENT_OCCLUSION_FRAC if ident_occlusion_frac is None else ident_occlusion_frac
    ident_dnoise = IDENT_DECISION_NOISE_STD if decision_noise_std is None else decision_noise_std

    if quick:
        n_per_color = 2
        n_stored_per_color = 1
        batch_size = min(batch_size, 4)

    out = output_dir or handoff_output("synthetic")
    out.mkdir(parents=True, exist_ok=True)

    run_n_per_color = n_per_color * 6 if task_mode == "position_only" else n_per_color
    run_n_stored = run_n_per_color // 2 if task_mode == "position_only" else n_stored_per_color
    colors = POSITION_ONLY_COLORS if task_mode == "position_only" else None
    fixed_position = task_mode == "color_only"

    stimulus_kwargs: dict[str, Any] = {
        "n_per_color": run_n_per_color,
        "image_size": image_size,
        "square_size": square_size,
        "seed": seed,
        "fixed_position": fixed_position,
    }
    if colors is not None:
        stimulus_kwargs["colors"] = colors
    images, metadata = generate_square_stimuli(**stimulus_kwargs)
    wrapper = VisionEmbeddingWrapper(model_name=model_name, pretrained=True, device=device, pooling="auto")
    raw_features = wrapper.extract(images, batch_size=batch_size)
    prep = LayerwisePreprocessor(use_zscore=False, l2_normalize=True)
    features = prep.fit_transform(raw_features)

    noisy_images = [
        corrupt_ident_probe(
            image,
            seed=seed + idx,
            noise_std=noise_std,
            max_shift=max_shift,
            occlusion_frac=occlusion_frac,
        )
        for idx, image in enumerate(images)
    ]
    noisy_features = prep.transform(wrapper.extract(noisy_images, batch_size=batch_size))
    clean_reextract = prep.transform(wrapper.extract(images, batch_size=batch_size))

    stored_indices, novel_probe_indices = build_balanced_splits_by_color(
        metadata,
        n_stored_per_color=run_n_stored,
    )
    ident_task = build_identification_task(metadata, stored_indices=stored_indices, probe_indices=stored_indices)
    gen_task = build_generalization_task_synthetic(
        metadata,
        stored_indices=stored_indices,
        probe_indices=novel_probe_indices,
        color_weight=0.0 if task_mode == "position_only" else 1.0,
        position_weight=0.0 if task_mode == "color_only" else 1.0,
    )

    stored_feature_bundle = subset_feature_bundle(features, ident_task.stored_indices)
    ident_probe_bundle = subset_feature_bundle(noisy_features, ident_task.probe_indices)
    clean_probe_bundle = subset_feature_bundle(clean_reextract, ident_task.probe_indices)
    gen_probe_bundle = subset_feature_bundle(features, gen_task.probe_indices)

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
        decision_noise_std=ident_dnoise,
        decision_noise_seed=seed + 9000,
    )
    gen_results = evaluate_layerwise_baseline(
        stored_feature_bundle,
        gen_probe_bundle,
        gen_task.ground_truth_idx,
        metric="cosine",
    )

    layer_rows: list[dict[str, Any]] = []
    for layer_name in sorted(ident_results):
        layer_rows.append(
            {
                "layer": layer_name,
                "ident_accuracy": ident_results[layer_name]["accuracy"],
                "gen_accuracy": gen_results[layer_name]["accuracy"],
                "ident_avg_margin": ident_results[layer_name]["avg_margin"],
                "gen_avg_margin": gen_results[layer_name]["avg_margin"],
            }
        )

    summary = {
        "model_name": model_name,
        "task_mode": task_mode,
        "ident_corruption": {
            "noise_std": noise_std,
            "max_shift": max_shift,
            "occlusion_frac": occlusion_frac,
            "decision_noise_std": ident_dnoise,
        },
        "layers": layer_rows,
        "exact_image_sanity": exact_sanity,
        "clean_reextract_sanity": clean_reextract_sanity,
    }
    summary_path = out / f"baseline_summary_{model_name.replace('.', '_')}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {"summary_path": str(summary_path), "summary": summary}


def dispatch_model_comparison(**kwargs: Any) -> dict[str, Any]:
    from vision.run_model_comparison import run_model_comparison

    kwargs.setdefault("batch_size", 16)
    kwargs.setdefault("image_size", 224)
    output_dir = Path(kwargs.pop("output_dir", handoff_output("model_comparison")))
    return run_model_comparison(output_dir=output_dir, **kwargs)


def dispatch_naturalistic_baseline(**kwargs: Any) -> dict[str, Any]:
    from vision.ident_corruption import (
        IDENT_CORRUPTION_MODE,
        IDENT_DECISION_NOISE_STD,
        IDENT_MAX_SHIFT,
        IDENT_NOISE_STD,
        IDENT_OCCLUSION_FRAC,
    )
    from vision.run_naturalistic_baseline import get_model_specs, run_naturalistic_baseline

    include_clip = kwargs.pop("include_clip", False)
    kwargs.setdefault("batch_size", 16)
    kwargs.setdefault("dataset_pkl", str(DEFAULT_DATASET_PKL))
    kwargs.setdefault("image_root", str(DEFAULT_IMAGE_ROOT))
    kwargs.setdefault("corruption_mode", IDENT_CORRUPTION_MODE)
    kwargs.setdefault("noise_std", IDENT_NOISE_STD)
    kwargs.setdefault("max_shift", IDENT_MAX_SHIFT)
    kwargs.setdefault("occlusion_frac", IDENT_OCCLUSION_FRAC)
    kwargs.setdefault("decision_noise_std", IDENT_DECISION_NOISE_STD)
    kwargs["output_dir"] = Path(kwargs.pop("output_dir", handoff_output("naturalistic")))
    kwargs["model_specs"] = kwargs.pop("model_specs", get_model_specs(include_clip=include_clip))
    return run_naturalistic_baseline(**kwargs)


def naturalistic_dam_notebook_overrides(*, quick: bool) -> dict[str, Any]:
    """Tune DAM live runs from the notebook (CLI keeps full sweep via flags)."""
    if quick:
        return {
            "category": "animals",
            "max_stage_a_configs": 1,
            "skip_stage_b": True,
            "include_clip": False,
            "n_seeds": 1,
            "storage_sizes": [40],
        }
    return {
        "max_stage_a_configs": 6,
        "skip_stage_b": True,
        "n_seeds": 1,
        "storage_sizes": [40],
        "include_clip": True,
    }


def dispatch_naturalistic_dam(**kwargs: Any) -> dict[str, Any]:
    from vision.naturalistic_dam import summarize_dam_cross_category, summarize_encoder_head_to_head
    from vision.run_naturalistic_dam import run_naturalistic_dam

    category = kwargs.pop("category", None)
    categories = [category] if category else ["fruits", "vegetables", "animals"]
    output_dir = Path(kwargs.pop("output_dir", handoff_output("naturalistic_dam")))
    kwargs.setdefault("batch_size", 16)
    kwargs.setdefault("dataset_pkl", str(DEFAULT_DATASET_PKL))
    kwargs.setdefault("image_root", str(DEFAULT_IMAGE_ROOT))
    kwargs.setdefault("backend", "numba")
    kwargs["output_dir"] = output_dir
    kwargs["baseline_dir"] = Path(kwargs.pop("baseline_dir", handoff_output("naturalistic")))
    include_clip = kwargs.pop("include_clip", True)
    model_specs_override = kwargs.pop("model_specs_override", None)
    summaries = [
        run_naturalistic_dam(
            category=cat,
            include_clip=include_clip,
            model_specs_override=model_specs_override,
            **kwargs,
        )
        for cat in categories
    ]
    if len(categories) > 1:
        summarize_dam_cross_category(output_dir, categories)
        summarize_encoder_head_to_head(output_dir, categories)
    return {"summaries": summaries}


def dispatch_animals_graphs(**kwargs: Any) -> list[Path]:
    from vision.run_animals_image_transformer_graphs import run_suite

    kwargs.setdefault("batch_size", 16)
    kwargs.setdefault("dataset_pkl", str(DEFAULT_DATASET_PKL))
    kwargs.setdefault("image_root", str(DEFAULT_IMAGE_ROOT))
    kwargs.setdefault("backend", "numba")
    kwargs.setdefault("n_seeds", 5)
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("seed", 0)
    kwargs["output_dir"] = Path(kwargs.pop("output_dir", DEFAULT_GRAPH_OUTPUT))
    return run_suite(**kwargs)


def dispatch_animals_similarity_ordered(**kwargs: Any) -> list[Path]:
    from handoff.lib.graph_display import LEGACY_ANIMALS_SETTING_NAMES
    from vision.run_animals_image_transformer_similarity_ordered import run_suite

    kwargs.setdefault("batch_size", 16)
    kwargs.setdefault("dataset_pkl", str(DEFAULT_DATASET_PKL))
    kwargs.setdefault("image_root", str(DEFAULT_IMAGE_ROOT))
    kwargs.setdefault("backend", "numba")
    kwargs.setdefault("n_seeds", 5)
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("setting_names", LEGACY_ANIMALS_SETTING_NAMES)
    kwargs["output_dir"] = Path(kwargs.pop("output_dir", DEFAULT_SIMILARITY_OUTPUT))
    return run_suite(**kwargs)


def dispatch_animals_similarity_ordered_fixed(**kwargs: Any) -> list[Path]:
    from handoff.lib.graph_display import LEGACY_ANIMALS_SETTING_NAMES
    from vision.run_animals_image_transformer_similarity_ordered_fixed import run_suite

    kwargs.setdefault("setting_names", LEGACY_ANIMALS_SETTING_NAMES)
    kwargs.setdefault("input_dir", DEFAULT_SIMILARITY_OUTPUT)
    kwargs["input_dir"] = Path(kwargs.pop("input_dir"))
    kwargs["output_dir"] = Path(kwargs.pop("output_dir", DEFAULT_SIMILARITY_FIXED_OUTPUT))
    return run_suite(**kwargs)
