"""Vision frontend package for synthetic feature extraction and DAM evaluation."""

from vision.feature_preprocess import LayerwisePreprocessor
from vision.image_generator import StimulusRecord, build_preview_grid, generate_square_stimuli, metadata_to_dicts, perturb_image
from vision.model_wrapper import VisionEmbeddingWrapper
from vision.tasks import (
    TaskBatch,
    build_balanced_splits_by_color,
    build_generalization_task_synthetic,
    build_identification_task,
    synthetic_metadata_distance,
)
from vision.vision_experiments import (
    DEFAULT_COLLAPSE_THRESHOLD,
    DEFAULT_OVERSHOOT_RATIO,
    compute_retrieved_norm_ratio,
    evaluate_layerwise_baseline,
    extract_feature_bundle,
    find_nearest_by_metric,
    is_retrieval_calibrated,
    is_retrieval_collapsed,
    is_retrieval_overshooting,
    load_feature_cache,
    rank_sweep_rows,
    run_layerwise_generalization,
    run_layerwise_identification,
    score_against_stored,
    sweep_layerwise_dam_configs,
    subset_feature_bundle,
)

__all__ = [
    "LayerwisePreprocessor",
    "StimulusRecord",
    "TaskBatch",
    "VisionEmbeddingWrapper",
    "build_balanced_splits_by_color",
    "build_generalization_task_synthetic",
    "build_identification_task",
    "build_preview_grid",
    "DEFAULT_COLLAPSE_THRESHOLD",
    "DEFAULT_OVERSHOOT_RATIO",
    "compute_retrieved_norm_ratio",
    "evaluate_layerwise_baseline",
    "extract_feature_bundle",
    "find_nearest_by_metric",
    "generate_square_stimuli",
    "is_retrieval_calibrated",
    "is_retrieval_collapsed",
    "is_retrieval_overshooting",
    "load_feature_cache",
    "metadata_to_dicts",
    "perturb_image",
    "rank_sweep_rows",
    "run_layerwise_generalization",
    "run_layerwise_identification",
    "score_against_stored",
    "subset_feature_bundle",
    "sweep_layerwise_dam_configs",
    "synthetic_metadata_distance",
]
