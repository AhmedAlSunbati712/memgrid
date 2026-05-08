"""Vision frontend package for synthetic feature extraction and baseline evaluation."""

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
    evaluate_layerwise_baseline,
    extract_feature_bundle,
    find_nearest_by_metric,
    load_feature_cache,
    score_against_stored,
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
    "evaluate_layerwise_baseline",
    "extract_feature_bundle",
    "find_nearest_by_metric",
    "generate_square_stimuli",
    "load_feature_cache",
    "metadata_to_dicts",
    "perturb_image",
    "score_against_stored",
    "subset_feature_bundle",
    "synthetic_metadata_distance",
]
