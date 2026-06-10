"""Vision frontend package for synthetic feature extraction and baseline evaluation."""

from vision.baseline_diagnostics import (
    evaluate_clean_reextract_sanity,
    evaluate_exact_image_sanity,
    summarize_perturbation_sweep,
)
from vision.feature_preprocess import LayerwisePreprocessor
from vision.image_generator import (
    StimulusRecord,
    build_preview_grid,
    corrupt_image,
    generate_square_stimuli,
    metadata_to_dicts,
    perturb_image,
)
from vision.io_naturalistic import (
    NaturalisticCategory,
    build_leave_one_exemplar_out_folds,
    load_naturalistic_category,
    load_peterson_dataset,
)
from vision.model_wrapper import VisionEmbeddingWrapper
from vision.model_comparison import (
    DEFAULT_MODEL_NAMES,
    ComparisonConfig,
    build_model_comparison_configs,
    model_poolings,
    summarize_model_rows,
)
from vision.tasks import (
    TaskBatch,
    build_balanced_splits_by_color,
    build_generalization_task_human_similarity,
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
    "DEFAULT_MODEL_NAMES",
    "ComparisonConfig",
    "NaturalisticCategory",
    "StimulusRecord",
    "TaskBatch",
    "VisionEmbeddingWrapper",
    "build_balanced_splits_by_color",
    "build_generalization_task_human_similarity",
    "build_generalization_task_synthetic",
    "build_identification_task",
    "build_leave_one_exemplar_out_folds",
    "build_model_comparison_configs",
    "build_preview_grid",
    "corrupt_image",
    "evaluate_clean_reextract_sanity",
    "evaluate_exact_image_sanity",
    "evaluate_layerwise_baseline",
    "extract_feature_bundle",
    "find_nearest_by_metric",
    "generate_square_stimuli",
    "load_naturalistic_category",
    "load_peterson_dataset",
    "load_feature_cache",
    "model_poolings",
    "metadata_to_dicts",
    "perturb_image",
    "score_against_stored",
    "subset_feature_bundle",
    "summarize_perturbation_sweep",
    "summarize_model_rows",
    "synthetic_metadata_distance",
]
