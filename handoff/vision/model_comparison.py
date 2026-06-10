from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vision.ident_corruption import (
    IDENT_DECISION_NOISE_STD,
    IDENT_MAX_SHIFT,
    IDENT_NOISE_STD,
    IDENT_OCCLUSION_FRAC,
)
from vision.synthetic_sweep import detect_tradeoff_candidates


TRANSFORMER_MODELS: tuple[str, ...] = (
    "vit_base_patch16_224",
    "vit_base_patch16_clip_224.openai",
)

CONV_MODELS: tuple[str, ...] = (
    "convnext_tiny",
)

DEFAULT_MODEL_NAMES: tuple[str, ...] = TRANSFORMER_MODELS + CONV_MODELS


@dataclass(frozen=True)
class ComparisonConfig:
    model_name: str
    pooling: str
    task_mode: str
    n_per_color: int
    n_stored_per_color: int
    square_size: int
    ident_noise_std: float
    ident_max_shift: int
    ident_occlusion_frac: float
    ident_decision_noise_std: float
    color_only_jitter_std: float


def model_poolings(model_name: str) -> tuple[str, ...]:
    if model_name in TRANSFORMER_MODELS:
        return ("cls", "mean_tokens")
    return ("auto",)


def build_model_comparison_configs(
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
) -> list[ComparisonConfig]:
    configs: list[ComparisonConfig] = []
    for model_name in model_names:
        for pooling in model_poolings(model_name):
            for task_mode in ("mixed_color_position", "color_only", "position_only"):
                jitter_values = (8.0,) if task_mode == "color_only" else (0.0,)
                for color_only_jitter_std in jitter_values:
                    for n_stored_per_color in (2, 4, 6):
                        configs.append(
                            ComparisonConfig(
                                model_name=model_name,
                                pooling=pooling,
                                task_mode=task_mode,
                                n_per_color=12,
                                n_stored_per_color=n_stored_per_color,
                                square_size=56,
                                ident_noise_std=IDENT_NOISE_STD,
                                ident_max_shift=IDENT_MAX_SHIFT,
                                ident_occlusion_frac=IDENT_OCCLUSION_FRAC,
                                ident_decision_noise_std=IDENT_DECISION_NOISE_STD,
                                color_only_jitter_std=color_only_jitter_std,
                            )
                        )
    return configs


def summarize_model_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model_name"]), []).append(row)

    best_identification = {}
    best_generalization = {}
    best_margin = {}
    for model_name, model_rows in by_model.items():
        best_identification[model_name] = max(
            model_rows,
            key=lambda row: (float(row["ident_accuracy"]), float(row["ident_avg_margin"])),
        )
        best_generalization[model_name] = max(
            model_rows,
            key=lambda row: (float(row["gen_accuracy"]), float(row["gen_avg_margin"])),
        )
        best_margin[model_name] = max(model_rows, key=lambda row: float(row["avg_margin"]))

    tradeoff_assessments = detect_tradeoff_candidates(rows)
    tradeoff_candidates = [item.__dict__ for item in tradeoff_assessments if item.qualifies]
    return {
        "models": sorted(by_model.keys()),
        "n_rows": len(rows),
        "best_identification": best_identification,
        "best_generalization": best_generalization,
        "best_margin": best_margin,
        "tradeoff_candidates": tradeoff_candidates,
        "n_tradeoff_candidates": len(tradeoff_candidates),
    }
