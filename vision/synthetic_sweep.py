from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


POSITION_ONLY_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (("neutral", (180, 180, 180)),)


@dataclass(frozen=True)
class SweepConfig:
    task_mode: str
    pooling: str
    n_per_color: int
    n_stored_per_color: int
    square_size: int
    ident_noise_std: float
    ident_max_shift: int
    color_only_jitter_std: float


@dataclass(frozen=True)
class TradeoffAssessment:
    task_mode: str
    pooling: str
    square_size: int
    ident_noise_std: float
    ident_max_shift: int
    color_only_jitter_std: float
    layer: str
    qualifies: bool
    reason: str
    peak_index: int | None
    peak_generalization_accuracy: float
    final_identification_accuracy: float
    post_peak_generalization_drop: float
    positive_margin_count: int
    mean_positive_margin: float


def _stored_count_choices(n_per_color: int) -> list[int]:
    choices = sorted({2, 4, n_per_color // 2})
    return [value for value in choices if 0 < value < n_per_color]


def build_sweep_configs() -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for task_mode in ("mixed_color_position", "color_only", "position_only"):
        for pooling in ("cls", "mean_tokens"):
            for n_per_color in (8, 12):
                for square_size in (32, 56):
                    for ident_noise_std in (0.0, 2.0, 5.0):
                        for ident_max_shift in (0, 1, 2):
                            jitter_values = (4.0, 8.0, 16.0) if task_mode == "color_only" else (0.0,)
                            for color_only_jitter_std in jitter_values:
                                for n_stored_per_color in _stored_count_choices(n_per_color):
                                    configs.append(
                                        SweepConfig(
                                            task_mode=task_mode,
                                            pooling=pooling,
                                            n_per_color=n_per_color,
                                            n_stored_per_color=n_stored_per_color,
                                            square_size=square_size,
                                            ident_noise_std=ident_noise_std,
                                            ident_max_shift=ident_max_shift,
                                            color_only_jitter_std=color_only_jitter_std,
                                        )
                                    )
    return configs


def build_reduced_sweep_configs() -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for task_mode in ("mixed_color_position", "color_only", "position_only"):
        for pooling in ("cls", "mean_tokens"):
            n_per_color = 12
            for square_size in (56,):
                for ident_noise_std in (0.0, 5.0):
                    for ident_max_shift in (0, 2):
                        jitter_values = (8.0,) if task_mode == "color_only" else (0.0,)
                        for color_only_jitter_std in jitter_values:
                            for n_stored_per_color in _stored_count_choices(n_per_color):
                                configs.append(
                                    SweepConfig(
                                        task_mode=task_mode,
                                        pooling=pooling,
                                        n_per_color=n_per_color,
                                        n_stored_per_color=n_stored_per_color,
                                        square_size=square_size,
                                        ident_noise_std=ident_noise_std,
                                        ident_max_shift=ident_max_shift,
                                        color_only_jitter_std=color_only_jitter_std,
                                    )
                                )
    return configs


def summarize_config(config: SweepConfig) -> dict[str, object]:
    return asdict(config)


def detect_tradeoff_candidates(
    rows: Iterable[dict[str, object]],
    *,
    minimum_drop: float = 10.0,
) -> list[TradeoffAssessment]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["task_mode"],
            row["pooling"],
            row["square_size"],
            row["ident_noise_std"],
            row["ident_max_shift"],
            row["color_only_jitter_std"],
            row["layer"],
        )
        grouped.setdefault(key, []).append(row)

    assessments: list[TradeoffAssessment] = []
    for key, group_rows in grouped.items():
        ordered = sorted(group_rows, key=lambda row: int(row["n_stored_per_color"]))
        gen_curve = [float(row["gen_accuracy"]) for row in ordered]
        ident_curve = [float(row["ident_accuracy"]) for row in ordered]
        margins = [float(row["avg_margin"]) for row in ordered]

        peak_generalization_accuracy = max(gen_curve)
        peak_indices = [idx for idx, value in enumerate(gen_curve) if value == peak_generalization_accuracy]
        positive_margin_values = [value for value in margins if value > 0.0]
        positive_margin_count = len(positive_margin_values)
        mean_positive_margin = (
            sum(positive_margin_values) / positive_margin_count if positive_margin_values else 0.0
        )

        if not peak_indices:
            assessments.append(
                TradeoffAssessment(
                    task_mode=str(key[0]),
                    pooling=str(key[1]),
                    square_size=int(key[2]),
                    ident_noise_std=float(key[3]),
                    ident_max_shift=int(key[4]),
                    color_only_jitter_std=float(key[5]),
                    layer=str(key[6]),
                    qualifies=False,
                    reason="no_peak",
                    peak_index=None,
                    peak_generalization_accuracy=peak_generalization_accuracy,
                    final_identification_accuracy=ident_curve[-1],
                    post_peak_generalization_drop=0.0,
                    positive_margin_count=positive_margin_count,
                    mean_positive_margin=mean_positive_margin,
                )
            )
            continue

        peak_index = peak_indices[0]
        if peak_index == 0 or peak_index == len(gen_curve) - 1:
            reason = "peak_not_interior"
            qualifies = False
            drop = 0.0
        else:
            drop = peak_generalization_accuracy - gen_curve[-1]
            qualifies = (
                ident_curve[-1] > ident_curve[peak_index]
                and drop >= minimum_drop
                and positive_margin_count > 0
            )
            if not qualifies:
                if ident_curve[-1] <= ident_curve[peak_index]:
                    reason = "identification_does_not_continue_improving"
                elif drop < minimum_drop:
                    reason = "post_peak_drop_too_small"
                else:
                    reason = "no_positive_margin"
            else:
                reason = "qualifies"

        assessments.append(
            TradeoffAssessment(
                task_mode=str(key[0]),
                pooling=str(key[1]),
                square_size=int(key[2]),
                ident_noise_std=float(key[3]),
                ident_max_shift=int(key[4]),
                color_only_jitter_std=float(key[5]),
                layer=str(key[6]),
                qualifies=qualifies,
                reason=reason,
                peak_index=peak_index,
                peak_generalization_accuracy=peak_generalization_accuracy,
                final_identification_accuracy=ident_curve[-1],
                post_peak_generalization_drop=drop,
                positive_margin_count=positive_margin_count,
                mean_positive_margin=mean_positive_margin,
            )
        )

    return sorted(
        assessments,
        key=lambda item: (
            not item.qualifies,
            -item.post_peak_generalization_drop,
            -item.final_identification_accuracy,
            -item.mean_positive_margin,
        ),
    )
