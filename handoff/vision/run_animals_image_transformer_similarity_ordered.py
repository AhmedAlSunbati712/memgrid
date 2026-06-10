from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import (
    LayerwisePreprocessor,
    VisionEmbeddingWrapper,
    build_generalization_task_human_similarity,
    build_identification_task,
    corrupt_image,
    evaluate_layerwise_baseline,
    load_naturalistic_category,
    subset_feature_bundle,
)
from vision.naturalistic_dam import _evaluate_dam_on_bundle
from vision.run_naturalistic_baseline import _random_storage_splits


MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("vit_base_patch16_224", "cls"),
    ("vit_base_patch16_clip_224.openai", "cls"),
)

DAM_ORDERS: tuple[int, ...] = (2, 4, 6, 8)
DAM_FIXED_PARAMS: dict[str, float | int] = {
    "beta": 0.02,
    "alpha": 0.1,
    "lmbda": 0.05,
    "steps_multiplier": 2,
}


@dataclass(frozen=True)
class SettingSpec:
    name: str
    n_stored: int
    corruption_mode: str
    noise_std: float = 5.0
    max_shift: int = 2
    occlusion_frac: float = 0.0
    mask_frac: float = 0.0
    n_chunks: int = 0
    affine_strength: float = 0.0
    decision_noise_std: float = 0.0


SETTING_SPECS: tuple[SettingSpec, ...] = (
    SettingSpec(name="easy_s40", n_stored=40, corruption_mode="noise_shift", noise_std=5.0, max_shift=2),
    SettingSpec(name="easy_s80", n_stored=80, corruption_mode="noise_shift", noise_std=5.0, max_shift=2),
    SettingSpec(name="easy_s100", n_stored=100, corruption_mode="noise_shift", noise_std=5.0, max_shift=2),
    SettingSpec(name="occ50_s40", n_stored=40, corruption_mode="occlusion", occlusion_frac=0.5),
    SettingSpec(name="occ50_s80", n_stored=80, corruption_mode="occlusion", occlusion_frac=0.5),
    SettingSpec(name="occ50_s100", n_stored=100, corruption_mode="occlusion", occlusion_frac=0.5),
    SettingSpec(name="occ50_dnoise001_s40", n_stored=40, corruption_mode="occlusion", occlusion_frac=0.5, decision_noise_std=0.01),
    SettingSpec(name="occ50_dnoise001_s80", n_stored=80, corruption_mode="occlusion", occlusion_frac=0.5, decision_noise_std=0.01),
    SettingSpec(name="occ50_dnoise001_s100", n_stored=100, corruption_mode="occlusion", occlusion_frac=0.5, decision_noise_std=0.01),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate similarity-ordered animals-only layerwise tradeoff plots for ViT and CLIP."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="result/image_transformers_similarity_ordered")
    parser.add_argument("--dataset-pkl", default="datasets_peterson.pkl")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numba")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-settings", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--max-dam-orders", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    return parser.parse_args()


def _load_images(paths: Iterable[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _layer_index(layer_name: str) -> int:
    return int(str(layer_name).split("_")[1])


def _all_layer_indices(wrapper: VisionEmbeddingWrapper, max_layers: int | None = None) -> list[int]:
    if not wrapper.is_transformer_like:
        raise ValueError(f"{wrapper.model_name} is not transformer-like; expected ViT/CLIP transformer.")
    total_layers = len(wrapper.model.blocks)
    indices = list(range(total_layers))
    if max_layers is not None:
        indices = indices[:max_layers]
    return indices


def _corrupt_images(images: list[Image.Image], setting: SettingSpec, seed: int) -> list[Image.Image]:
    return [
        corrupt_image(
            image,
            mode=setting.corruption_mode,
            seed=seed + 1000 + idx,
            noise_std=setting.noise_std,
            max_shift=setting.max_shift,
            occlusion_frac=setting.occlusion_frac,
            mask_frac=setting.mask_frac,
            n_chunks=setting.n_chunks,
            affine_strength=setting.affine_strength,
        )
        for idx, image in enumerate(images)
    ]


def _mean_pairwise_cosine_stats(stored: np.ndarray) -> tuple[float, float, float, float]:
    stored = np.asarray(stored, dtype=np.float64)
    if stored.ndim != 2 or stored.shape[0] < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sims = stored @ stored.T
    tri = np.triu_indices(stored.shape[0], k=1)
    pairwise_sim = sims[tri]
    pairwise_dist = 1.0 - pairwise_sim
    return (
        float(np.mean(pairwise_sim)),
        float(np.mean(pairwise_dist)),
        float(np.std(pairwise_sim)),
        float(np.std(pairwise_dist)),
    )


def _aggregate_rows(rows: list[dict[str, object]], group_keys: tuple[str, ...], metric_keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    out: list[dict[str, object]] = []
    for group_rows in grouped.values():
        sample = group_rows[0]
        agg = {key: sample[key] for key in group_keys}
        agg["n_splits"] = len(group_rows)
        for metric in metric_keys:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
            agg[f"{metric}_mean"] = float(np.mean(values))
            agg[f"{metric}_std"] = float(np.std(values))
        out.append(agg)
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rank_tercile(rank_idx: int, n_total: int) -> str:
    if n_total <= 0:
        return "unknown"
    if rank_idx < n_total / 3.0:
        return "low"
    if rank_idx < 2.0 * n_total / 3.0:
        return "middle"
    return "high"


def _spearman_corr(x_vals: list[float], y_vals: list[float]) -> float:
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    corr = np.corrcoef(xr, yr)[0, 1]
    return float(corr)


def _classify_tradeoff(ident_spearman: float, best_gen_tercile: str) -> str:
    monotonic_id = np.isfinite(ident_spearman) and ident_spearman > 0.3
    mid_gen_peak = best_gen_tercile == "middle"
    if monotonic_id and mid_gen_peak:
        return "tradeoff_supported"
    if monotonic_id or mid_gen_peak:
        return "tradeoff_partial"
    return "tradeoff_not_supported"


def _build_tradeoff_summary(
    rows: list[dict[str, object]],
    *,
    kind: str,
) -> list[dict[str, object]]:
    group_keys = ["setting_name", "model_name"]
    if kind == "dam":
        group_keys.append("dam_n")

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    out: list[dict[str, object]] = []
    for key, group in grouped.items():
        group_sorted = sorted(group, key=lambda row: float(row["mean_pairwise_cosine_distance_mean"]))
        ident_spearman = _spearman_corr(
            [float(row["mean_pairwise_cosine_distance_mean"]) for row in group_sorted],
            [float(row["ident_accuracy_mean"]) for row in group_sorted],
        )
        gen_spearman = _spearman_corr(
            [float(row["mean_pairwise_cosine_distance_mean"]) for row in group_sorted],
            [float(row["gen_accuracy_mean"]) for row in group_sorted],
        )
        best_gen_idx = int(np.argmax([float(row["gen_accuracy_mean"]) for row in group_sorted]))
        best_gen_row = group_sorted[best_gen_idx]
        best_gen_tercile = _rank_tercile(best_gen_idx, len(group_sorted))
        summary = {
            "kind": kind,
            "setting_name": group_sorted[0]["setting_name"],
            "model_name": group_sorted[0]["model_name"],
            "dam_n": int(group_sorted[0]["dam_n"]) if kind == "dam" else "",
            "ident_vs_distance_spearman": ident_spearman,
            "gen_vs_distance_spearman": gen_spearman,
            "best_gen_layer": best_gen_row["layer"],
            "best_gen_distance": float(best_gen_row["mean_pairwise_cosine_distance_mean"]),
            "best_gen_tercile": best_gen_tercile,
            "classification": _classify_tradeoff(ident_spearman, best_gen_tercile),
        }
        out.append(summary)
    return out


def _setting_subtitle(setting: SettingSpec, *, n_probe: int, n_seeds: int) -> list[str]:
    return [
        f"dataset=animals | n_stored={setting.n_stored} | n_probe={n_probe} | n_seeds={n_seeds}",
        (
            f"corruption={setting.corruption_mode}"
            f", noise_std={setting.noise_std:g}, max_shift={setting.max_shift}"
            f", occlusion_frac={setting.occlusion_frac:g}, decision_noise={setting.decision_noise_std:g}"
        ),
        (
            f"DAM fixed: beta={float(DAM_FIXED_PARAMS['beta']):g}, alpha={float(DAM_FIXED_PARAMS['alpha']):g}, "
            f"lambda={float(DAM_FIXED_PARAMS['lmbda']):g}, steps_multiplier={int(DAM_FIXED_PARAMS['steps_multiplier'])}"
        ),
    ]


def _plot_accuracy_vs_distance(
    *,
    rows: list[dict[str, object]],
    title: str,
    subtitle_lines: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    styles = {
        ("vit_base_patch16_224", "ident"): {"color": "#1f77b4", "marker": "o", "label": "ViT ident"},
        ("vit_base_patch16_224", "gen"): {"color": "#1f77b4", "marker": "^", "label": "ViT gen", "linestyle": "--"},
        ("vit_base_patch16_clip_224.openai", "ident"): {"color": "#d62728", "marker": "s", "label": "CLIP ident"},
        ("vit_base_patch16_clip_224.openai", "gen"): {"color": "#d62728", "marker": "D", "label": "CLIP gen", "linestyle": "--"},
    }
    for model_name in ("vit_base_patch16_224", "vit_base_patch16_clip_224.openai"):
        subset = sorted(rows, key=lambda row: float(row["mean_pairwise_cosine_distance_mean"]))
        subset = [row for row in subset if row["model_name"] == model_name]
        x_vals = [float(row["mean_pairwise_cosine_distance_mean"]) for row in subset]
        ident_vals = [float(row["ident_accuracy_mean"]) for row in subset]
        gen_vals = [float(row["gen_accuracy_mean"]) for row in subset]
        for metric_name, y_vals in (("ident", ident_vals), ("gen", gen_vals)):
            style = styles[(model_name, metric_name)]
            ax.plot(
                x_vals,
                y_vals,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.5,
                linestyle=style.get("linestyle", "-"),
                label=style["label"],
            )
            for row, x_val, y_val in zip(subset, x_vals, y_vals):
                ax.annotate(
                    f"L{_layer_index(str(row['layer']))}",
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color=style["color"],
                )
    ax.set_xlabel("Mean Pairwise Cosine Distance Among Stored Patterns")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.text(0.02, 0.02, "\n".join(subtitle_lines), fontsize=9, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_scatter_colored_by_distance(
    *,
    rows: list[dict[str, object]],
    title: str,
    subtitle_lines: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap_specs = {
        "vit_base_patch16_224": ("Blues", "ViT"),
        "vit_base_patch16_clip_224.openai": ("Reds", "CLIP"),
    }
    for model_name, (cmap_name, label) in cmap_specs.items():
        subset = [row for row in rows if row["model_name"] == model_name]
        subset.sort(key=lambda row: float(row["mean_pairwise_cosine_distance_mean"]))
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(0.35 + 0.55 * (idx / max(1, len(subset) - 1))) for idx in range(len(subset))]
        x_vals = [float(row["gen_accuracy_mean"]) for row in subset]
        y_vals = [float(row["ident_accuracy_mean"]) for row in subset]
        ax.plot(x_vals, y_vals, color=colors[-1], linewidth=1.2, alpha=0.8, label=f"{label} ordered by distance")
        for idx, (row, x_val, y_val) in enumerate(zip(subset, x_vals, y_vals)):
            ax.scatter([x_val], [y_val], color=[colors[idx]], marker="o" if "clip" not in model_name else "s", s=40)
            ax.annotate(
                f"L{_layer_index(str(row['layer']))}",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=colors[idx],
            )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Generalization Accuracy (%)")
    ax.set_ylabel("Identification Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.text(0.02, 0.02, "\n".join(subtitle_lines), fontsize=9, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _format_model_label(model_name: str) -> str:
    return "ViT" if "clip" not in model_name else "CLIP"


def _build_setting_insights(
    setting: SettingSpec,
    baseline_agg: list[dict[str, object]],
    dam_agg: list[dict[str, object]],
    baseline_tradeoff: list[dict[str, object]],
    dam_tradeoff: list[dict[str, object]],
    dam_orders: list[int],
) -> str:
    lines: list[str] = ["# Insights", "", "## Setting"]
    lines.append(
        f"Animals only, `{setting.name}`, `n_stored={setting.n_stored}`, "
        f"`corruption={setting.corruption_mode}`, `decision_noise_std={setting.decision_noise_std:g}`."
    )
    lines.extend(["", "## Baseline geometry trend"])
    for model_name in ("vit_base_patch16_224", "vit_base_patch16_clip_224.openai"):
        summary = next(row for row in baseline_tradeoff if row["model_name"] == model_name)
        lines.append(
            f"- {_format_model_label(model_name)}: ident-vs-distance Spearman={float(summary['ident_vs_distance_spearman']):.2f}, "
            f"gen-vs-distance Spearman={float(summary['gen_vs_distance_spearman']):.2f}, "
            f"best gen at `{summary['best_gen_layer']}` in the `{summary['best_gen_tercile']}` distance tercile, "
            f"classification=`{summary['classification']}`."
        )
    lines.extend(["", "## DAM geometry trend by n"])
    for dam_n in dam_orders:
        parts: list[str] = []
        for model_name in ("vit_base_patch16_224", "vit_base_patch16_clip_224.openai"):
            summary = next(
                row for row in dam_tradeoff if row["model_name"] == model_name and int(row["dam_n"]) == dam_n
            )
            base_summary = next(row for row in baseline_tradeoff if row["model_name"] == model_name)
            base_best = max(
                [row for row in baseline_agg if row["model_name"] == model_name],
                key=lambda row: float(row["gen_accuracy_mean"]),
            )
            dam_best = max(
                [row for row in dam_agg if row["model_name"] == model_name and int(row["dam_n"]) == dam_n],
                key=lambda row: float(row["gen_accuracy_mean"]),
            )
            dx = float(dam_best["gen_accuracy_mean"]) - float(base_best["gen_accuracy_mean"])
            dy = float(dam_best["ident_accuracy_mean"]) - float(base_best["ident_accuracy_mean"])
            shift = _shift_label(dx, dy)
            parts.append(
                f"{_format_model_label(model_name)}: identρ={float(summary['ident_vs_distance_spearman']):.2f}, "
                f"genρ={float(summary['gen_vs_distance_spearman']):.2f}, best-gen-tercile=`{summary['best_gen_tercile']}`, "
                f"class=`{summary['classification']}`, best-point-shift=`{shift}` vs baseline `{base_summary['classification']}`"
            )
        lines.append(f"- `n={dam_n}`: " + "; ".join(parts) + ".")
    lines.extend(["", "## Tradeoff assessment"])
    for model_name in ("vit_base_patch16_224", "vit_base_patch16_clip_224.openai"):
        base_summary = next(row for row in baseline_tradeoff if row["model_name"] == model_name)
        best_dam_summary = max(
            [row for row in dam_tradeoff if row["model_name"] == model_name],
            key=lambda row: (row["classification"] == "tradeoff_supported", row["classification"] == "tradeoff_partial"),
        )
        lines.append(
            f"- {_format_model_label(model_name)} baseline=`{base_summary['classification']}`; "
            f"best DAM classification=`{best_dam_summary['classification']}` at `n={best_dam_summary['dam_n']}`."
        )
    return "\n".join(lines) + "\n"


def _shift_label(dx: float, dy: float) -> str:
    x = "right" if dx > 0.5 else "left" if dx < -0.5 else "flat"
    y = "up" if dy > 0.5 else "down" if dy < -0.5 else "flat"
    return f"{x}/{y}"


def _write_config(
    *,
    setting_dir: Path,
    setting: SettingSpec,
    n_probe: int,
    n_seeds: int,
    dam_orders: list[int],
    layer_names: list[str],
    dims_by_model: dict[str, dict[str, int]],
) -> None:
    lines = [
        f"generated_at: {datetime.now().isoformat()}",
        "dataset: animals",
        "models:",
        "  - vit_base_patch16_224",
        "  - vit_base_patch16_clip_224.openai",
        "pooling: cls",
        f"layers: {', '.join(layer_names)}",
        "split_mode: random_storage_curve",
        f"n_stored_baseline: {setting.n_stored}",
        f"n_stored_dam: {setting.n_stored}",
        f"n_probe: {n_probe}",
        f"n_seeds: {n_seeds}",
        f"corruption_mode: {setting.corruption_mode}",
        f"noise_std: {setting.noise_std}",
        f"max_shift: {setting.max_shift}",
        f"occlusion_frac: {setting.occlusion_frac}",
        f"mask_frac: {setting.mask_frac}",
        f"n_chunks: {setting.n_chunks}",
        f"affine_strength: {setting.affine_strength}",
        f"decision_noise_std: {setting.decision_noise_std}",
        "stored_geometry_metric: mean_pairwise_cosine_distance",
        f"dam_energy_orders: {', '.join(str(n) for n in dam_orders)}",
        f"dam_beta: {DAM_FIXED_PARAMS['beta']}",
        f"dam_alpha: {DAM_FIXED_PARAMS['alpha']}",
        f"dam_lmbda: {DAM_FIXED_PARAMS['lmbda']}",
        f"dam_steps_multiplier: {DAM_FIXED_PARAMS['steps_multiplier']}",
        "neurons_by_model_and_layer:",
    ]
    for model_name, layer_dims in dims_by_model.items():
        lines.append(f"  {model_name}:")
        for layer_name in layer_names:
            dim = layer_dims[layer_name]
            lines.append(
                f"    {layer_name}: dim={dim}, steps={int(DAM_FIXED_PARAMS['steps_multiplier']) * dim}"
            )
    lines.extend(
        [
            "files:",
            "  - baseline_raw.csv",
            "  - baseline_aggregated.csv",
            "  - dam_raw.csv",
            "  - dam_aggregated.csv",
            "  - tradeoff_summary.csv",
            "  - baseline_accuracy_vs_pairwise_distance.png",
            "  - baseline_layers_gen_vs_ident_colored_by_distance.png",
            "  - insights.md",
        ]
    )
    for dam_n in dam_orders:
        lines.insert(-1, f"  - dam_n{dam_n}_accuracy_vs_pairwise_distance.png")
        lines.insert(-1, f"  - dam_n{dam_n}_layers_gen_vs_ident_colored_by_distance.png")
    (setting_dir / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_readme(root_dir: Path, settings: list[SettingSpec]) -> None:
    lines = [
        "# Image Transformer Similarity-Ordered Tradeoff Suite",
        "",
        "Animals-only naturalistic layerwise plots for ViT and CLIP, ordered by stored-pattern pairwise cosine distance.",
        "",
        "Each subdirectory contains:",
        "- baseline and DAM raw/aggregated CSVs",
        "- accuracy-vs-pairwise-distance plots",
        "- gen-vs-ident scatter plots colored and connected by distance order",
        "- tradeoff summary CSV",
        "- config and concise insights",
        "",
        "Settings:",
    ]
    for setting in settings:
        lines.append(f"- `{setting.name}`")
    (root_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_overall_insights(
    baseline_tradeoff: list[dict[str, object]],
    dam_tradeoff: list[dict[str, object]],
) -> str:
    def _count(rows: list[dict[str, object]], *, model_name: str, classification: str) -> int:
        return sum(1 for row in rows if row["model_name"] == model_name and row["classification"] == classification)

    lines = ["# Overall Insights", ""]
    for model_name in ("vit_base_patch16_224", "vit_base_patch16_clip_224.openai"):
        label = _format_model_label(model_name)
        base_rows = [row for row in baseline_tradeoff if row["model_name"] == model_name]
        dam_rows = [row for row in dam_tradeoff if row["model_name"] == model_name]
        lines.append(
            f"- {label} baseline: supported={_count(base_rows, model_name=model_name, classification='tradeoff_supported')}, "
            f"partial={_count(base_rows, model_name=model_name, classification='tradeoff_partial')}, "
            f"not_supported={_count(base_rows, model_name=model_name, classification='tradeoff_not_supported')}."
        )
        lines.append(
            f"- {label} DAM: supported={_count(dam_rows, model_name=model_name, classification='tradeoff_supported')}, "
            f"partial={_count(dam_rows, model_name=model_name, classification='tradeoff_partial')}, "
            f"not_supported={_count(dam_rows, model_name=model_name, classification='tradeoff_not_supported')}."
        )
    lines.append("")
    all_rows = baseline_tradeoff + dam_tradeoff
    strongest = [
        row for row in all_rows if row["classification"] == "tradeoff_supported"
    ]
    if strongest:
        strongest.sort(key=lambda row: (row["kind"] == "dam", row["best_gen_tercile"] == "middle"), reverse=True)
        best = strongest[0]
        lines.append(
            f"- Clearest tradeoff-looking condition: `{best['setting_name']}` `{best['model_name']}` "
            f"{'DAM n=' + str(best['dam_n']) if best['kind'] == 'dam' else 'baseline'}."
        )
    else:
        lines.append("- No condition met the full tradeoff-supported rule; inspect partial cases first.")
    return "\n".join(lines) + "\n"


def _select_setting_specs(
    *,
    max_settings: int | None,
    setting_names: list[str] | tuple[str, ...] | None,
) -> list[SettingSpec]:
    if setting_names is not None:
        name_set = set(setting_names)
        selected = [spec for spec in SETTING_SPECS if spec.name in name_set]
    else:
        selected = list(SETTING_SPECS)
    if max_settings is not None:
        selected = selected[:max_settings]
    return selected


def run_suite(
    *,
    device: str,
    batch_size: int,
    seed: int,
    output_dir: Path,
    dataset_pkl: str,
    image_root: str,
    backend: str,
    n_seeds: int,
    max_settings: int | None = None,
    max_seeds: int | None = None,
    max_dam_orders: int | None = None,
    max_layers: int | None = None,
    setting_names: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_naturalistic_category("animals", dataset_pkl=dataset_pkl, image_root=image_root)
    total_images = len(data.filenames)
    images = _load_images(data.image_paths)
    selected_settings = _select_setting_specs(max_settings=max_settings, setting_names=setting_names)
    selected_dam_orders = list(DAM_ORDERS[: max_dam_orders or len(DAM_ORDERS)])
    actual_n_seeds = min(n_seeds, max_seeds) if max_seeds is not None else n_seeds

    wrappers: dict[str, VisionEmbeddingWrapper] = {}
    clean_features_by_model: dict[str, dict[str, np.ndarray]] = {}
    clean_reextract_by_model: dict[str, dict[str, np.ndarray]] = {}
    preprocessors_by_model: dict[str, LayerwisePreprocessor] = {}
    dims_by_model: dict[str, dict[str, int]] = {}
    layer_names: list[str] | None = None
    for model_name, pooling in MODEL_SPECS:
        wrapper = VisionEmbeddingWrapper(model_name=model_name, pretrained=True, device=device, pooling=pooling)
        wrappers[model_name] = wrapper
        layer_indices = _all_layer_indices(wrapper, max_layers=max_layers)
        raw = wrapper.extract(images, layer_indices=layer_indices, batch_size=batch_size)
        prep = LayerwisePreprocessor(use_zscore=False, l2_normalize=True)
        clean = prep.fit_transform(raw)
        clean_re = prep.transform(wrapper.extract(images, layer_indices=layer_indices, batch_size=batch_size))
        preprocessors_by_model[model_name] = prep
        clean_features_by_model[model_name] = clean
        clean_reextract_by_model[model_name] = clean_re
        dims_by_model[model_name] = {layer: int(feats.shape[1]) for layer, feats in clean.items()}
        if layer_names is None:
            layer_names = sorted(clean.keys(), key=_layer_index)
    assert layer_names is not None

    written_dirs: list[Path] = []
    combined_baseline_agg: list[dict[str, object]] = []
    combined_dam_agg: list[dict[str, object]] = []
    combined_tradeoff: list[dict[str, object]] = []

    for setting in selected_settings:
        if setting.n_stored >= total_images:
            raise ValueError(
                f"Setting {setting.name} has n_stored={setting.n_stored}, but animals has only {total_images} images."
            )
        setting_dir = output_dir / setting.name
        setting_dir.mkdir(parents=True, exist_ok=True)
        n_probe = total_images - setting.n_stored
        splits = _random_storage_splits(
            total_images,
            storage_sizes=[setting.n_stored],
            n_seeds=actual_n_seeds,
            seed=seed,
        )

        corrupted_features_by_model: dict[str, dict[str, np.ndarray]] = {}
        corrupted_images = _corrupt_images(images, setting, seed)
        for model_name, _pooling in MODEL_SPECS:
            raw_corrupt = wrappers[model_name].extract(
                corrupted_images,
                layer_indices=[_layer_index(name) for name in layer_names],
                batch_size=batch_size,
            )
            corrupted_features_by_model[model_name] = preprocessors_by_model[model_name].transform(raw_corrupt)

        baseline_rows: list[dict[str, object]] = []
        dam_rows: list[dict[str, object]] = []
        for split_label, split_seed, stored_indices, probe_indices in splits:
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
            for model_name, pooling in MODEL_SPECS:
                stored_clean = subset_feature_bundle(clean_features_by_model[model_name], ident_task.stored_indices)
                corrupted_probe_bundle = subset_feature_bundle(corrupted_features_by_model[model_name], ident_task.probe_indices)
                gen_probe_bundle = subset_feature_bundle(clean_features_by_model[model_name], gen_task.probe_indices)

                ident_results = evaluate_layerwise_baseline(
                    stored_clean,
                    corrupted_probe_bundle,
                    ident_task.ground_truth_idx,
                    metric="cosine",
                    decision_noise_std=setting.decision_noise_std,
                    decision_noise_seed=seed + split_seed,
                )
                gen_results = evaluate_layerwise_baseline(
                    stored_clean,
                    gen_probe_bundle,
                    gen_task.ground_truth_idx,
                    metric="cosine",
                    decision_noise_std=setting.decision_noise_std,
                    decision_noise_seed=seed + 50000 + split_seed,
                )

                for layer in layer_names:
                    pair_sim_mean, pair_dist_mean, pair_sim_std, pair_dist_std = _mean_pairwise_cosine_stats(
                        stored_clean[layer]
                    )
                    baseline_rows.append(
                        {
                            "setting_name": setting.name,
                            "split_label": split_label,
                            "split_seed": split_seed,
                            "model_name": model_name,
                            "pooling": pooling,
                            "layer": layer,
                            "n_stored": setting.n_stored,
                            "n_probe": n_probe,
                            "decision_noise_std": setting.decision_noise_std,
                            "corruption_mode": setting.corruption_mode,
                            "noise_std": setting.noise_std,
                            "max_shift": setting.max_shift,
                            "occlusion_frac": setting.occlusion_frac,
                            "mean_pairwise_cosine_similarity": pair_sim_mean,
                            "mean_pairwise_cosine_distance": pair_dist_mean,
                            "pairwise_cosine_similarity_std": pair_sim_std,
                            "pairwise_cosine_distance_std": pair_dist_std,
                            "ident_accuracy": float(ident_results[layer]["accuracy"]),
                            "gen_accuracy": float(gen_results[layer]["accuracy"]),
                        }
                    )

                for dam_n in selected_dam_orders:
                    config = {
                        "n": dam_n,
                        "beta": DAM_FIXED_PARAMS["beta"],
                        "alpha": DAM_FIXED_PARAMS["alpha"],
                        "lmbda": DAM_FIXED_PARAMS["lmbda"],
                        "steps_multiplier": DAM_FIXED_PARAMS["steps_multiplier"],
                    }
                    for layer in layer_names:
                        ident_dam = _evaluate_dam_on_bundle(
                            np.asarray(stored_clean[layer], dtype=np.float32),
                            np.asarray(corrupted_probe_bundle[layer], dtype=np.float32),
                            np.asarray(ident_task.ground_truth_idx, dtype=np.int64),
                            config=config,
                            seed=seed + split_seed * 1000 + dam_n,
                            backend=backend,
                            decision_noise_std=setting.decision_noise_std,
                            decision_noise_seed=seed + 100000 + split_seed,
                        )
                        gen_dam = _evaluate_dam_on_bundle(
                            np.asarray(stored_clean[layer], dtype=np.float32),
                            np.asarray(gen_probe_bundle[layer], dtype=np.float32),
                            np.asarray(gen_task.ground_truth_idx, dtype=np.int64),
                            config=config,
                            seed=seed + split_seed * 1000 + dam_n + 5000,
                            backend=backend,
                            decision_noise_std=setting.decision_noise_std,
                            decision_noise_seed=seed + 150000 + split_seed,
                        )
                        pair_sim_mean, pair_dist_mean, pair_sim_std, pair_dist_std = _mean_pairwise_cosine_stats(
                            stored_clean[layer]
                        )
                        dam_rows.append(
                            {
                                "setting_name": setting.name,
                                "split_label": split_label,
                                "split_seed": split_seed,
                                "model_name": model_name,
                                "pooling": pooling,
                                "layer": layer,
                                "dam_n": dam_n,
                                "dam_beta": float(DAM_FIXED_PARAMS["beta"]),
                                "dam_alpha": float(DAM_FIXED_PARAMS["alpha"]),
                                "dam_lmbda": float(DAM_FIXED_PARAMS["lmbda"]),
                                "dam_steps_multiplier": int(DAM_FIXED_PARAMS["steps_multiplier"]),
                                "n_stored": setting.n_stored,
                                "n_probe": n_probe,
                                "decision_noise_std": setting.decision_noise_std,
                                "corruption_mode": setting.corruption_mode,
                                "noise_std": setting.noise_std,
                                "max_shift": setting.max_shift,
                                "occlusion_frac": setting.occlusion_frac,
                                "neurons": dims_by_model[model_name][layer],
                                "mean_pairwise_cosine_similarity": pair_sim_mean,
                                "mean_pairwise_cosine_distance": pair_dist_mean,
                                "pairwise_cosine_similarity_std": pair_sim_std,
                                "pairwise_cosine_distance_std": pair_dist_std,
                                "ident_accuracy": float(ident_dam["accuracy"]),
                                "gen_accuracy": float(gen_dam["accuracy"]),
                            }
                        )

        baseline_agg = _aggregate_rows(
            baseline_rows,
            group_keys=(
                "setting_name", "model_name", "pooling", "layer", "n_stored", "n_probe", "decision_noise_std",
                "corruption_mode", "noise_std", "max_shift", "occlusion_frac"
            ),
            metric_keys=(
                "ident_accuracy", "gen_accuracy", "mean_pairwise_cosine_similarity", "mean_pairwise_cosine_distance",
                "pairwise_cosine_similarity_std", "pairwise_cosine_distance_std"
            ),
        )
        dam_agg = _aggregate_rows(
            dam_rows,
            group_keys=(
                "setting_name", "model_name", "pooling", "layer", "dam_n", "dam_beta", "dam_alpha",
                "dam_lmbda", "dam_steps_multiplier", "n_stored", "n_probe", "decision_noise_std",
                "corruption_mode", "noise_std", "max_shift", "occlusion_frac", "neurons"
            ),
            metric_keys=(
                "ident_accuracy", "gen_accuracy", "mean_pairwise_cosine_similarity", "mean_pairwise_cosine_distance",
                "pairwise_cosine_similarity_std", "pairwise_cosine_distance_std"
            ),
        )
        baseline_tradeoff = _build_tradeoff_summary(baseline_agg, kind="baseline")
        dam_tradeoff = _build_tradeoff_summary(dam_agg, kind="dam")

        _write_csv(setting_dir / "baseline_raw.csv", baseline_rows)
        _write_csv(setting_dir / "dam_raw.csv", dam_rows)
        _write_csv(setting_dir / "baseline_aggregated.csv", baseline_agg)
        _write_csv(setting_dir / "dam_aggregated.csv", dam_agg)
        _write_csv(setting_dir / "tradeoff_summary.csv", baseline_tradeoff + dam_tradeoff)

        subtitle = _setting_subtitle(setting, n_probe=n_probe, n_seeds=actual_n_seeds)
        _plot_accuracy_vs_distance(
            rows=baseline_agg,
            title=f"Animals baseline accuracy vs pairwise distance: {setting.name}",
            subtitle_lines=subtitle,
            output_path=setting_dir / "baseline_accuracy_vs_pairwise_distance.png",
        )
        _plot_scatter_colored_by_distance(
            rows=baseline_agg,
            title=f"Animals baseline gen vs ident ordered by distance: {setting.name}",
            subtitle_lines=subtitle,
            output_path=setting_dir / "baseline_layers_gen_vs_ident_colored_by_distance.png",
        )
        for dam_n in selected_dam_orders:
            dam_subset = [row for row in dam_agg if int(row["dam_n"]) == dam_n]
            _plot_accuracy_vs_distance(
                rows=dam_subset,
                title=f"Animals DAM accuracy vs pairwise distance: {setting.name} | n={dam_n}",
                subtitle_lines=subtitle,
                output_path=setting_dir / f"dam_n{dam_n}_accuracy_vs_pairwise_distance.png",
            )
            _plot_scatter_colored_by_distance(
                rows=dam_subset,
                title=f"Animals DAM gen vs ident ordered by distance: {setting.name} | n={dam_n}",
                subtitle_lines=subtitle,
                output_path=setting_dir / f"dam_n{dam_n}_layers_gen_vs_ident_colored_by_distance.png",
            )

        _write_config(
            setting_dir=setting_dir,
            setting=setting,
            n_probe=n_probe,
            n_seeds=actual_n_seeds,
            dam_orders=selected_dam_orders,
            layer_names=layer_names,
            dims_by_model=dims_by_model,
        )
        (setting_dir / "insights.md").write_text(
            _build_setting_insights(
                setting,
                baseline_agg,
                dam_agg,
                baseline_tradeoff,
                dam_tradeoff,
                selected_dam_orders,
            ),
            encoding="utf-8",
        )

        combined_baseline_agg.extend(baseline_agg)
        combined_dam_agg.extend(dam_agg)
        combined_tradeoff.extend(baseline_tradeoff)
        combined_tradeoff.extend(dam_tradeoff)
        written_dirs.append(setting_dir)

    _write_csv(output_dir / "combined_baseline_aggregated.csv", combined_baseline_agg)
    _write_csv(output_dir / "combined_dam_aggregated.csv", combined_dam_agg)
    _write_csv(output_dir / "tradeoff_summary.csv", combined_tradeoff)
    (output_dir / "overall_insights.md").write_text(
        _build_overall_insights(
            [row for row in combined_tradeoff if row["kind"] == "baseline"],
            [row for row in combined_tradeoff if row["kind"] == "dam"],
        ),
        encoding="utf-8",
    )
    _write_root_readme(output_dir, selected_settings)
    return written_dirs


def main() -> None:
    args = parse_args()
    written = run_suite(
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        dataset_pkl=args.dataset_pkl,
        image_root=args.image_root,
        backend=args.backend,
        n_seeds=args.n_seeds,
        max_settings=args.max_settings,
        max_seeds=args.max_seeds,
        max_dam_orders=args.max_dam_orders,
        max_layers=args.max_layers,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
