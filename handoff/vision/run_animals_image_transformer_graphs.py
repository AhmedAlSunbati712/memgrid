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
    SettingSpec(
        name="hard_s40",
        n_stored=40,
        corruption_mode="noise_shift_occlusion",
        noise_std=5.0,
        max_shift=2,
        occlusion_frac=0.5,
        decision_noise_std=0.01,
    ),
    SettingSpec(
        name="hard_s80",
        n_stored=80,
        corruption_mode="noise_shift_occlusion",
        noise_std=5.0,
        max_shift=2,
        occlusion_frac=0.5,
        decision_noise_std=0.01,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate animals-only layerwise gen-vs-ident graphs for ViT and CLIP.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="result/image_transformers")
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


def _aggregate_rows(rows: list[dict[str, object]], group_keys: tuple[str, ...], metric_keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    out: list[dict[str, object]] = []
    for _, group_rows in grouped.items():
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


def _plot_layer_graph(
    *,
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    title: str,
    subtitle_lines: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    styles = {
        "vit_base_patch16_224": {"color": "#1f77b4", "marker": "o", "label": "ViT"},
        "vit_base_patch16_clip_224.openai": {"color": "#d62728", "marker": "s", "label": "CLIP"},
    }
    for model_name, style in styles.items():
        subset = [row for row in rows if row["model_name"] == model_name]
        subset.sort(key=lambda row: _layer_index(str(row["layer"])))
        xs = [float(row[x_key]) for row in subset]
        ys = [float(row[y_key]) for row in subset]
        ax.plot(xs, ys, color=style["color"], marker=style["marker"], linewidth=1.5, label=style["label"])
        for row, x_val, y_val in zip(subset, xs, ys):
            ax.annotate(
                f"L{_layer_index(str(row['layer']))}",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=style["color"],
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


def _shift_label(dx: float, dy: float) -> str:
    x = "right" if dx > 0.5 else "left" if dx < -0.5 else "flat"
    y = "up" if dy > 0.5 else "down" if dy < -0.5 else "flat"
    return f"{x}/{y}"


def _tradeoff_label(rows: list[dict[str, object]], x_key: str, y_key: str) -> str:
    if len(rows) < 3:
        return "insufficient layers"
    x_vals = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
    y_vals = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
    corr = float(np.corrcoef(x_vals, y_vals)[0, 1]) if np.std(x_vals) > 1e-12 and np.std(y_vals) > 1e-12 else 0.0
    if corr < -0.3:
        return f"weak negative layer trend (corr={corr:.2f})"
    if corr > 0.3:
        return f"positive layer trend (corr={corr:.2f})"
    return f"no clear layer tradeoff (corr={corr:.2f})"


def _build_insights(
    setting: SettingSpec,
    baseline_agg: list[dict[str, object]],
    dam_agg: list[dict[str, object]],
    dam_orders: list[int],
) -> str:
    lines: list[str] = []
    lines.append("# Insights")
    lines.append("")
    lines.append("## Setting")
    lines.append(
        f"Animals only, `{setting.name}`, `n_stored={setting.n_stored}`, "
        f"`corruption={setting.corruption_mode}`, `decision_noise_std={setting.decision_noise_std:g}`."
    )
    lines.append("")
    lines.append("## Baseline")
    for model_name in ("vit_base_patch16_224", "vit_base_patch16_clip_224.openai"):
        rows = [row for row in baseline_agg if row["model_name"] == model_name]
        best_gen = max(rows, key=lambda row: float(row["gen_accuracy_mean"]))
        best_ident = max(rows, key=lambda row: float(row["ident_accuracy_mean"]))
        tradeoff = _tradeoff_label(rows, "gen_accuracy_mean", "ident_accuracy_mean")
        label = "ViT" if "clip" not in model_name else "CLIP"
        lines.append(
            f"- {label}: best gen at `{best_gen['layer']}` "
            f"({float(best_gen['gen_accuracy_mean']):.2f}, {float(best_gen['ident_accuracy_mean']):.2f}); "
            f"best ident at `{best_ident['layer']}` "
            f"({float(best_ident['gen_accuracy_mean']):.2f}, {float(best_ident['ident_accuracy_mean']):.2f}); "
            f"{tradeoff}."
        )
    lines.append("")
    lines.append("## DAM by energy order")
    for dam_n in dam_orders:
        subset = [row for row in dam_agg if int(row["dam_n"]) == dam_n]
        parts: list[str] = []
        for model_name in ("vit_base_patch16_224", "vit_base_patch16_clip_224.openai"):
            label = "ViT" if "clip" not in model_name else "CLIP"
            dam_rows = [row for row in subset if row["model_name"] == model_name]
            base_rows = [row for row in baseline_agg if row["model_name"] == model_name]
            best_dam = max(dam_rows, key=lambda row: float(row["gen_accuracy_mean"]))
            best_base = max(base_rows, key=lambda row: float(row["gen_accuracy_mean"]))
            dx = float(best_dam["gen_accuracy_mean"]) - float(best_base["gen_accuracy_mean"])
            dy = float(best_dam["ident_accuracy_mean"]) - float(best_base["ident_accuracy_mean"])
            parts.append(
                f"{label}: best `{best_dam['layer']}` shift `{_shift_label(dx, dy)}` "
                f"(Δgen={dx:+.2f}, Δident={dy:+.2f})"
            )
        lines.append(f"- `n={dam_n}`: " + "; ".join(parts) + ".")
    lines.append("")
    lines.append("## Takeaway")
    overall_base = max(baseline_agg, key=lambda row: float(row["gen_accuracy_mean"]))
    overall_dam = max(dam_agg, key=lambda row: float(row["gen_accuracy_mean"]))
    lines.append(
        f"- Best baseline overall: `{overall_base['model_name']}` `{overall_base['layer']}` "
        f"at ({float(overall_base['gen_accuracy_mean']):.2f}, {float(overall_base['ident_accuracy_mean']):.2f})."
    )
    lines.append(
        f"- Best DAM overall: `n={int(overall_dam['dam_n'])}` `{overall_dam['model_name']}` `{overall_dam['layer']}` "
        f"at ({float(overall_dam['gen_accuracy_mean']):.2f}, {float(overall_dam['ident_accuracy_mean']):.2f})."
    )
    best_by_order = {
        int(dam_n): max(
            [row for row in dam_agg if int(row["dam_n"]) == dam_n],
            key=lambda row: float(row["gen_accuracy_mean"]),
        )
        for dam_n in dam_orders
    }
    best_order = max(best_by_order, key=lambda n: float(best_by_order[n]["gen_accuracy_mean"]))
    lines.append(f"- Best energy order in this folder: `n={best_order}`.")
    if all(float(best_by_order[n]["gen_accuracy_mean"]) <= float(best_by_order[best_order]["gen_accuracy_mean"]) + 1e-9 for n in dam_orders):
        lines.append("- Higher energy orders are not automatically better; compare each `n` directly to the baseline graph.")
    lines.append("- Use the layer labels on the plots to see whether the same layers dominate baseline and DAM.")
    return "\n".join(lines) + "\n"


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
            "  - baseline_layers_gen_vs_ident.png",
            "  - insights.md",
        ]
    )
    for dam_n in dam_orders:
        lines.insert(-1, f"  - dam_n{dam_n}_layers_gen_vs_ident.png")
    (setting_dir / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_readme(root_dir: Path, settings: list[SettingSpec]) -> None:
    lines = [
        "# Image Transformer Graph Suite",
        "",
        "Animals-only naturalistic layerwise graphs for ViT and CLIP.",
        "",
        "Each subdirectory contains:",
        "- baseline layerwise `(gen_accuracy, ident_accuracy)` graph",
        "- one DAM graph per energy order `n`",
        "- raw and aggregated CSVs",
        "- `config.txt` with exact evaluation parameters",
        "- `insights.md` with concise findings",
        "",
        "Settings:",
    ]
    for setting in settings:
        lines.append(f"- `{setting.name}`")
    (root_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_naturalistic_category("animals", dataset_pkl=dataset_pkl, image_root=image_root)
    images = _load_images(data.image_paths)
    selected_settings = list(SETTING_SPECS[: max_settings or len(SETTING_SPECS)])
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
    for setting in selected_settings:
        setting_dir = output_dir / setting.name
        setting_dir.mkdir(parents=True, exist_ok=True)
        n_probe = len(data.filenames) - setting.n_stored
        splits = _random_storage_splits(
            len(data.filenames),
            storage_sizes=[setting.n_stored],
            n_seeds=actual_n_seeds,
            seed=seed,
        )

        corrupted_features_by_model: dict[str, dict[str, np.ndarray]] = {}
        for model_name, _pooling in MODEL_SPECS:
            corrupted_images = _corrupt_images(images, setting, seed)
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
                clean_re_bundle = subset_feature_bundle(clean_reextract_by_model[model_name], ident_task.probe_indices)
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
                                "ident_accuracy": float(ident_dam["accuracy"]),
                                "gen_accuracy": float(gen_dam["accuracy"]),
                            }
                        )

        baseline_agg = _aggregate_rows(
            baseline_rows,
            group_keys=("setting_name", "model_name", "pooling", "layer", "n_stored", "n_probe", "decision_noise_std", "corruption_mode", "noise_std", "max_shift", "occlusion_frac"),
            metric_keys=("ident_accuracy", "gen_accuracy"),
        )
        dam_agg = _aggregate_rows(
            dam_rows,
            group_keys=(
                "setting_name", "model_name", "pooling", "layer", "dam_n", "dam_beta", "dam_alpha",
                "dam_lmbda", "dam_steps_multiplier", "n_stored", "n_probe", "decision_noise_std",
                "corruption_mode", "noise_std", "max_shift", "occlusion_frac", "neurons"
            ),
            metric_keys=("ident_accuracy", "gen_accuracy"),
        )

        _write_csv(setting_dir / "baseline_raw.csv", baseline_rows)
        _write_csv(setting_dir / "dam_raw.csv", dam_rows)
        _write_csv(setting_dir / "baseline_aggregated.csv", baseline_agg)
        _write_csv(setting_dir / "dam_aggregated.csv", dam_agg)

        subtitle = _setting_subtitle(setting, n_probe=n_probe, n_seeds=actual_n_seeds)
        _plot_layer_graph(
            rows=baseline_agg,
            x_key="gen_accuracy_mean",
            y_key="ident_accuracy_mean",
            title=f"Animals baseline layers: {setting.name}",
            subtitle_lines=subtitle,
            output_path=setting_dir / "baseline_layers_gen_vs_ident.png",
        )
        for dam_n in selected_dam_orders:
            _plot_layer_graph(
                rows=[row for row in dam_agg if int(row["dam_n"]) == dam_n],
                x_key="gen_accuracy_mean",
                y_key="ident_accuracy_mean",
                title=f"Animals DAM layers: {setting.name} | n={dam_n}",
                subtitle_lines=subtitle,
                output_path=setting_dir / f"dam_n{dam_n}_layers_gen_vs_ident.png",
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
            _build_insights(setting, baseline_agg, dam_agg, selected_dam_orders),
            encoding="utf-8",
        )
        written_dirs.append(setting_dir)

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
