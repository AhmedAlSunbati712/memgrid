from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.run_animals_image_transformer_similarity_ordered import DAM_ORDERS, SETTING_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process similarity-ordered animals plots so distance rank is the explicit ordering variable."
    )
    parser.add_argument("--input-dir", default="result/image_transformers_similarity_ordered")
    parser.add_argument("--output-dir", default="result/image_transformers_similarity_ordered_fixed")
    parser.add_argument("--max-settings", type=int, default=None)
    parser.add_argument("--max-dam-orders", type=int, default=None)
    return parser.parse_args()


def _layer_index(layer_name: str) -> int:
    return int(str(layer_name).split("_")[1])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _distance_ranked(rows: list[dict[str, str]], *, with_dam_n: bool) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["setting_name"], row["model_name"])
        if with_dam_n:
            key = key + (row["dam_n"],)
        grouped.setdefault(key, []).append(row)

    ranked_rows: list[dict[str, object]] = []
    for group_rows in grouped.values():
        ordered = sorted(
            group_rows,
            key=lambda row: (
                float(row["mean_pairwise_cosine_distance_mean"]),
                _layer_index(row["layer"]),
            ),
        )
        for rank, row in enumerate(ordered):
            ranked = dict(row)
            ranked["distance_rank"] = rank
            ranked_rows.append(ranked)
    return ranked_rows


def _plot_scatter_ordered(
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
        subset.sort(key=lambda row: int(row["distance_rank"]))
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(0.35 + 0.55 * (idx / max(1, len(subset) - 1))) for idx in range(len(subset))]
        x_vals = [float(row["gen_accuracy_mean"]) for row in subset]
        y_vals = [float(row["ident_accuracy_mean"]) for row in subset]
        ax.plot(x_vals, y_vals, color=colors[-1], linewidth=1.5, alpha=0.85, label=f"{label} ordered by distance rank")
        marker = "o" if "clip" not in model_name else "s"
        for idx, (row, x_val, y_val) in enumerate(zip(subset, x_vals, y_vals)):
            ax.scatter([x_val], [y_val], color=[colors[idx]], marker=marker, s=42)
            ax.annotate(
                f"R{int(row['distance_rank'])}",
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


def _build_distance_order_rows(
    baseline_rows: list[dict[str, object]],
    dam_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in baseline_rows:
        out.append(
            {
                "kind": "baseline",
                "setting_name": row["setting_name"],
                "model_name": row["model_name"],
                "dam_n": "",
                "distance_rank": int(row["distance_rank"]),
                "layer": row["layer"],
                "mean_pairwise_cosine_distance_mean": float(row["mean_pairwise_cosine_distance_mean"]),
                "gen_accuracy_mean": float(row["gen_accuracy_mean"]),
                "ident_accuracy_mean": float(row["ident_accuracy_mean"]),
            }
        )
    for row in dam_rows:
        out.append(
            {
                "kind": "dam",
                "setting_name": row["setting_name"],
                "model_name": row["model_name"],
                "dam_n": int(row["dam_n"]),
                "distance_rank": int(row["distance_rank"]),
                "layer": row["layer"],
                "mean_pairwise_cosine_distance_mean": float(row["mean_pairwise_cosine_distance_mean"]),
                "gen_accuracy_mean": float(row["gen_accuracy_mean"]),
                "ident_accuracy_mean": float(row["ident_accuracy_mean"]),
            }
        )
    return out


def _format_model_label(model_name: str) -> str:
    return "ViT" if "clip" not in model_name else "CLIP"


def _build_setting_insights(
    baseline_rows: list[dict[str, object]],
    dam_rows: list[dict[str, object]],
    dam_orders: list[int],
) -> str:
    lines = ["# Insights", "", "## Distance Rank Mapping"]
    for model_name in ("vit_base_patch16_224", "vit_base_patch16_clip_224.openai"):
        subset = [row for row in baseline_rows if row["model_name"] == model_name]
        subset.sort(key=lambda row: int(row["distance_rank"]))
        mapping = ", ".join(
            f"R{int(row['distance_rank'])}={row['layer']}" for row in subset
        )
        lines.append(f"- {_format_model_label(model_name)} baseline: {mapping}")
    lines.extend(["", "## Baseline"])
    for model_name in ("vit_base_patch16_224", "vit_base_patch16_clip_224.openai"):
        subset = [row for row in baseline_rows if row["model_name"] == model_name]
        best_gen = max(subset, key=lambda row: float(row["gen_accuracy_mean"]))
        best_ident = max(subset, key=lambda row: float(row["ident_accuracy_mean"]))
        lines.append(
            f"- {_format_model_label(model_name)}: best gen at `R{int(best_gen['distance_rank'])}` "
            f"({best_gen['layer']}, {float(best_gen['gen_accuracy_mean']):.2f}, {float(best_gen['ident_accuracy_mean']):.2f}); "
            f"best ident at `R{int(best_ident['distance_rank'])}` "
            f"({best_ident['layer']}, {float(best_ident['gen_accuracy_mean']):.2f}, {float(best_ident['ident_accuracy_mean']):.2f})."
        )
    lines.extend(["", "## DAM by energy order"])
    for dam_n in dam_orders:
        parts: list[str] = []
        for model_name in ("vit_base_patch16_224", "vit_base_patch16_clip_224.openai"):
            subset = [row for row in dam_rows if row["model_name"] == model_name and int(row["dam_n"]) == dam_n]
            best = max(subset, key=lambda row: float(row["gen_accuracy_mean"]))
            parts.append(
                f"{_format_model_label(model_name)} best at `R{int(best['distance_rank'])}` "
                f"({best['layer']}, gen={float(best['gen_accuracy_mean']):.2f}, ident={float(best['ident_accuracy_mean']):.2f})"
            )
        lines.append(f"- `n={dam_n}`: " + "; ".join(parts) + ".")
    lines.extend(["", "## Takeaway"])
    lines.append("- These corrected plots are ordered by pairwise-distance rank, not by layer sequence.")
    lines.append("- Use `distance_order.csv` to map each rank label back to its layer and mean pairwise cosine distance.")
    return "\n".join(lines) + "\n"


def _copy_root_csvs(input_dir: Path, output_dir: Path) -> None:
    for name in ("combined_baseline_aggregated.csv", "combined_dam_aggregated.csv", "tradeoff_summary.csv"):
        shutil.copy2(input_dir / name, output_dir / name)


def _write_root_readme(output_dir: Path, settings: list[str]) -> None:
    lines = [
        "# Image Transformer Similarity-Ordered Tradeoff Suite (Fixed)",
        "",
        "Post-processed animals-only plots where the sequence is explicitly ordered by pairwise-distance rank.",
        "",
        "Each setting directory contains:",
        "- copied aggregated/raw CSVs from the original run",
        "- corrected gen-vs-ident plots ordered by pairwise-distance rank",
        "- `distance_order.csv` mapping rank labels to layers",
        "- updated `insights.md`",
        "",
        "Settings:",
    ]
    for setting in settings:
        lines.append(f"- `{setting}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overall_insights(output_dir: Path, settings: list[str]) -> None:
    lines = [
        "# Overall Insights",
        "",
        "- This fixed output root corrects the visualization only; it does not change the metrics or tradeoff classification.",
        "- The sequence of points in each corrected scatter plot is determined by ascending mean pairwise cosine distance.",
        "- Layer identity is now secondary and is mapped through `distance_order.csv` instead of being the organizing visual cue.",
        "",
        "Settings included:",
    ]
    for setting in settings:
        lines.append(f"- `{setting}`")
    (output_dir / "overall_insights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_setting_names(
    *,
    max_settings: int | None,
    setting_names: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if setting_names is not None:
        name_set = set(setting_names)
        selected = [spec.name for spec in SETTING_SPECS if spec.name in name_set]
    else:
        selected = [setting.name for setting in SETTING_SPECS]
    if max_settings is not None:
        selected = selected[:max_settings]
    return selected


def run_suite(
    *,
    input_dir: Path,
    output_dir: Path,
    max_settings: int | None = None,
    max_dam_orders: int | None = None,
    setting_names: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_settings = _select_setting_names(max_settings=max_settings, setting_names=setting_names)
    selected_dam_orders = list(DAM_ORDERS[: max_dam_orders or len(DAM_ORDERS)])
    written_dirs: list[Path] = []

    for setting_name in selected_settings:
        src_dir = input_dir / setting_name
        dst_dir = output_dir / setting_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        baseline_rows = _distance_ranked(_read_csv(src_dir / "baseline_aggregated.csv"), with_dam_n=False)
        dam_rows = _distance_ranked(_read_csv(src_dir / "dam_aggregated.csv"), with_dam_n=True)
        distance_order_rows = _build_distance_order_rows(baseline_rows, dam_rows)

        for name in ("baseline_raw.csv", "baseline_aggregated.csv", "dam_raw.csv", "dam_aggregated.csv", "tradeoff_summary.csv", "config.txt"):
            shutil.copy2(src_dir / name, dst_dir / name)
        _write_csv(dst_dir / "distance_order.csv", distance_order_rows)

        subtitle_lines = []
        with (src_dir / "config.txt").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("n_stored_baseline:"):
                    n_stored = line.split(":", 1)[1].strip()
                if line.startswith("n_probe:"):
                    n_probe = line.split(":", 1)[1].strip()
                if line.startswith("n_seeds:"):
                    n_seeds = line.split(":", 1)[1].strip()
                if line.startswith("corruption_mode:"):
                    corruption_mode = line.split(":", 1)[1].strip()
                if line.startswith("noise_std:"):
                    noise_std = line.split(":", 1)[1].strip()
                if line.startswith("max_shift:"):
                    max_shift = line.split(":", 1)[1].strip()
                if line.startswith("occlusion_frac:"):
                    occlusion_frac = line.split(":", 1)[1].strip()
                if line.startswith("decision_noise_std:"):
                    decision_noise = line.split(":", 1)[1].strip()
        subtitle_lines = [
            f"dataset=animals | n_stored={n_stored} | n_probe={n_probe} | n_seeds={n_seeds}",
            f"corruption={corruption_mode}, noise_std={noise_std}, max_shift={max_shift}, occlusion_frac={occlusion_frac}, decision_noise={decision_noise}",
            "point sequence is ordered by ascending mean pairwise cosine distance rank",
        ]

        _plot_scatter_ordered(
            rows=baseline_rows,
            title=f"Animals baseline gen vs ident ordered by distance rank: {setting_name}",
            subtitle_lines=subtitle_lines,
            output_path=dst_dir / "baseline_layers_gen_vs_ident_ordered_by_distance.png",
        )
        for dam_n in selected_dam_orders:
            subset = [row for row in dam_rows if int(row["dam_n"]) == dam_n]
            _plot_scatter_ordered(
                rows=subset,
                title=f"Animals DAM gen vs ident ordered by distance rank: {setting_name} | n={dam_n}",
                subtitle_lines=subtitle_lines,
                output_path=dst_dir / f"dam_n{dam_n}_layers_gen_vs_ident_ordered_by_distance.png",
            )

        (dst_dir / "insights.md").write_text(
            _build_setting_insights(baseline_rows, dam_rows, selected_dam_orders),
            encoding="utf-8",
        )
        written_dirs.append(dst_dir)

    _copy_root_csvs(input_dir, output_dir)
    _write_root_readme(output_dir, selected_settings)
    _write_overall_insights(output_dir, selected_settings)
    return written_dirs


def main() -> None:
    args = parse_args()
    written = run_suite(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        max_settings=args.max_settings,
        max_dam_orders=args.max_dam_orders,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
