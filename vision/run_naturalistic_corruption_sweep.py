from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.run_naturalistic_baseline import _corruption_label, get_model_specs, run_naturalistic_baseline


CORRUPTION_CONFIGS: tuple[dict[str, object], ...] = (
    {"corruption_mode": "noise_shift", "noise_std": 5.0, "max_shift": 2},
    {"corruption_mode": "noise_shift", "noise_std": 15.0, "max_shift": 8},
    {"corruption_mode": "noise_shift", "noise_std": 30.0, "max_shift": 16},
    {"corruption_mode": "occlusion", "occlusion_frac": 0.15},
    {"corruption_mode": "occlusion", "occlusion_frac": 0.30},
    {"corruption_mode": "occlusion", "occlusion_frac": 0.50},
    {"corruption_mode": "multi_cutout", "mask_frac": 0.15, "n_chunks": 4},
    {"corruption_mode": "multi_cutout", "mask_frac": 0.30, "n_chunks": 6},
    {"corruption_mode": "multi_cutout", "mask_frac": 0.50, "n_chunks": 8},
    {"corruption_mode": "warp", "affine_strength": 0.02},
    {"corruption_mode": "warp", "affine_strength": 0.05},
    {"corruption_mode": "warp", "affine_strength": 0.10},
)

DECISION_NOISE_LEVELS: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused naturalistic corruption sweep.")
    parser.add_argument("--categories", nargs="*", default=["animals", "fruits"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/vision/naturalistic_corruption")
    parser.add_argument("--dataset-pkl", default="datasets_peterson.pkl")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--include-clip", action="store_true", default=True)
    parser.add_argument("--decision-noise-stage", action="store_true")
    parser.add_argument("--max-configs", type=int, default=None)
    return parser.parse_args()


def _default_category_kwargs(category: str) -> dict[str, object]:
    if category == "animals":
        return {"split_mode": "random_storage_curve", "storage_sizes": [40, 80], "n_seeds": 5}
    if category == "vegetables":
        return {"required_exemplars": 3}
    return {}


def _select_model_specs(include_clip: bool) -> tuple[tuple[str, str], ...]:
    specs = [spec for spec in get_model_specs(include_clip=include_clip) if spec[0] != "convnext_tiny"]
    return tuple(specs)


def summarize_corruption_sweep(rows: list[dict[str, object]], output_dir: Path) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["category"]), str(row["corruption_label"]))
        grouped.setdefault(key, []).append(row)

    by_regime: list[dict[str, object]] = []
    for (category, label), group in grouped.items():
        ident_values = [float(row["ident_accuracy"]) for row in group]
        gen_values = [float(row["gen_accuracy"]) for row in group]
        by_regime.append(
            {
                "category": category,
                "corruption_label": label,
                "corruption_mode": group[0]["corruption_mode"],
                "decision_noise_std": float(group[0]["decision_noise_std"]),
                "ident_accuracy_mean": sum(ident_values) / len(ident_values),
                "ident_accuracy_min": min(ident_values),
                "ident_accuracy_max": max(ident_values),
                "gen_accuracy_mean": sum(gen_values) / len(gen_values),
                "human_rdm_spearman_max": max(float(row["human_rdm_spearman"]) for row in group),
            }
        )

    useful = [
        row for row in by_regime
        if 40.0 <= float(row["ident_accuracy_mean"]) <= 90.0 and float(row["gen_accuracy_mean"]) > 5.0
    ]
    summary = {
        "n_rows": len(rows),
        "n_regimes": len(by_regime),
        "useful_regimes": sorted(useful, key=lambda row: (row["category"], -row["gen_accuracy_mean"], row["ident_accuracy_mean"])),
        "best_ident_breakers": sorted(by_regime, key=lambda row: (abs(row["ident_accuracy_mean"] - 70.0), -row["gen_accuracy_mean"]))[:10],
    }
    with (output_dir / "corruption_sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    model_specs = _select_model_specs(include_clip=args.include_clip)
    decision_levels = DECISION_NOISE_LEVELS if args.decision_noise_stage else (0.0,)
    corruption_configs = CORRUPTION_CONFIGS[: args.max_configs] if args.max_configs is not None else CORRUPTION_CONFIGS

    for category in args.categories:
        category_kwargs = _default_category_kwargs(category)
        for config in corruption_configs:
            for decision_noise_std in decision_levels:
                label = _corruption_label(
                    corruption_mode=str(config["corruption_mode"]),
                    noise_std=float(config.get("noise_std", 5.0)),
                    max_shift=int(config.get("max_shift", 2)),
                    occlusion_frac=float(config.get("occlusion_frac", 0.3)),
                    mask_frac=float(config.get("mask_frac", 0.3)),
                    n_chunks=int(config.get("n_chunks", 6)),
                    affine_strength=float(config.get("affine_strength", 0.05)),
                    decision_noise_std=decision_noise_std,
                )
                run_output_dir = output_dir / category / label
                summary = run_naturalistic_baseline(
                    category=category,
                    device=args.device,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    output_dir=run_output_dir,
                    dataset_pkl=args.dataset_pkl,
                    image_root=args.image_root,
                    model_specs=model_specs,
                    decision_noise_std=decision_noise_std,
                    save_preview=True,
                    **category_kwargs,
                    **config,
                )
                csv_path = run_output_dir / f"{category}_combined.csv"
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    rows.extend(csv.DictReader(handle))
                print(json.dumps({
                    "category": category,
                    "corruption_mode": config["corruption_mode"],
                    "corruption_label": summary["corruption_label"],
                    "decision_noise_std": decision_noise_std,
                    "preview_path": summary["preview_path"],
                }))

    if rows:
        fieldnames = list(rows[0].keys())
        with (output_dir / "corruption_sweep_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize_corruption_sweep(rows, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
