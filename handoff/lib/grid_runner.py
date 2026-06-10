from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from handoff.lib.paths import ensure_handoff_paths

ensure_handoff_paths()

from experiments import (
    get_generalization_optimum_c,
    run_breakit_sweep,
    run_multiscale_ident_gen_sweep,
    sample_biased,
    sample_clustered,
    sample_uniform,
    structure_preservation_analysis,
)
from plotting import plot_tradeoff_breakit, summarize_multiscale_tradeoff_table

from handoff.lib import grid_defaults as defaults


def _nested_results_to_json(results: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c_key, n_map in results.items():
        c_str = f"{float(c_key):.6f}"
        out[c_str] = {}
        for n_key, k_map in n_map.items():
            out[c_str][str(int(n_key))] = {
                str(int(k_key)): {
                    "accuracy": float(v["accuracy"]),
                    "avg_sim": float(v["avg_sim"]),
                    "std_sim": float(v["std_sim"]),
                }
                for k_key, v in k_map.items()
            }
    return out


def _load_nested_results(data: dict[str, Any]) -> dict:
    out: dict = {}
    for c_str, n_map in data.items():
        c_key = float(c_str)
        out[c_key] = {}
        for n_str, k_map in n_map.items():
            n_key = int(n_str)
            out[c_key][n_key] = {int(k_str): dict(v) for k_str, v in k_map.items()}
    return out


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(label))


def scale_factors_from_results(
    ident_results: dict,
    scale_factors: list[float] | np.ndarray | None = None,
) -> list[float]:
    if not ident_results:
        return [float(c) for c in scale_factors] if scale_factors is not None else []

    by_round = {round(float(key), 6): float(key) for key in ident_results.keys()}
    if scale_factors is None:
        return sorted(by_round.values())

    aligned: list[float] = []
    for candidate in scale_factors:
        c = float(candidate)
        if c in ident_results:
            aligned.append(c)
            continue
        rounded = round(c, 6)
        if rounded in by_round:
            aligned.append(by_round[rounded])
            continue
        raise KeyError(f"Missing scale factor {c} in saved results")
    return aligned


def load_breakit_raw_entry(path: Path, label: str) -> tuple[dict, dict, list[float]]:
    payload = json.loads(path.read_text())
    if label not in payload:
        raise KeyError(f"Label {label!r} not found in {path}")
    ident_results = _load_nested_results(payload[label]["ident"])
    gen_results = _load_nested_results(payload[label]["gen"])
    scale_factors = scale_factors_from_results(ident_results)
    return ident_results, gen_results, scale_factors


def make_structure_preservation_figure(
    structure: dict[float, dict[str, float]],
) -> Figure:
    scale_factors = sorted(float(c) for c in structure.keys())
    correlations = [float(structure[c]["correlation"]) for c in scale_factors]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scale_factors, correlations, marker="o")
    ax.set_xlabel("Scale factor c")
    ax.set_ylabel("Distance correlation")
    ax.set_title("Structure preservation vs scale factor")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_multiscale_tradeoff_figure(
    ident_results: dict,
    gen_results: dict,
    scale_factors: list[float],
    n_values: list[int],
    K_values: list[int],
) -> Figure:
    plot_scale_factors = scale_factors_from_results(ident_results, scale_factors)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(n_values), 2)))
    for ni, n in enumerate(n_values):
        x_gen = [
            float(np.mean([gen_results[c][n][K]["accuracy"] for K in K_values]))
            for c in plot_scale_factors
        ]
        y_ident = [
            float(np.mean([ident_results[c][n][K]["accuracy"] for K in K_values]))
            for c in plot_scale_factors
        ]
        ax.scatter(x_gen, y_ident, color=colors[ni], s=60, label=f"n={n}", alpha=0.8)
    ax.set_xlabel("Generalization Accuracy (%)")
    ax.set_ylabel("Identification Accuracy (%)")
    ax.set_title("Multiscale ident vs gen tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_breakit_tradeoff_figure(
    ident_results: dict,
    gen_results: dict,
    scale_factors: list[float],
    n_values: list[int],
    K_values: list[int],
    *,
    title_suffix: str = "",
) -> Figure:
    plot_scale_factors = scale_factors_from_results(ident_results, scale_factors)
    return plot_tradeoff_breakit(
        ident_results,
        gen_results,
        plot_scale_factors,
        n_values,
        K_values,
        title_suffix=title_suffix,
    )


def save_multiscale_results(
    output_dir: Path,
    *,
    ident_results: dict,
    gen_results: dict,
    scale_factors: np.ndarray | list[float],
    n_values: list[int],
    K_values: list[int],
    config: dict[str, Any],
    save_csv: bool = False,
) -> tuple[dict[str, Path], Figure]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_list = [float(c) for c in scale_factors]

    ident_path = output_dir / "ident_results.json"
    gen_path = output_dir / "gen_results.json"
    config_path = output_dir / "config.json"
    ident_path.write_text(json.dumps(_nested_results_to_json(ident_results), indent=2) + "\n")
    gen_path.write_text(json.dumps(_nested_results_to_json(gen_results), indent=2) + "\n")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    paths: dict[str, Path] = {
        "ident_results": ident_path,
        "gen_results": gen_path,
        "config": config_path,
    }

    if save_csv:
        summary = summarize_multiscale_tradeoff_table(
            ident_results,
            gen_results,
            scale_list,
            n_values,
            K_values,
            base_freq=config.get("base_freq", defaults.BASE_FREQ),
            n_modules=config.get("n_modules", defaults.N_MODULES),
        )
        csv_path = output_dir / "tradeoff_summary.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "scale_c",
                    "n",
                    "K",
                    "ident_accuracy",
                    "gen_accuracy",
                    "diff_ident_minus_gen",
                    "tradeoff_label",
                ],
            )
            writer.writeheader()
            for row in summary["rows"]:
                writer.writerow(
                    {
                        "scale_c": row["scale_c"],
                        "n": row["n"],
                        "K": row["K"],
                        "ident_accuracy": row["ident_accuracy"],
                        "gen_accuracy": row["gen_accuracy"],
                        "diff_ident_minus_gen": row["diff_ident_minus_gen"],
                        "tradeoff_label": row["tradeoff_label"],
                    }
                )
        paths["tradeoff_summary"] = csv_path

    fig = make_multiscale_tradeoff_figure(
        ident_results, gen_results, scale_list, n_values, K_values
    )
    plot_path = output_dir / "tradeoff_by_n.png"
    fig.savefig(plot_path, dpi=120)
    paths["tradeoff_plot"] = plot_path
    return paths, fig


def load_multiscale_results(output_dir: Path) -> tuple[dict, dict, dict, list[float]]:
    ident_results = _load_nested_results(json.loads((output_dir / "ident_results.json").read_text()))
    gen_results = _load_nested_results(json.loads((output_dir / "gen_results.json").read_text()))
    config = json.loads((output_dir / "config.json").read_text())
    scale_factors = scale_factors_from_results(ident_results, config.get("scale_factors"))
    return ident_results, gen_results, config, scale_factors


def run_multiscale_sweep(
    *,
    scale_factors: np.ndarray | list[float],
    n_values: list[int],
    K_values: list[int],
    n_modules: int = defaults.N_MODULES,
    n_orientations: int = defaults.N_ORIENTATIONS,
    n_cells_per_orientation: int = defaults.N_CELLS_PER_ORIENTATION,
    base_freq: float = defaults.BASE_FREQ,
    noise_level: float = defaults.NOISE_LEVEL,
    num_trials: int = defaults.NUM_TRIALS_FULL,
    num_test_points: int = defaults.NUM_TEST_POINTS,
    steps_multiplier: int = defaults.STEPS_MULTIPLIER,
    beta: float = defaults.BETA,
    alpha: float = defaults.ALPHA,
    lmbda: float = defaults.LMBDA,
    retrieval_backend: str = defaults.RETRIEVAL_BACKEND,
    seed: int | None = None,
) -> tuple[dict, dict]:
    if seed is not None:
        np.random.seed(seed)
    ident_results, gen_results = run_multiscale_ident_gen_sweep(
        scale_factors,
        n_values,
        K_values,
        noise_level=noise_level,
        num_test_points=num_test_points,
        num_trials=num_trials,
        steps_multiplier=steps_multiplier,
        n_modules=n_modules,
        n_orientations=n_orientations,
        n_cells_per_orientation=n_cells_per_orientation,
        base_freq=base_freq,
        beta=beta,
        alpha=alpha,
        lmbda=lmbda,
        retrieval_backend=retrieval_backend,
    )
    return ident_results, gen_results


@dataclass(frozen=True)
class BreakitRow:
    setting_value: str
    optimum_c: float
    gen_accuracy: float
    ident_accuracy: float


def _sample_fn(name: str) -> Callable[[int], np.ndarray]:
    if name == "uniform":
        return sample_uniform
    if name == "clustered":
        return lambda K: sample_clustered(K)
    if name == "biased":
        return sample_biased
    raise ValueError(f"Unknown distribution: {name}")


def run_breakit_experiment(
    experiment_name: str,
    *,
    setting_values: list[Any],
    scale_factors: np.ndarray | list[float],
    n_values: list[int],
    K_values: list[int],
    base_encoder: dict[str, Any],
    noise_level: float = defaults.NOISE_LEVEL,
    num_trials: int = defaults.NUM_TRIALS_FULL,
    num_test_points: int = defaults.NUM_TEST_POINTS,
    steps_multiplier: int = defaults.STEPS_MULTIPLIER,
    retrieval_backend: str = defaults.RETRIEVAL_BACKEND,
    seed: int | None = None,
) -> tuple[list[BreakitRow], dict[str, tuple[dict, dict]]]:
    spec = defaults.BREAKIT_EXPERIMENTS[experiment_name]
    param = str(spec["param"])
    rows: list[BreakitRow] = []
    raw: dict[str, tuple[dict, dict]] = {}

    for value in setting_values:
        if seed is not None:
            np.random.seed(seed)

        encoder_kwargs = dict(base_encoder)
        sample_2d_fn = None
        label = str(value)

        if param == "sample_2d_fn":
            sample_2d_fn = _sample_fn(str(value))
        else:
            encoder_kwargs[param] = value

        ident_res, gen_res = run_breakit_sweep(
            scale_factors,
            n_values,
            K_values,
            noise_level=noise_level,
            num_trials=num_trials,
            num_test_points=num_test_points,
            steps_multiplier=steps_multiplier,
            retrieval_backend=retrieval_backend,
            sample_2d_fn=sample_2d_fn,
            **encoder_kwargs,
        )
        raw[label] = (ident_res, gen_res)
        best_c, best_gen, mean_ident = get_generalization_optimum_c(
            gen_res,
            ident_res,
            scale_factors,
            n_values,
            K_values,
            n_for_opt=defaults.N_FOR_OPT,
        )
        rows.append(
            BreakitRow(
                setting_value=label,
                optimum_c=float(best_c),
                gen_accuracy=float(best_gen),
                ident_accuracy=float(mean_ident),
            )
        )
    return rows, raw


def save_breakit_optimum_summary_json(
    output_dir: Path,
    experiment_name: str,
    rows: list[BreakitRow],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{experiment_name}_optimum_summary.json"
    summary_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def save_breakit_experiment(
    output_dir: Path,
    experiment_name: str,
    *,
    rows: list[BreakitRow],
    raw: dict[str, tuple[dict, dict]],
    scale_factors: list[float],
    n_values: list[int],
    K_values: list[int],
    save_csv: bool = False,
) -> tuple[dict[str, Path], list[Figure]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = experiment_name
    paths: dict[str, Path] = {}

    if save_csv:
        optimum_path = output_dir / f"{prefix}_optimum_c.csv"
        with optimum_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["setting_value", "optimum_c", "gen_accuracy", "ident_accuracy"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        paths["optimum_c"] = optimum_path

    paths["optimum_summary"] = save_breakit_optimum_summary_json(output_dir, experiment_name, rows)

    raw_path = output_dir / f"{prefix}_raw.json"
    raw_payload: dict[str, Any] = {}
    for label, (ident_res, gen_res) in raw.items():
        raw_payload[label] = {
            "ident": _nested_results_to_json(ident_res),
            "gen": _nested_results_to_json(gen_res),
        }
    raw_path.write_text(json.dumps(raw_payload, indent=2) + "\n")
    paths["raw"] = raw_path

    figures: list[Figure] = []
    scale_list = [float(c) for c in scale_factors]
    for label, (ident_res, gen_res) in raw.items():
        title_suffix = f" ({experiment_name}={label})"
        fig = make_breakit_tradeoff_figure(
            ident_res,
            gen_res,
            scale_list,
            n_values,
            K_values,
            title_suffix=title_suffix,
        )
        plot_path = output_dir / f"{prefix}_{_sanitize_label(label)}_tradeoff.png"
        fig.savefig(plot_path, dpi=120)
        paths[f"tradeoff_plot_{_sanitize_label(label)}"] = plot_path
        figures.append(fig)

    return paths, figures


def run_breakit_section(
    experiment_name: str,
    *,
    output_dir: Path,
    scale_factors: np.ndarray | list[float],
    n_values: list[int],
    K_values: list[int],
    base_encoder: dict[str, Any],
    setting_values: list[Any] | None = None,
    noise_level: float = defaults.NOISE_LEVEL,
    num_trials: int = defaults.NUM_TRIALS_FULL,
    num_test_points: int = defaults.NUM_TEST_POINTS,
    steps_multiplier: int = defaults.STEPS_MULTIPLIER,
    retrieval_backend: str = defaults.RETRIEVAL_BACKEND,
    seed: int | None = None,
    save_csv: bool = False,
) -> tuple[list[BreakitRow], dict[str, tuple[dict, dict]], list[Figure]]:
    spec = defaults.BREAKIT_EXPERIMENTS[experiment_name]
    values = list(setting_values if setting_values is not None else spec["values"])
    rows, raw = run_breakit_experiment(
        experiment_name,
        setting_values=values,
        scale_factors=scale_factors,
        n_values=n_values,
        K_values=K_values,
        base_encoder=base_encoder,
        noise_level=noise_level,
        num_trials=num_trials,
        num_test_points=num_test_points,
        steps_multiplier=steps_multiplier,
        retrieval_backend=retrieval_backend,
        seed=seed,
    )
    _, figures = save_breakit_experiment(
        output_dir,
        experiment_name,
        rows=rows,
        raw=raw,
        scale_factors=[float(c) for c in scale_factors],
        n_values=n_values,
        K_values=K_values,
        save_csv=save_csv,
    )
    return rows, raw, figures


def run_structure_preservation(
    scale_factors: list[float] | np.ndarray,
    *,
    n_modules: int = defaults.N_MODULES,
    n_orientations: int = defaults.N_ORIENTATIONS,
    n_cells_per_orientation: int = defaults.N_CELLS_PER_ORIENTATION,
    base_freq: float = defaults.BASE_FREQ,
    seed: int | None = None,
) -> dict[float, dict[str, float]]:
    if seed is not None:
        np.random.seed(seed)
    return structure_preservation_analysis(
        scale_factors,
        n_modules=n_modules,
        n_orientations=n_orientations,
        n_cells_per_orientation=n_cells_per_orientation,
        base_freq=base_freq,
    )
