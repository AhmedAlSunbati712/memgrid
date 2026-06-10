from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib.pyplot as plt

from handoff.scripts._bootstrap import bootstrap_script_imports

bootstrap_script_imports(__file__)

from handoff.lib import grid_defaults as defaults
from handoff.lib.grid_runner import run_breakit_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 4b break-it grid DAM sweeps.")
    parser.add_argument(
        "--experiment",
        choices=("orientations", "base_freq", "modules", "cells", "distributions", "all"),
        default="all",
    )
    parser.add_argument("--values", nargs="*", default=None, help="Override sweep values for the experiment.")
    parser.add_argument("--scale-min", type=float, default=1.0)
    parser.add_argument("--scale-max", type=float, default=3.2)
    parser.add_argument("--n-scales", type=int, default=22)
    parser.add_argument("--n-values", nargs="*", type=int, default=None)
    parser.add_argument("--k-values", nargs="*", type=int, default=None)
    parser.add_argument("--n-modules", type=int, default=defaults.BASE_ENCODER["n_modules"])
    parser.add_argument("--n-orientations", type=int, default=defaults.BASE_ENCODER["n_orientations"])
    parser.add_argument("--n-cells-per-orientation", type=int, default=defaults.BASE_ENCODER["n_cells_per_orientation"])
    parser.add_argument("--base-freq", type=float, default=defaults.BASE_ENCODER["base_freq"])
    parser.add_argument("--steps-multiplier", type=int, default=defaults.STEPS_MULTIPLIER)
    parser.add_argument("--backend", choices=("numpy", "numba"), default=defaults.RETRIEVAL_BACKEND)
    parser.add_argument("--num-trials", type=int, default=defaults.NUM_TRIALS_FULL)
    parser.add_argument("--num-test-points", type=int, default=defaults.NUM_TEST_POINTS)
    parser.add_argument("--noise-level", type=float, default=defaults.NOISE_LEVEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(defaults.DEFAULT_BREAKIT_OUTPUT))
    parser.add_argument("--quick", action="store_true", help="One setting per experiment, tiny sweep.")
    parser.add_argument("--save-csv", action="store_true", help="Also write optimum_c.csv files.")
    return parser.parse_args()


def _resolve_setting_values(experiment_name: str, args: argparse.Namespace) -> list[Any]:
    if args.values is not None:
        spec = defaults.BREAKIT_EXPERIMENTS[experiment_name]
        if spec["param"] == "sample_2d_fn":
            return list(args.values)
        return [type(spec["values"][0])(v) for v in args.values]
    spec = defaults.BREAKIT_EXPERIMENTS[experiment_name]
    if args.quick:
        return list(spec["quick_values"])
    return list(spec["values"])


def main() -> None:
    args = parse_args()
    if args.quick:
        scale_factors = list(defaults.QUICK_SCALE_FACTORS)
        n_values = list(defaults.QUICK_N_VALUES)
        k_values = list(defaults.QUICK_K_VALUES)
        num_trials = defaults.NUM_TRIALS_QUICK
    else:
        scale_factors = list(np.linspace(args.scale_min, args.scale_max, args.n_scales))
        n_values = args.n_values or list(defaults.N_VALUES_4B)
        k_values = args.k_values or list(defaults.K_VALUES_4B)
        num_trials = args.num_trials

    base_encoder = {
        "n_modules": args.n_modules,
        "n_orientations": args.n_orientations,
        "n_cells_per_orientation": args.n_cells_per_orientation,
        "base_freq": args.base_freq,
    }

    experiment_names = (
        list(defaults.BREAKIT_EXPERIMENTS.keys())
        if args.experiment == "all"
        else [args.experiment]
    )
    if args.quick and args.experiment == "all":
        experiment_names = [experiment_names[0]]

    output_dir = Path(args.output_dir)
    all_outputs: dict[str, dict[str, str]] = {}

    for experiment_name in experiment_names:
        setting_values = _resolve_setting_values(experiment_name, args)
        section_dir = output_dir / experiment_name
        rows, _raw, figures = run_breakit_section(
            experiment_name,
            output_dir=section_dir,
            setting_values=setting_values,
            scale_factors=scale_factors,
            n_values=n_values,
            K_values=k_values,
            base_encoder=base_encoder,
            noise_level=args.noise_level,
            num_trials=num_trials,
            num_test_points=args.num_test_points,
            steps_multiplier=args.steps_multiplier,
            retrieval_backend=args.backend,
            seed=args.seed,
            save_csv=args.save_csv,
        )
        for fig in figures:
            plt.close(fig)
        all_outputs[experiment_name] = {
            "output_dir": str(section_dir),
            "num_settings": len(rows),
            "num_figures": len(figures),
        }

    print(json.dumps(all_outputs, indent=2))


if __name__ == "__main__":
    main()
