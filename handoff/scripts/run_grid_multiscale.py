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

import numpy as np

from handoff.scripts._bootstrap import bootstrap_script_imports

bootstrap_script_imports(__file__)

from handoff.lib import grid_defaults as defaults
from handoff.lib.grid_runner import run_multiscale_sweep, save_multiscale_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 4 multiscale ident/gen grid DAM sweep.")
    parser.add_argument("--scale-min", type=float, default=1.0)
    parser.add_argument("--scale-max", type=float, default=3.2)
    parser.add_argument("--n-scales", type=int, default=22)
    parser.add_argument("--n-values", nargs="*", type=int, default=None)
    parser.add_argument("--k-values", nargs="*", type=int, default=None)
    parser.add_argument("--n-modules", type=int, default=defaults.N_MODULES)
    parser.add_argument("--n-orientations", type=int, default=defaults.N_ORIENTATIONS)
    parser.add_argument("--n-cells-per-orientation", type=int, default=defaults.N_CELLS_PER_ORIENTATION)
    parser.add_argument("--base-freq", type=float, default=defaults.BASE_FREQ)
    parser.add_argument("--beta", type=float, default=defaults.BETA)
    parser.add_argument("--alpha", type=float, default=defaults.ALPHA)
    parser.add_argument("--lmbda", type=float, default=defaults.LMBDA)
    parser.add_argument("--steps-multiplier", type=int, default=defaults.STEPS_MULTIPLIER)
    parser.add_argument("--backend", choices=("numpy", "numba"), default=defaults.RETRIEVAL_BACKEND)
    parser.add_argument("--num-trials", type=int, default=defaults.NUM_TRIALS_FULL)
    parser.add_argument("--num-test-points", type=int, default=defaults.NUM_TEST_POINTS)
    parser.add_argument("--noise-level", type=float, default=defaults.NOISE_LEVEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(defaults.DEFAULT_MULTISCALE_OUTPUT))
    parser.add_argument("--quick", action="store_true", help="Small sweep for smoke tests.")
    parser.add_argument("--save-csv", action="store_true", help="Also write tradeoff_summary.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        scale_factors = list(defaults.QUICK_SCALE_FACTORS)
        n_values = list(defaults.QUICK_N_VALUES)
        k_values = list(defaults.QUICK_K_VALUES)
        num_trials = defaults.NUM_TRIALS_QUICK
    else:
        scale_factors = list(np.linspace(args.scale_min, args.scale_max, args.n_scales))
        n_values = args.n_values or list(defaults.FULL_N_VALUES)
        k_values = args.k_values or list(defaults.FULL_K_VALUES)
        num_trials = args.num_trials

    output_dir = Path(args.output_dir)
    ident_results, gen_results = run_multiscale_sweep(
        scale_factors=scale_factors,
        n_values=n_values,
        K_values=k_values,
        n_modules=args.n_modules,
        n_orientations=args.n_orientations,
        n_cells_per_orientation=args.n_cells_per_orientation,
        base_freq=args.base_freq,
        noise_level=args.noise_level,
        num_trials=num_trials,
        num_test_points=args.num_test_points,
        steps_multiplier=args.steps_multiplier,
        beta=args.beta,
        alpha=args.alpha,
        lmbda=args.lmbda,
        retrieval_backend=args.backend,
        seed=args.seed,
    )

    config = {
        "scale_factors": scale_factors,
        "n_values": n_values,
        "k_values": k_values,
        "n_modules": args.n_modules,
        "n_orientations": args.n_orientations,
        "n_cells_per_orientation": args.n_cells_per_orientation,
        "base_freq": args.base_freq,
        "beta": args.beta,
        "alpha": args.alpha,
        "lmbda": args.lmbda,
        "steps_multiplier": args.steps_multiplier,
        "backend": args.backend,
        "num_trials": num_trials,
        "num_test_points": args.num_test_points,
        "noise_level": args.noise_level,
        "seed": args.seed,
        "quick": bool(args.quick),
    }
    paths, _fig = save_multiscale_results(
        output_dir,
        ident_results=ident_results,
        gen_results=gen_results,
        scale_factors=scale_factors,
        n_values=n_values,
        K_values=k_values,
        config=config,
        save_csv=args.save_csv,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
