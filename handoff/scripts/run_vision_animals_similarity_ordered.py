from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
from pathlib import Path

from handoff.scripts._bootstrap import bootstrap_script_imports

bootstrap_script_imports(__file__)

from handoff.lib.graph_display import LEGACY_ANIMALS_SETTING_NAMES
from handoff.lib.vision_runner import (
    DEFAULT_DATASET_PKL,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_SIMILARITY_OUTPUT,
    dispatch_animals_similarity_ordered,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate similarity-ordered animals layerwise graphs.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_SIMILARITY_OUTPUT))
    parser.add_argument("--dataset-pkl", default=str(DEFAULT_DATASET_PKL))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numba")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-settings", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--max-dam-orders", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Single setting, 1 seed, 2 layers, dam_n=2 only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_settings = 1 if args.quick else args.max_settings
    max_seeds = 1 if args.quick else args.max_seeds
    max_layers = 2 if args.quick else args.max_layers
    max_dam_orders = 1 if args.quick else args.max_dam_orders
    written = dispatch_animals_similarity_ordered(
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        dataset_pkl=args.dataset_pkl,
        image_root=args.image_root,
        backend=args.backend,
        n_seeds=args.n_seeds,
        max_settings=max_settings,
        max_seeds=max_seeds,
        max_dam_orders=max_dam_orders,
        max_layers=max_layers,
        setting_names=LEGACY_ANIMALS_SETTING_NAMES,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
