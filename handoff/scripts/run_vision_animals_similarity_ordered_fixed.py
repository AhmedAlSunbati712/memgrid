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
    DEFAULT_SIMILARITY_FIXED_OUTPUT,
    DEFAULT_SIMILARITY_OUTPUT,
    dispatch_animals_similarity_ordered_fixed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process similarity-ordered animals plots with explicit distance-rank ordering."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_SIMILARITY_OUTPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_SIMILARITY_FIXED_OUTPUT))
    parser.add_argument("--max-settings", type=int, default=None)
    parser.add_argument("--max-dam-orders", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Single setting, dam_n=2 only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_settings = 1 if args.quick else args.max_settings
    max_dam_orders = 1 if args.quick else args.max_dam_orders
    written = dispatch_animals_similarity_ordered_fixed(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        max_settings=max_settings,
        max_dam_orders=max_dam_orders,
        setting_names=LEGACY_ANIMALS_SETTING_NAMES,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
