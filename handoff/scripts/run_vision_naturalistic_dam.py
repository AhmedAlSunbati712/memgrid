from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import json
from pathlib import Path

from handoff.scripts._bootstrap import bootstrap_script_imports

bootstrap_script_imports(__file__)

from handoff.lib.vision_runner import dispatch_naturalistic_dam, handoff_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DAM on the naturalistic benchmark.")
    parser.add_argument("--category", choices=("fruits", "vegetables", "animals"), default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(handoff_output("naturalistic_dam")))
    parser.add_argument("--baseline-dir", default=str(handoff_output("naturalistic")))
    parser.add_argument("--dataset-pkl", default="datasets_peterson.pkl")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--include-clip", action="store_true", default=True)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numba")
    parser.add_argument("--model-spec", nargs="*", default=None)
    parser.add_argument("--max-stage-a-configs", type=int, default=None)
    parser.add_argument("--skip-stage-b", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Single category animals, capped stage A.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category = "animals" if args.quick and args.category is None else args.category
    include_clip = False if args.no_clip or args.quick else args.include_clip
    max_stage_a = 1 if args.quick else args.max_stage_a_configs
    result = dispatch_naturalistic_dam(
        category=category,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        baseline_dir=Path(args.baseline_dir),
        dataset_pkl=args.dataset_pkl,
        image_root=args.image_root,
        include_clip=include_clip,
        backend=args.backend,
        model_specs_override=(
            tuple(tuple(spec.split("::", 1)) for spec in args.model_spec) if args.model_spec else None
        ),
        max_stage_a_configs=max_stage_a,
        skip_stage_b=args.skip_stage_b or args.quick,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
