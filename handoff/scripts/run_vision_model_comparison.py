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

from handoff.lib.vision_runner import dispatch_model_comparison, handoff_output
from vision.model_comparison import DEFAULT_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic multi-model baseline comparison.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(handoff_output("model_comparison")))
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODEL_NAMES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = list(args.models)
    if len(models) == 1 and models[0] == "quick":
        models = ["vit_base_patch16_224"]
    summary = dispatch_model_comparison(
        model_names=models,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
