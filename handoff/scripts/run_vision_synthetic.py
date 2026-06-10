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

from handoff.lib.vision_runner import handoff_output, run_synthetic_baseline
from vision.ident_corruption import (
    IDENT_DECISION_NOISE_STD,
    IDENT_MAX_SHIFT,
    IDENT_NOISE_STD,
    IDENT_OCCLUSION_FRAC,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic vision layerwise baseline.")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--square-size", type=int, default=56)
    parser.add_argument("--n-per-color", type=int, default=4)
    parser.add_argument("--n-stored-per-color", type=int, default=2)
    parser.add_argument(
        "--task-mode",
        choices=("mixed_color_position", "color_only", "position_only"),
        default="mixed_color_position",
    )
    parser.add_argument("--ident-noise-std", type=float, default=IDENT_NOISE_STD)
    parser.add_argument("--ident-max-shift", type=int, default=IDENT_MAX_SHIFT)
    parser.add_argument("--occlusion-frac", type=float, default=IDENT_OCCLUSION_FRAC)
    parser.add_argument("--decision-noise-std", type=float, default=IDENT_DECISION_NOISE_STD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(handoff_output("synthetic")))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_synthetic_baseline(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        square_size=args.square_size,
        n_per_color=args.n_per_color,
        n_stored_per_color=args.n_stored_per_color,
        task_mode=args.task_mode,
        ident_noise_std=args.ident_noise_std,
        ident_max_shift=args.ident_max_shift,
        ident_occlusion_frac=args.occlusion_frac,
        decision_noise_std=args.decision_noise_std,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        quick=args.quick,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
