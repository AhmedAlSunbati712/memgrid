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

from handoff.lib.vision_runner import dispatch_naturalistic_baseline, handoff_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run naturalistic Peterson baseline.")
    parser.add_argument("--category", default="fruits")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(handoff_output("naturalistic")))
    parser.add_argument("--dataset-pkl", default="datasets_peterson.pkl")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--required-exemplars", type=int, default=None)
    parser.add_argument("--split-mode", choices=("balanced_exemplar_folds", "random_storage_curve"), default=None)
    parser.add_argument("--storage-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--include-clip", action="store_true")
    parser.add_argument("--corruption-mode", default="noise_shift_occlusion")
    parser.add_argument("--noise-std", type=float, default=5.0)
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--occlusion-frac", type=float, default=0.5)
    parser.add_argument("--mask-frac", type=float, default=0.3)
    parser.add_argument("--n-chunks", type=int, default=6)
    parser.add_argument("--affine-strength", type=float, default=0.05)
    parser.add_argument("--decision-noise-std", type=float, default=0.01)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Single seed, no CLIP.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_seeds = 1 if args.quick else args.n_seeds
    include_clip = False if args.quick else args.include_clip
    summary = dispatch_naturalistic_baseline(
        category=args.category,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        dataset_pkl=args.dataset_pkl,
        image_root=args.image_root,
        required_exemplars=args.required_exemplars,
        split_mode=args.split_mode,
        storage_sizes=args.storage_sizes,
        n_seeds=n_seeds,
        include_clip=include_clip,
        corruption_mode=args.corruption_mode,
        noise_std=args.noise_std,
        max_shift=args.max_shift,
        occlusion_frac=args.occlusion_frac,
        mask_frac=args.mask_frac,
        n_chunks=args.n_chunks,
        affine_strength=args.affine_strength,
        decision_noise_std=args.decision_noise_std,
        save_preview=args.save_preview,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
