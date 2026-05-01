from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision.feature_preprocess import LayerwisePreprocessor
from vision.image_generator import build_preview_grid, metadata_to_dicts, generate_square_stimuli
from vision.model_wrapper import VisionEmbeddingWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic stimuli and cache layer features.")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-per-color", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--square-size", type=int, default=56)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="results/vision",
        help="Relative to repository root unless absolute path is provided.",
    )
    parser.add_argument("--no-zscore", action="store_true", help="Disable per-layer z-score normalization.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Use randomly initialized model weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, metadata = generate_square_stimuli(
        n_per_color=args.n_per_color,
        image_size=args.image_size,
        square_size=args.square_size,
        seed=args.seed,
        fixed_position=True,
    )

    wrapper = VisionEmbeddingWrapper(
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        device=args.device,
        pooling="auto",
    )
    features_raw = wrapper.extract(images, layer_indices=None, batch_size=args.batch_size)

    preprocessor = LayerwisePreprocessor(use_zscore=not args.no_zscore, l2_normalize=True)
    features = preprocessor.fit_transform(features_raw)

    preview_path = output_dir / f"preview_synth_{args.model_name}.png"
    build_preview_grid(images[: min(18, len(images))], ncols=6).save(preview_path)

    npz_path = output_dir / f"features_synth_{args.model_name}.npz"
    payload: dict[str, np.ndarray] = {}
    for layer_name, feats in features.items():
        payload[f"features_{layer_name}"] = feats
    payload.update(preprocessor.to_state_dict())
    payload["stimulus_id"] = np.array([m.stimulus_id for m in metadata], dtype=object)
    payload["color_name"] = np.array([m.color_name for m in metadata], dtype=object)
    payload["x"] = np.array([m.x for m in metadata], dtype=np.int32)
    payload["y"] = np.array([m.y for m in metadata], dtype=np.int32)
    np.savez_compressed(npz_path, **payload)

    metadata_json_path = output_dir / f"metadata_synth_{args.model_name}.json"
    metadata_json_path.write_text(json.dumps(metadata_to_dicts(metadata), indent=2), encoding="utf-8")

    print(f"Saved: {npz_path}")
    print(f"Saved: {metadata_json_path}")
    print(f"Saved: {preview_path}")
    print("\nFeature shapes:")
    for layer, feats in features.items():
        print(f"  {layer}: {tuple(feats.shape)}")


if __name__ == "__main__":
    main()
