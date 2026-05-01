import numpy as np
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.image_generator import generate_square_stimuli, perturb_image
from vision.feature_preprocess import LayerwisePreprocessor


def test_image_generator_and_perturb_smoke():
    images, metadata = generate_square_stimuli(
        n_per_color=2,
        image_size=64,
        square_size=16,
        fixed_position=True,
        seed=3,
    )
    assert len(images) == 12  # 6 default colors * 2
    assert len(metadata) == 12
    assert images[0].size == (64, 64)
    perturbed = perturb_image(images[0], noise_std=1.0, max_shift=1, seed=1)
    assert perturbed.size == images[0].size


def test_layerwise_preprocessor_smoke():
    rng = np.random.default_rng(0)
    features = {
        "layer_0": rng.normal(size=(10, 16)).astype(np.float32),
        "layer_6": rng.normal(size=(10, 32)).astype(np.float32),
        "layer_11": rng.normal(size=(10, 48)).astype(np.float32),
    }
    prep = LayerwisePreprocessor(use_zscore=True, l2_normalize=True)
    out = prep.fit_transform(features)
    assert set(out.keys()) == set(features.keys())
    for arr in out.values():
        assert arr.ndim == 2
        assert np.isfinite(arr).all()


def test_model_wrapper_extract_smoke():
    pytest.importorskip("timm")

    from vision.model_wrapper import VisionEmbeddingWrapper

    images, _ = generate_square_stimuli(
        n_per_color=1,
        image_size=64,
        square_size=16,
        fixed_position=True,
        seed=11,
    )
    wrapper = VisionEmbeddingWrapper(model_name="vit_base_patch16_224", pretrained=False, device="cpu")
    feats = wrapper.extract(images, batch_size=4)
    assert len(feats) == 3
    for key, arr in feats.items():
        assert key.startswith("layer_")
        assert arr.shape[0] == len(images)
        assert np.isfinite(arr).all()
