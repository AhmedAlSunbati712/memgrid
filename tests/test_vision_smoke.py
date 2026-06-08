import numpy as np
from pathlib import Path
import sys
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.image_generator import corrupt_image, generate_square_stimuli, perturb_image
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


def test_color_jitter_makes_fixed_position_samples_non_identical():
    images, metadata = generate_square_stimuli(
        n_per_color=3,
        image_size=32,
        square_size=8,
        colors=(("grayish", (100, 100, 100)),),
        fixed_position=True,
        color_jitter_std=5.0,
        seed=7,
    )

    assert len({record.color_rgb for record in metadata}) > 1
    assert len({(record.x, record.y) for record in metadata}) == 1
    assert len({np.asarray(image).tobytes() for image in images}) > 1


def test_perturb_image_shift_does_not_wrap_pixels():
    arr = np.zeros((5, 5, 3), dtype=np.uint8)
    arr[4, 4] = np.array([255, 0, 0], dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")

    perturbed = perturb_image(image, noise_std=0.0, max_shift=2, seed=0)
    out = np.asarray(perturbed)

    assert out[0, 1, 0] == 0
    assert out[:, :, 0].sum() == 0


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("noise_shift", {"noise_std": 15.0, "max_shift": 8}),
        ("occlusion", {"occlusion_frac": 0.3}),
        ("multi_cutout", {"mask_frac": 0.3, "n_chunks": 6}),
        ("warp", {"affine_strength": 0.05}),
    ],
)
def test_corrupt_image_modes_are_deterministic_and_change_pixels(mode, kwargs):
    images, _ = generate_square_stimuli(
        n_per_color=1,
        image_size=64,
        square_size=16,
        fixed_position=True,
        seed=13,
    )
    image = images[0]
    out_a = corrupt_image(image, mode=mode, seed=99, **kwargs)
    out_b = corrupt_image(image, mode=mode, seed=99, **kwargs)

    assert out_a.size == image.size
    assert np.array_equal(np.asarray(out_a), np.asarray(out_b))
    assert not np.array_equal(np.asarray(out_a), np.asarray(image))


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


def test_model_wrapper_supports_conv_and_transformer_pooling():
    pytest.importorskip("timm")

    from vision.model_wrapper import VisionEmbeddingWrapper

    images, _ = generate_square_stimuli(
        n_per_color=1,
        image_size=64,
        square_size=16,
        fixed_position=True,
        seed=5,
    )

    vit_cls = VisionEmbeddingWrapper(
        model_name="vit_base_patch16_224.augreg_in1k",
        pretrained=False,
        device="cpu",
        pooling="cls",
    )
    vit_mean = VisionEmbeddingWrapper(
        model_name="vit_base_patch16_224.augreg_in1k",
        pretrained=False,
        device="cpu",
        pooling="mean_tokens",
    )
    conv = VisionEmbeddingWrapper(
        model_name="resnet50.a1_in1k",
        pretrained=False,
        device="cpu",
        pooling="auto",
    )

    cls_feats = vit_cls.extract(images, batch_size=2)
    mean_feats = vit_mean.extract(images, batch_size=2)
    conv_feats = conv.extract(images, batch_size=2)

    assert cls_feats["layer_0"].shape == mean_feats["layer_0"].shape
    assert conv_feats["layer_0"].shape[0] == len(images)
    assert not np.allclose(cls_feats["layer_0"], mean_feats["layer_0"])


def test_model_wrapper_supports_clip_transformer_pooling():
    pytest.importorskip("timm")

    from vision.model_wrapper import VisionEmbeddingWrapper

    images, _ = generate_square_stimuli(
        n_per_color=1,
        image_size=64,
        square_size=16,
        fixed_position=True,
        seed=9,
    )

    clip_cls = VisionEmbeddingWrapper(
        model_name="vit_base_patch16_clip_224.openai",
        pretrained=False,
        device="cpu",
        pooling="cls",
    )
    clip_mean = VisionEmbeddingWrapper(
        model_name="vit_base_patch16_clip_224.openai",
        pretrained=False,
        device="cpu",
        pooling="mean_tokens",
    )

    cls_feats = clip_cls.extract(images, batch_size=2)
    mean_feats = clip_mean.extract(images, batch_size=2)

    assert cls_feats["layer_0"].shape == mean_feats["layer_0"].shape
    assert cls_feats["layer_0"].shape[0] == len(images)
    assert not np.allclose(cls_feats["layer_0"], mean_feats["layer_0"])
