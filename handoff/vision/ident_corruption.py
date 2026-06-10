from __future__ import annotations

from PIL import Image

from vision.image_generator import corrupt_image

IDENT_NOISE_STD = 5.0
IDENT_MAX_SHIFT = 2
IDENT_OCCLUSION_FRAC = 0.5
IDENT_DECISION_NOISE_STD = 0.01
IDENT_CORRUPTION_MODE = "noise_shift_occlusion"


def corrupt_ident_probe(
    image: Image.Image,
    *,
    seed: int | None = None,
    noise_std: float = IDENT_NOISE_STD,
    max_shift: int = IDENT_MAX_SHIFT,
    occlusion_frac: float = IDENT_OCCLUSION_FRAC,
) -> Image.Image:
    """Stack noise_shift then occlusion for identification probes."""
    return corrupt_image(
        image,
        mode=IDENT_CORRUPTION_MODE,
        seed=seed,
        noise_std=noise_std,
        max_shift=max_shift,
        occlusion_frac=occlusion_frac,
    )
