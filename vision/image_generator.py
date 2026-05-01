from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StimulusRecord:
    stimulus_id: str
    color_name: str
    color_rgb: tuple[int, int, int]
    x: int
    y: int
    image_size: int
    square_size: int


DEFAULT_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red", (220, 20, 60)),
    ("green", (34, 139, 34)),
    ("blue", (30, 144, 255)),
    ("yellow", (255, 215, 0)),
    ("purple", (138, 43, 226)),
    ("orange", (255, 140, 0)),
)


def _resolve_position(image_size: int, square_size: int, x: int | None, y: int | None) -> tuple[int, int]:
    max_coord = image_size - square_size
    if max_coord < 0:
        raise ValueError("square_size must be <= image_size")

    if x is None:
        x = max_coord // 2
    if y is None:
        y = max_coord // 2

    if not (0 <= x <= max_coord and 0 <= y <= max_coord):
        raise ValueError("x/y must place the square fully inside the image")
    return x, y


def generate_square_stimuli(
    n_per_color: int = 8,
    image_size: int = 224,
    square_size: int = 56,
    background_rgb: tuple[int, int, int] = (127, 127, 127),
    colors: Sequence[tuple[str, tuple[int, int, int]]] = DEFAULT_COLORS,
    fixed_position: bool = True,
    x: int | None = None,
    y: int | None = None,
    seed: int = 0,
) -> tuple[list[Image.Image], list[StimulusRecord]]:
    """
    Generate synthetic colored-square images and aligned metadata.

    Returns
    -------
    images: list[PIL.Image.Image]
    metadata: list[StimulusRecord]
    """
    if n_per_color <= 0:
        raise ValueError("n_per_color must be positive")

    x_default, y_default = _resolve_position(image_size, square_size, x, y)
    max_coord = image_size - square_size
    rng = np.random.default_rng(seed)

    images: list[Image.Image] = []
    metadata: list[StimulusRecord] = []

    for color_name, color_rgb in colors:
        for i in range(n_per_color):
            if fixed_position:
                cx, cy = x_default, y_default
            else:
                cx = int(rng.integers(0, max_coord + 1))
                cy = int(rng.integers(0, max_coord + 1))

            arr = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            arr[:, :] = np.asarray(background_rgb, dtype=np.uint8)
            arr[cy : cy + square_size, cx : cx + square_size] = np.asarray(color_rgb, dtype=np.uint8)

            stimulus_id = f"{color_name}_{i:04d}"
            images.append(Image.fromarray(arr, mode="RGB"))
            metadata.append(
                StimulusRecord(
                    stimulus_id=stimulus_id,
                    color_name=color_name,
                    color_rgb=color_rgb,
                    x=cx,
                    y=cy,
                    image_size=image_size,
                    square_size=square_size,
                )
            )

    return images, metadata


def perturb_image(
    image: Image.Image,
    noise_std: float = 5.0,
    max_shift: int = 2,
    seed: int | None = None,
) -> Image.Image:
    """
    Add small pixel noise and a small translation to an RGB image.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(image).astype(np.float32)

    if max_shift > 0:
        shift_x = int(rng.integers(-max_shift, max_shift + 1))
        shift_y = int(rng.integers(-max_shift, max_shift + 1))
        arr = np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))

    if noise_std > 0:
        arr += rng.normal(0.0, noise_std, arr.shape).astype(np.float32)

    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def build_preview_grid(
    images: Sequence[Image.Image],
    ncols: int = 6,
    pad: int = 4,
    pad_rgb: tuple[int, int, int] = (25, 25, 25),
) -> Image.Image:
    """
    Build a simple tiled preview image (no matplotlib dependency).
    """
    if not images:
        raise ValueError("images must be non-empty")
    if ncols <= 0:
        raise ValueError("ncols must be positive")

    w, h = images[0].size
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols

    grid_w = ncols * w + (ncols + 1) * pad
    grid_h = nrows * h + (nrows + 1) * pad
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid[:, :] = np.asarray(pad_rgb, dtype=np.uint8)

    for idx, img in enumerate(images):
        row, col = divmod(idx, ncols)
        y0 = pad + row * (h + pad)
        x0 = pad + col * (w + pad)
        grid[y0 : y0 + h, x0 : x0 + w] = np.asarray(img.convert("RGB"), dtype=np.uint8)

    return Image.fromarray(grid, mode="RGB")


def metadata_to_dicts(metadata: Iterable[StimulusRecord]) -> list[dict[str, object]]:
    return [record.__dict__.copy() for record in metadata]
