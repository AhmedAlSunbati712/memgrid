from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from PIL import ImageDraw


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
    color_jitter_std: float = 0.0,
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
    if color_jitter_std < 0:
        raise ValueError("color_jitter_std must be non-negative")

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

            if color_jitter_std > 0:
                actual_rgb_arr = np.asarray(color_rgb, dtype=np.float32)
                actual_rgb_arr += rng.normal(0.0, color_jitter_std, size=3).astype(np.float32)
                actual_rgb = tuple(int(v) for v in np.clip(np.rint(actual_rgb_arr), 0, 255))
            else:
                actual_rgb = tuple(int(v) for v in color_rgb)

            arr = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            arr[:, :] = np.asarray(background_rgb, dtype=np.uint8)
            arr[cy : cy + square_size, cx : cx + square_size] = np.asarray(actual_rgb, dtype=np.uint8)

            stimulus_id = f"{color_name}_{i:04d}"
            images.append(Image.fromarray(arr, mode="RGB"))
            metadata.append(
                StimulusRecord(
                    stimulus_id=stimulus_id,
                    color_name=color_name,
                    color_rgb=actual_rgb,
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
        fill_rgb = arr[0, 0].copy()
        shifted = np.empty_like(arr)
        shifted[:, :] = fill_rgb

        src_y0 = max(0, -shift_y)
        src_y1 = arr.shape[0] - max(0, shift_y)
        dst_y0 = max(0, shift_y)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        src_x0 = max(0, -shift_x)
        src_x1 = arr.shape[1] - max(0, shift_x)
        dst_x0 = max(0, shift_x)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
        arr = shifted

    if noise_std > 0:
        arr += rng.normal(0.0, noise_std, arr.shape).astype(np.float32)

    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _occlude_image(
    image: Image.Image,
    *,
    occlusion_frac: float,
    seed: int | None = None,
) -> Image.Image:
    if not (0.0 <= occlusion_frac <= 1.0):
        raise ValueError("occlusion_frac must be in [0, 1]")
    rng = np.random.default_rng(seed)
    out = image.convert("RGB").copy()
    w, h = out.size
    area = max(1, int(round(occlusion_frac * w * h)))
    rect_w = max(1, int(round(np.sqrt(area))))
    rect_h = max(1, int(round(area / rect_w)))
    rect_w = min(rect_w, w)
    rect_h = min(rect_h, h)
    x0 = int(rng.integers(0, max(1, w - rect_w + 1)))
    y0 = int(rng.integers(0, max(1, h - rect_h + 1)))
    draw = ImageDraw.Draw(out)
    draw.rectangle([x0, y0, x0 + rect_w - 1, y0 + rect_h - 1], fill=(0, 0, 0))
    return out


def _multi_cutout_image(
    image: Image.Image,
    *,
    mask_frac: float,
    n_chunks: int,
    seed: int | None = None,
) -> Image.Image:
    if not (0.0 <= mask_frac <= 1.0):
        raise ValueError("mask_frac must be in [0, 1]")
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")
    rng = np.random.default_rng(seed)
    out = image.convert("RGB").copy()
    w, h = out.size
    total_area = max(1, int(round(mask_frac * w * h)))
    chunk_area = max(1, total_area // n_chunks)
    draw = ImageDraw.Draw(out)
    for _ in range(n_chunks):
        rect_w = max(1, int(round(np.sqrt(chunk_area))))
        rect_h = max(1, int(round(chunk_area / rect_w)))
        rect_w = min(rect_w, w)
        rect_h = min(rect_h, h)
        x0 = int(rng.integers(0, max(1, w - rect_w + 1)))
        y0 = int(rng.integers(0, max(1, h - rect_h + 1)))
        draw.rectangle([x0, y0, x0 + rect_w - 1, y0 + rect_h - 1], fill=(0, 0, 0))
    return out


def _warp_image(
    image: Image.Image,
    *,
    affine_strength: float,
    seed: int | None = None,
) -> Image.Image:
    if affine_strength < 0.0:
        raise ValueError("affine_strength must be non-negative")
    rng = np.random.default_rng(seed)
    w, h = image.size
    shear_x = float(rng.uniform(-affine_strength, affine_strength))
    shear_y = float(rng.uniform(-affine_strength, affine_strength))
    scale_x = float(1.0 + rng.uniform(-affine_strength, affine_strength))
    scale_y = float(1.0 + rng.uniform(-affine_strength, affine_strength))
    trans_x = float(rng.uniform(-affine_strength, affine_strength) * w)
    trans_y = float(rng.uniform(-affine_strength, affine_strength) * h)
    coeffs = (scale_x, shear_x, trans_x, shear_y, scale_y, trans_y)
    return image.convert("RGB").transform(
        (w, h),
        Image.AFFINE,
        data=coeffs,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def corrupt_image(
    image: Image.Image,
    *,
    mode: str,
    seed: int | None = None,
    noise_std: float = 5.0,
    max_shift: int = 2,
    occlusion_frac: float = 0.3,
    mask_frac: float = 0.3,
    n_chunks: int = 6,
    affine_strength: float = 0.05,
) -> Image.Image:
    """
    Apply a named corruption mode used to build identification probes.
    """
    mode = str(mode)
    if mode == "noise_shift":
        return perturb_image(image, noise_std=noise_std, max_shift=max_shift, seed=seed)
    if mode == "occlusion":
        return _occlude_image(image, occlusion_frac=occlusion_frac, seed=seed)
    if mode == "multi_cutout":
        return _multi_cutout_image(image, mask_frac=mask_frac, n_chunks=n_chunks, seed=seed)
    if mode == "warp":
        return _warp_image(image, affine_strength=affine_strength, seed=seed)
    raise ValueError(f"Unknown corruption mode: {mode}")


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
