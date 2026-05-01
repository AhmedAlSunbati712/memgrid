from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except Exception as exc:  # pragma: no cover
    timm = None
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None

from PIL import Image


def _require_timm() -> None:
    if timm is None:
        raise ImportError(f"timm is required for VisionEmbeddingWrapper: {_TIMM_IMPORT_ERROR}")


def _as_rgb_images(images: Iterable[Image.Image]) -> list[Image.Image]:
    return [img.convert("RGB") for img in images]


def _pool_activation(tensor: torch.Tensor, pooling: str) -> torch.Tensor:
    """
    Convert activation tensor to shape [B, D].
    """
    if tensor.ndim == 2:  # [B, D]
        return tensor
    if tensor.ndim == 3:  # [B, T, C], ViT-style
        if pooling == "cls":
            return tensor[:, 0, :]
        if pooling == "mean_tokens":
            return tensor.mean(dim=1)
        # auto: use cls token by default for transformer token sequences
        return tensor[:, 0, :]
    if tensor.ndim == 4:  # [B, C, H, W], CNN-style
        return tensor.mean(dim=(2, 3))
    return tensor.reshape(tensor.shape[0], -1)


def _resolve_layer_indices(total_layers: int, layer_indices: Sequence[int] | None) -> list[int]:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if layer_indices is None:
        picks = sorted({0, total_layers // 2, total_layers - 1})
        return [int(i) for i in picks]
    resolved: list[int] = []
    for idx in layer_indices:
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise ValueError(f"layer index {idx} is out of range for total_layers={total_layers}")
        resolved.append(int(idx))
    return resolved


@dataclass
class VisionEmbeddingWrapper:
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    device: str = "cpu"
    pooling: str = "auto"  # auto | cls | mean_tokens

    def __post_init__(self) -> None:
        _require_timm()
        self._device = torch.device(self.device)
        self.model = timm.create_model(self.model_name, pretrained=self.pretrained).to(self._device).eval()
        data_cfg = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**data_cfg)

    def list_default_layer_indices(self) -> list[int]:
        if hasattr(self.model, "blocks") and len(self.model.blocks) > 0:
            return _resolve_layer_indices(len(self.model.blocks), None)
        if hasattr(self.model, "feature_info"):
            # fallback for convnets created in standard mode may not expose these cleanly
            n = len(getattr(self.model.feature_info, "info", []))
            if n > 0:
                return _resolve_layer_indices(n, None)
        raise ValueError("Could not infer default layers for this model.")

    def _extract_intermediates(
        self, batch: torch.Tensor, layer_indices: Sequence[int]
    ) -> list[torch.Tensor] | None:
        if not hasattr(self.model, "forward_intermediates"):
            return None
        out = self.model.forward_intermediates(batch, indices=list(layer_indices), intermediates_only=True)
        if isinstance(out, tuple):
            # Some timm models return (last, intermediates) unless intermediates_only honored.
            if len(out) == 2 and isinstance(out[1], (list, tuple)):
                return [x for x in out[1]]
        if isinstance(out, (list, tuple)):
            return [x for x in out]
        return None

    def _extract_via_hooks(self, batch: torch.Tensor, layer_indices: Sequence[int]) -> list[torch.Tensor]:
        if not hasattr(self.model, "blocks"):
            raise ValueError("Hook fallback currently expects transformer models with .blocks")

        cache: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        try:
            for idx in layer_indices:
                module = self.model.blocks[idx]
                handle = module.register_forward_hook(
                    lambda _m, _inp, out, i=idx: cache.__setitem__(i, out.detach())
                )
                handles.append(handle)
            _ = self.model(batch)
        finally:
            for handle in handles:
                handle.remove()

        return [cache[i] for i in layer_indices]

    def extract(
        self,
        images: Sequence[Image.Image],
        layer_indices: Sequence[int] | None = None,
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not images:
            raise ValueError("images must be non-empty")

        images_rgb = _as_rgb_images(images)
        default_indices = self.list_default_layer_indices()
        if hasattr(self.model, "blocks"):
            total_layers = len(self.model.blocks)
            indices = _resolve_layer_indices(total_layers, layer_indices or default_indices)
        else:
            indices = list(layer_indices or default_indices)

        layer_names = [f"layer_{idx}" for idx in indices]
        collected = {name: [] for name in layer_names}
        pooling = self.pooling
        if pooling not in {"auto", "cls", "mean_tokens"}:
            raise ValueError("pooling must be one of: auto, cls, mean_tokens")

        with torch.no_grad():
            for start in range(0, len(images_rgb), batch_size):
                batch_imgs = images_rgb[start : start + batch_size]
                batch = torch.stack([self.transform(img) for img in batch_imgs], dim=0).to(self._device)

                inter = self._extract_intermediates(batch, indices)
                if inter is None:
                    inter = self._extract_via_hooks(batch, indices)

                for name, act in zip(layer_names, inter):
                    pooled = _pool_activation(act, pooling=pooling)
                    collected[name].append(pooled.detach().cpu().numpy().astype(np.float32))

        return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
