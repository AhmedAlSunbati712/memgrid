from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LayerwisePreprocessor:
    use_zscore: bool = True
    l2_normalize: bool = True
    eps: float = 1e-8
    means_: dict[str, np.ndarray] = field(default_factory=dict)
    stds_: dict[str, np.ndarray] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, layer_to_features: dict[str, np.ndarray]) -> "LayerwisePreprocessor":
        if not layer_to_features:
            raise ValueError("layer_to_features must be non-empty")

        self.means_.clear()
        self.stds_.clear()

        for layer, feats in layer_to_features.items():
            x = np.asarray(feats, dtype=np.float32)
            if x.ndim != 2:
                raise ValueError(f"Expected 2D feature matrix for {layer}, got shape {x.shape}")
            mean = x.mean(axis=0, keepdims=True).astype(np.float32)
            std = x.std(axis=0, keepdims=True).astype(np.float32)
            std = np.maximum(std, self.eps)
            self.means_[layer] = mean
            self.stds_[layer] = std

        self.fitted_ = True
        return self

    def transform(self, layer_to_features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("LayerwisePreprocessor must be fit before transform.")

        out: dict[str, np.ndarray] = {}
        for layer, feats in layer_to_features.items():
            x = np.asarray(feats, dtype=np.float32)
            if x.ndim != 2:
                raise ValueError(f"Expected 2D feature matrix for {layer}, got shape {x.shape}")

            if self.use_zscore:
                if layer not in self.means_ or layer not in self.stds_:
                    raise KeyError(f"Layer {layer} was not seen during fit.")
                x = (x - self.means_[layer]) / self.stds_[layer]

            if self.l2_normalize:
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                norms = np.maximum(norms, self.eps)
                x = x / norms

            out[layer] = x.astype(np.float32, copy=False)
        return out

    def fit_transform(self, layer_to_features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.fit(layer_to_features).transform(layer_to_features)

    def to_state_dict(self) -> dict[str, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("LayerwisePreprocessor must be fit before export.")

        state: dict[str, np.ndarray] = {}
        for layer, mean in self.means_.items():
            state[f"mean_{layer}"] = np.asarray(mean, dtype=np.float32)
        for layer, std in self.stds_.items():
            state[f"std_{layer}"] = np.asarray(std, dtype=np.float32)
        return state

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, np.ndarray],
        *,
        use_zscore: bool = True,
        l2_normalize: bool = True,
        eps: float = 1e-8,
    ) -> "LayerwisePreprocessor":
        prep = cls(use_zscore=use_zscore, l2_normalize=l2_normalize, eps=eps)
        means = {
            key.removeprefix("mean_"): np.asarray(value, dtype=np.float32)
            for key, value in state.items()
            if key.startswith("mean_")
        }
        stds = {
            key.removeprefix("std_"): np.asarray(value, dtype=np.float32)
            for key, value in state.items()
            if key.startswith("std_")
        }
        if set(means) != set(stds):
            raise ValueError("Preprocessor state must contain matching mean/std entries.")
        prep.means_ = means
        prep.stds_ = stds
        prep.fitted_ = True
        return prep
