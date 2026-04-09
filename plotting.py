"""
Reusable matplotlib helpers for DenseAM / multiscale experiment results.

Result schemas (nested dicts):
  ident_results[c][n][K] -> {'accuracy', 'avg_sim', 'std_sim'}
  gen_results[c][n][K]  -> same keys

`c` is typically a scale factor (float); `n` is DAM order; `K` is pattern count.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import BASE_FREQ, N_MODULES

# Type aliases for nested experiment dicts
_Cell = Mapping[str, float]
_RowN = MutableMapping[int, MutableMapping[int, _Cell]]
_ScaleResults = MutableMapping[Any, _RowN]


def summarize_multiscale_tradeoff_table(
    ident_results: _ScaleResults,
    gen_results: _ScaleResults,
    scale_factors: Sequence[Any],
    n_values: Sequence[int],
    K_values: Sequence[int],
    *,
    base_freq: float = BASE_FREQ,
    n_modules: int = N_MODULES,
    diff_threshold: float = 10.0,
) -> Dict[str, Any]:
    """
    Build structured tables from multiscale identification / generalization results
    (no plotting, no printing).

    Returns
    -------
    dict with:
      - ``rows``: one entry per (c, n, K) configuration
      - ``averages_by_cn``: mean accuracy over K for each (n, c)
      - ``trends``: compare first vs last c in ``scale_factors`` for each n
    """
    rows: List[Dict[str, Any]] = []
    for n in n_values:
        for c in scale_factors:
            freqs = base_freq * (float(c) ** np.arange(n_modules))
            freq_str = "[" + ", ".join(f"{f:.2f}" for f in freqs) + "]"
            for K in K_values:
                ident_acc = float(ident_results[c][n][K]["accuracy"])
                gen_acc = float(gen_results[c][n][K]["accuracy"])
                diff = ident_acc - gen_acc
                if diff > diff_threshold:
                    tradeoff = "Ident+"
                elif diff < -diff_threshold:
                    tradeoff = "Gen+"
                else:
                    tradeoff = "Balanced"
                rows.append(
                    {
                        "scale_c": float(c),
                        "n": int(n),
                        "K": int(K),
                        "frequencies": freqs.tolist(),
                        "frequencies_str": freq_str,
                        "ident_accuracy": ident_acc,
                        "gen_accuracy": gen_acc,
                        "diff_ident_minus_gen": diff,
                        "tradeoff_label": tradeoff,
                    }
                )

    averages_by_cn: List[Dict[str, Any]] = []
    for n in n_values:
        for c in scale_factors:
            avg_ident = float(
                np.mean([ident_results[c][n][K]["accuracy"] for K in K_values])
            )
            avg_gen = float(
                np.mean([gen_results[c][n][K]["accuracy"] for K in K_values])
            )
            averages_by_cn.append(
                {
                    "n": int(n),
                    "scale_c": float(c),
                    "mean_ident_accuracy": avg_ident,
                    "mean_gen_accuracy": avg_gen,
                    "diff_ident_minus_gen": avg_ident - avg_gen,
                }
            )

    trends: List[Dict[str, Any]] = []
    if len(scale_factors) >= 2:
        small_c = scale_factors[0]
        large_c = scale_factors[-1]
        for n in n_values:
            small_ident = float(
                np.mean([ident_results[small_c][n][K]["accuracy"] for K in K_values])
            )
            large_ident = float(
                np.mean([ident_results[large_c][n][K]["accuracy"] for K in K_values])
            )
            small_gen = float(
                np.mean([gen_results[small_c][n][K]["accuracy"] for K in K_values])
            )
            large_gen = float(
                np.mean([gen_results[large_c][n][K]["accuracy"] for K in K_values])
            )
            ident_trend = "increases" if large_ident > small_ident else "decreases"
            gen_trend = "increases" if large_gen > small_gen else "decreases"
            if ident_trend == "increases" and gen_trend == "decreases":
                verdict = "tradeoff_ident_up_gen_down"
            elif ident_trend == "decreases" and gen_trend == "increases":
                verdict = "tradeoff_ident_down_gen_up"
            else:
                verdict = "no_clear_tradeoff"
            trends.append(
                {
                    "n": int(n),
                    "c_small": float(small_c),
                    "c_large": float(large_c),
                    "ident_small": small_ident,
                    "ident_large": large_ident,
                    "gen_small": small_gen,
                    "gen_large": large_gen,
                    "ident_trend": ident_trend,
                    "gen_trend": gen_trend,
                    "verdict": verdict,
                }
            )

    return {"rows": rows, "averages_by_cn": averages_by_cn, "trends": trends}


def plot_tradeoff_breakit(
    ident_results: _ScaleResults,
    gen_results: _ScaleResults,
    scale_factors: Sequence[Any],
    n_values: Sequence[int],
    K_values: Sequence[int],
    title_suffix: str = "",
):
    """
    Two subplots: (1) Gen vs Ident averaged over K for each n; (2) same for n=4 only
    when n=4 is in ``n_values`` (matches DenseAM Part 4b notebook).
    """
    fig, (ax_all, ax_n4) = plt.subplots(1, 2, figsize=(12, 5))
    colors_n = plt.cm.Set1(np.linspace(0, 1, max(len(n_values), 2)))
    for ni, n in enumerate(n_values):
        x_gen = [
            float(np.mean([gen_results[c][n][K]["accuracy"] for K in K_values]))
            for c in scale_factors
        ]
        y_ident = [
            float(np.mean([ident_results[c][n][K]["accuracy"] for K in K_values]))
            for c in scale_factors
        ]
        ax_all.scatter(x_gen, y_ident, color=colors_n[ni], s=60, label=f"n={n}", alpha=0.8)
        for i, c in enumerate(scale_factors):
            ax_all.annotate(f"{float(c):.2f}", (x_gen[i], y_ident[i]), fontsize=6, alpha=0.6)
    ax_all.set_xlabel("Generalization Accuracy (%)")
    ax_all.set_ylabel("Identification Accuracy (%)")
    ax_all.set_title("Tradeoff (all n)" + title_suffix)
    ax_all.legend()
    ax_all.grid(True, alpha=0.3)

    n4 = 4
    if n4 in n_values:
        colors_c = plt.cm.viridis(np.linspace(0, 1, len(scale_factors)))
        x_gen_n4 = [
            float(np.mean([gen_results[c][n4][K]["accuracy"] for K in K_values]))
            for c in scale_factors
        ]
        y_ident_n4 = [
            float(np.mean([ident_results[c][n4][K]["accuracy"] for K in K_values]))
            for c in scale_factors
        ]
        ax_n4.scatter(x_gen_n4, y_ident_n4, c=colors_c, s=80)
        for i, c in enumerate(scale_factors):
            ax_n4.annotate(f"c={float(c):.2f}", (x_gen_n4[i], y_ident_n4[i]), fontsize=7)
        ax_n4.set_xlabel("Generalization Accuracy (%)")
        ax_n4.set_ylabel("Identification Accuracy (%)")
        ax_n4.set_title("Tradeoff (n=4 only)" + title_suffix)
        ax_n4.grid(True, alpha=0.3)
    else:
        ax_n4.set_visible(False)

    plt.tight_layout()
    return fig


def plot_pattern_correlations(patterns: np.ndarray):
    """
    Heatmap of pairwise cosine similarity between stored patterns (shape (K, N)).
    """
    patterns = np.asarray(patterns, dtype=float)
    K = patterns.shape[0]
    norms = np.linalg.norm(patterns, axis=1, keepdims=True) + 1e-12
    similarity_matrix = (patterns @ patterns.T) / (norms @ norms.T)

    annot = K <= 24
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        similarity_matrix,
        annot=annot,
        cmap="RdBu_r",
        center=0,
        xticklabels=[f"P{i}" for i in range(K)],
        yticklabels=[f"P{i}" for i in range(K)],
        ax=ax,
    )
    ax.set_title("Pattern Similarity Matrix (Cosine)")
    ax.set_xlabel("Pattern Index")
    ax.set_ylabel("Pattern Index")
    plt.tight_layout()
    return fig


def plot_dam_retrieval_traces(
    energy_trace: Sequence[float],
    similarity_trace: Union[Sequence[Sequence[float]], np.ndarray],
    target_pattern_idx: int = 0,
):
    """
    Plot energy vs iteration and per-pattern similarity traces from DAM retrieval.

    ``similarity_trace`` is a time series of length-K cosine (or dot-normalized) values,
    e.g. list of lists from ``retrieve`` / ``retrieve_differential`` (sampled every 10 steps).
    """
    energy_trace = np.asarray(energy_trace, dtype=float)
    sim_arr = np.asarray(similarity_trace, dtype=float)
    if sim_arr.ndim != 2:
        raise ValueError("similarity_trace must be 2-D (time steps x patterns)")
    K = sim_arr.shape[1]
    if not (0 <= target_pattern_idx < K):
        raise ValueError("target_pattern_idx out of range")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(energy_trace)
    ax[0].set_title("Energy Minimization")
    ax[0].set_xlabel("Iterations (x10)")
    ax[0].set_ylabel("Energy")

    for k in range(K):
        label = f"Pattern {k}" + (" (Target)" if k == target_pattern_idx else "")
        ax[1].plot(
            sim_arr[:, k],
            label=label,
            alpha=0.7 if k == target_pattern_idx else 0.3,
        )
    ax[1].set_title("Similarity to Stored Patterns")
    ax[1].set_xlabel("Iterations (x10)")
    ax[1].legend()
    plt.tight_layout()
    return fig
