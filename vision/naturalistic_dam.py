from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np
from scipy.stats import spearmanr

from DAM import SilentDAM
from vision.run_naturalistic_baseline import _random_storage_splits
from vision.tasks import (
    build_generalization_task_human_similarity,
    build_identification_task,
)
from vision.vision_experiments import evaluate_layerwise_baseline, subset_feature_bundle


STAGE_A_CONFIGS: tuple[dict[str, float | int], ...] = tuple(
    {
        "n": n,
        "beta": beta,
        "alpha": alpha,
        "lmbda": lmbda,
        "steps_multiplier": steps_multiplier,
    }
    for n in (2, 4)
    for beta in (0.05, 0.1, 0.2)
    for alpha in (0.05, 0.1)
    for lmbda in (0.0, 0.05)
    for steps_multiplier in (1, 2)
)

STAGE_B_CONFIGS: tuple[dict[str, float | int], ...] = tuple(
    {
        "n": n,
        "beta": beta,
        "alpha": alpha,
        "lmbda": lmbda,
        "steps_multiplier": steps_multiplier,
    }
    for n in (2, 4, 6)
    for beta in (0.2, 0.5)
    for alpha in (0.05, 0.1)
    for lmbda in (0.0, 0.05, 0.1)
    for steps_multiplier in (2, 5)
    if not (
        n in (2, 4)
        and beta in (0.05, 0.1, 0.2)
        and lmbda in (0.0, 0.05)
    )
)


@dataclass(frozen=True)
class AnchorSpec:
    anchor_kind: str
    layer: str


@dataclass(frozen=True)
class BranchPair:
    category: str
    pooling: str
    anchor_kind: str
    vit_layer: str
    clip_layer: str


@dataclass(frozen=True)
class FrontierBranchSpec:
    category: str
    model_name: str
    pooling: str
    layer: str
    split_mode: str
    storage_sizes: tuple[int, ...] | None
    n_seeds: int


def get_dam_model_specs(include_clip: bool = True) -> tuple[tuple[str, str], ...]:
    base = (
        ("vit_base_patch16_224", "cls"),
        ("vit_base_patch16_224", "mean_tokens"),
    )
    if not include_clip:
        return base
    return base + (
        ("vit_base_patch16_clip_224.openai", "cls"),
        ("vit_base_patch16_clip_224.openai", "mean_tokens"),
    )


def load_baseline_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_anchor_layers(
    baseline_rows: list[dict[str, str]],
    model_specs: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], list[AnchorSpec]]:
    anchors: dict[tuple[str, str], list[AnchorSpec]] = {}
    for model_name, pooling in model_specs:
        subset = [
            row
            for row in baseline_rows
            if row["model_name"] == model_name and row["pooling"] == pooling
        ]
        if not subset:
            continue
        retrieval_row = max(subset, key=lambda row: float(row["gen_accuracy"]))
        alignment_row = max(subset, key=lambda row: float(row["human_rdm_spearman"]))
        anchor_list = [AnchorSpec(anchor_kind="retrieval", layer=retrieval_row["layer"])]
        if alignment_row["layer"] != retrieval_row["layer"]:
            anchor_list.append(AnchorSpec(anchor_kind="alignment", layer=alignment_row["layer"]))
        anchors[(model_name, pooling)] = anchor_list
    return anchors


def build_category_split_specs(
    *,
    category: str,
    concepts: tuple[str, ...],
    n_items: int,
    split_mode: str,
    storage_sizes: list[int] | None,
    n_seeds: int,
    seed: int,
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    if split_mode == "balanced_exemplar_folds":
        from vision.io_naturalistic import build_leave_one_exemplar_out_folds

        return [
            (f"fold_{fold_index}", fold_index, stored_indices, probe_indices)
            for fold_index, (stored_indices, probe_indices) in enumerate(
                build_leave_one_exemplar_out_folds(concepts)
            )
        ]
    if split_mode == "random_storage_curve":
        return _random_storage_splits(
            n_items,
            storage_sizes=storage_sizes or [40, 80],
            n_seeds=n_seeds,
            seed=seed,
        )
    raise ValueError(f"Unknown split mode for {category}: {split_mode}")


def should_trigger_stage_b(rows: list[dict[str, object]]) -> bool:
    return not any(bool(row.get("qualifying_win", False)) for row in rows)


def select_stage_b_focus_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    chosen: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    categories = sorted({str(row["category"]) for row in rows})
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        top_retrieval = sorted(
            subset,
            key=lambda row: float(row["gen_accuracy_baseline"]),
            reverse=True,
        )[:2]
        for row in top_retrieval:
            key = (str(row["category"]), str(row["model_name"]), str(row["pooling"]))
            if key not in seen:
                chosen.append(row)
                seen.add(key)
        clip_rows = [
            row
            for row in subset
            if "clip" in str(row["model_name"]).lower()
        ]
        if clip_rows:
            best_clip = max(clip_rows, key=lambda row: float(row["human_rdm_spearman_baseline"]))
            key = (str(best_clip["category"]), str(best_clip["model_name"]), str(best_clip["pooling"]))
            if key not in seen:
                chosen.append(best_clip)
                seen.add(key)
    return chosen


def _feature_rdm(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    normalized = features / norms
    cosine_sim = normalized @ normalized.T
    return 1.0 - cosine_sim


def probe_rdm_spearman(features: np.ndarray, similarity_matrix: np.ndarray, probe_indices: np.ndarray) -> float:
    probe_indices = np.asarray(probe_indices, dtype=np.int64)
    if probe_indices.size < 2:
        return float("nan")
    feature_rdm = _feature_rdm(np.asarray(features, dtype=np.float64))
    human_rdm = 1.0 - np.asarray(similarity_matrix, dtype=np.float64)[np.ix_(probe_indices, probe_indices)]
    tri = np.triu_indices_from(feature_rdm, k=1)
    corr = spearmanr(feature_rdm[tri], human_rdm[tri]).correlation
    return float(corr)


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    a_norm = a_arr / (np.linalg.norm(a_arr, axis=1, keepdims=True) + 1e-12)
    b_norm = b_arr / (np.linalg.norm(b_arr, axis=1, keepdims=True) + 1e-12)
    return np.sum(a_norm * b_norm, axis=1)


def _evaluate_dam_on_bundle(
    stored_features: np.ndarray,
    probe_features: np.ndarray,
    ground_truth_idx: np.ndarray,
    *,
    config: dict[str, float | int],
    seed: int,
    backend: str = "numba",
    decision_noise_std: float = 0.0,
    decision_noise_seed: int | None = None,
) -> dict[str, object]:
    stored = np.asarray(stored_features, dtype=np.float64)
    probes = np.asarray(probe_features, dtype=np.float64)
    targets = np.asarray(ground_truth_idx, dtype=np.int64)
    center = stored.mean(axis=0, keepdims=True)
    stored_dam = stored - center
    stored_dam = stored_dam / (np.linalg.norm(stored_dam, axis=1, keepdims=True) + 1e-12)
    probes_dam = probes - center
    probes_dam = probes_dam / (np.linalg.norm(probes_dam, axis=1, keepdims=True) + 1e-12)
    dim = int(stored.shape[1])
    steps = int(config["steps_multiplier"]) * dim
    dam = SilentDAM(
        stored_dam,
        n=int(config["n"]),
        beta=float(config["beta"]),
        alpha=float(config["alpha"]),
        lmbda=float(config["lmbda"]),
    )

    winners: list[int] = []
    retrieved_states: list[np.ndarray] = []
    for idx, probe in enumerate(probes_dam):
        rng = np.random.default_rng(seed + idx)
        update_indices = rng.integers(0, dim, size=steps, dtype=np.int64)
        retrieved, _, _, _, best_idx = dam.retrieve_differential(
            probe,
            steps=steps,
            update_indices=update_indices,
            trace_every=0,
            backend=backend,
            return_best_idx=True,
        )
        winners.append(int(best_idx))
        retrieved_states.append(np.asarray(retrieved, dtype=np.float32))

    retrieved_arr = np.asarray(retrieved_states, dtype=np.float32)
    scored = evaluate_layerwise_baseline(
        {"layer": stored_dam.astype(np.float32)},
        {"layer": retrieved_arr},
        targets,
        metric="cosine",
        decision_noise_std=decision_noise_std,
        decision_noise_seed=decision_noise_seed,
    )["layer"]
    scored["winners"] = winners
    scored["retrieved_states"] = retrieved_arr
    return scored


def evaluate_dam_trial(
    *,
    layer: str,
    stored_feature_bundle: dict[str, np.ndarray],
    clean_probe_bundle: dict[str, np.ndarray],
    noisy_probe_bundle: dict[str, np.ndarray],
    gen_probe_bundle: dict[str, np.ndarray],
    ident_ground_truth_idx: np.ndarray,
    gen_ground_truth_idx: np.ndarray,
    similarity_matrix: np.ndarray,
    gen_probe_indices: np.ndarray,
    gen_stored_indices: np.ndarray,
    concepts: tuple[str, ...],
    config: dict[str, float | int],
    seed: int,
    decision_noise_std: float = 0.0,
) -> dict[str, object]:
    stored = np.asarray(stored_feature_bundle[layer], dtype=np.float32)
    clean_probe = np.asarray(clean_probe_bundle[layer], dtype=np.float32)
    noisy_probe = np.asarray(noisy_probe_bundle[layer], dtype=np.float32)
    gen_probe = np.asarray(gen_probe_bundle[layer], dtype=np.float32)

    exact = _evaluate_dam_on_bundle(
        stored,
        stored,
        np.arange(stored.shape[0], dtype=np.int64),
        config=config,
        seed=seed,
    )
    clean = _evaluate_dam_on_bundle(
        stored,
        clean_probe,
        ident_ground_truth_idx,
        config=config,
        seed=seed + 1000,
    )
    ident = _evaluate_dam_on_bundle(
        stored,
        noisy_probe,
        ident_ground_truth_idx,
        config=config,
        seed=seed + 2000,
        decision_noise_std=decision_noise_std,
        decision_noise_seed=seed + 3000,
        )
    gen = _evaluate_dam_on_bundle(
        stored,
        gen_probe,
        gen_ground_truth_idx,
        config=config,
        seed=seed + 5000,
        decision_noise_std=decision_noise_std,
        decision_noise_seed=seed + 6000,
    )

    center = stored.mean(axis=0, keepdims=True)
    stored_dam = stored - center
    stored_dam = stored_dam / (np.linalg.norm(stored_dam, axis=1, keepdims=True) + 1e-12)
    gen_probe_dam = gen_probe - center
    gen_probe_dam = gen_probe_dam / (np.linalg.norm(gen_probe_dam, axis=1, keepdims=True) + 1e-12)
    target_states = stored_dam[np.asarray(gen_ground_truth_idx, dtype=np.int64)]
    cue_recovered_cos = rowwise_cosine(gen_probe_dam, gen["retrieved_states"])
    target_pull_gain = rowwise_cosine(gen["retrieved_states"], target_states) - rowwise_cosine(gen_probe_dam, target_states)

    probe_rdm_pre = probe_rdm_spearman(gen_probe, similarity_matrix, gen_probe_indices)
    probe_rdm_post = probe_rdm_spearman(gen["retrieved_states"], similarity_matrix, gen_probe_indices)

    winners = np.asarray(gen["winners"], dtype=np.int64)
    stored_global = np.asarray(gen_stored_indices, dtype=np.int64)
    probe_global = np.asarray(gen_probe_indices, dtype=np.int64)
    retrieved_global = stored_global[winners]
    target_global = stored_global[np.asarray(gen_ground_truth_idx, dtype=np.int64)]

    retrieved_sims = np.asarray(
        [similarity_matrix[int(probe), int(ret)] for probe, ret in zip(probe_global, retrieved_global)],
        dtype=np.float64,
    )
    target_sims = np.asarray(
        [similarity_matrix[int(probe), int(tgt)] for probe, tgt in zip(probe_global, target_global)],
        dtype=np.float64,
    )
    same_concept = np.asarray(
        [concepts[int(probe)] == concepts[int(ret)] for probe, ret in zip(probe_global, retrieved_global)],
        dtype=np.float64,
    )
    has_meaningful_concepts = len(set(concepts)) < len(concepts)

    return {
        "layer": layer,
        "exact_accuracy_dam": float(exact["accuracy"]),
        "clean_reextract_accuracy_dam": float(clean["accuracy"]),
        "exact_avg_target_sim_dam": float(exact["avg_target_sim"]),
        "clean_reextract_avg_target_sim_dam": float(clean["avg_target_sim"]),
        "ident_accuracy_dam": float(ident["accuracy"]),
        "ident_avg_target_sim_dam": float(ident["avg_target_sim"]),
        "ident_avg_margin_dam": float(ident["avg_margin"]),
        "gen_accuracy_dam": float(gen["accuracy"]),
        "gen_avg_target_sim_dam": float(gen["avg_target_sim"]),
        "gen_avg_margin_dam": float(gen["avg_margin"]),
        "gen_avg_retrieved_human_similarity_dam": float(np.mean(retrieved_sims)),
        "gen_avg_human_similarity_regret_dam": float(np.mean(target_sims - retrieved_sims)),
        "gen_same_concept_accuracy_dam": float(np.mean(same_concept) * 100.0) if has_meaningful_concepts else float("nan"),
        "probe_rdm_spearman_pre": probe_rdm_pre,
        "probe_rdm_spearman_post": probe_rdm_post,
        "probe_rdm_spearman_delta": float(probe_rdm_post - probe_rdm_pre),
        "cue_recovered_cosine_mean": float(np.mean(cue_recovered_cos)),
        "cue_recovered_cosine_std": float(np.std(cue_recovered_cos)),
        "cue_displacement_mean": float(np.mean(1.0 - cue_recovered_cos)),
        "target_pull_gain_mean": float(np.mean(target_pull_gain)),
        "target_pull_gain_std": float(np.std(target_pull_gain)),
        "gen_winners_dam": list(map(int, winners.tolist())),
    }


def summarize_dam_cross_category(output_dir: Path, categories: list[str]) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for category in categories:
        csv_path = output_dir / f"{category}_combined.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    summary = {
        "categories": categories,
        "best_qualifying_win": {},
        "best_retrieval_gain": {},
        "best_alignment_preserving_gain": {},
    }
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        if not subset:
            continue
        wins = [row for row in subset if str(row["qualifying_win"]).lower() == "true"]
        summary["best_qualifying_win"][category] = (
            max(wins, key=lambda row: float(row["gen_accuracy_delta"])) if wins else None
        )
        summary["best_retrieval_gain"][category] = max(
            subset,
            key=lambda row: float(row["gen_accuracy_delta"]),
        )
        summary["best_alignment_preserving_gain"][category] = max(
            subset,
            key=lambda row: (
                float(row["gen_accuracy_delta"]) - max(0.0, -float(row["probe_rdm_spearman_delta"]))
            ),
        )
    comparison_name = "_vs_".join(categories)
    with (output_dir / f"{comparison_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def classify_comparison_outcome(vit_row: dict[str, str] | None, clip_row: dict[str, str] | None) -> str:
    vit_clean = bool(vit_row) and str(vit_row["qualifying_win"]).lower() == "true"
    clip_clean = bool(clip_row) and str(clip_row["qualifying_win"]).lower() == "true"
    if vit_clean and clip_clean:
        return "both_clean_wins"
    if vit_clean:
        return "vit_clean_win"
    if clip_clean:
        return "clip_clean_win"

    vit_retrieval = bool(vit_row) and float(vit_row["gen_accuracy_delta"]) > 0.0
    clip_retrieval = bool(clip_row) and float(clip_row["gen_accuracy_delta"]) > 0.0
    vit_alignment = bool(vit_row) and float(vit_row["probe_rdm_spearman_delta"]) > 0.0
    clip_alignment = bool(clip_row) and float(clip_row["probe_rdm_spearman_delta"]) > 0.0
    if vit_retrieval or clip_retrieval:
        if vit_alignment or clip_alignment:
            return "retrieval_and_alignment_gain"
        return "retrieval_only_gain"
    if vit_alignment or clip_alignment:
        return "alignment_only_gain"
    return "no_useful_gain"


def select_best_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    clean_rows = [row for row in rows if str(row["qualifying_win"]).lower() == "true"]
    if clean_rows:
        return max(
            clean_rows,
            key=lambda row: (
                float(row["gen_accuracy_delta"]),
                float(row["gen_avg_margin_delta"]),
                float(row["probe_rdm_spearman_delta"]),
            ),
        )
    return max(
        rows,
        key=lambda row: (
            float(row["gen_accuracy_delta"]),
            float(row["probe_rdm_spearman_delta"]),
            -float(row["gen_avg_human_similarity_regret_delta"]),
        ),
    )


def build_branch_pairs(rows: list[dict[str, str]]) -> list[BranchPair]:
    grouped: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    for row in rows:
        model_name = row["model_name"]
        if model_name not in {"vit_base_patch16_224", "vit_base_patch16_clip_224.openai"}:
            continue
        key = (row["category"], row["pooling"], row["anchor_kind"])
        grouped.setdefault(key, {}).setdefault(model_name, set()).add(row["layer"])

    pairs: list[BranchPair] = []
    for (category, pooling, anchor_kind), models in sorted(grouped.items()):
        vit_layers = sorted(models.get("vit_base_patch16_224", set()))
        clip_layers = sorted(models.get("vit_base_patch16_clip_224.openai", set()))
        if not vit_layers or not clip_layers:
            continue
        vit_layer = vit_layers[0]
        clip_layer = clip_layers[0]
        if anchor_kind == "retrieval":
            vit_layer = vit_layers[-1]
            clip_layer = clip_layers[-1]
        pairs.append(
            BranchPair(
                category=category,
                pooling=pooling,
                anchor_kind=anchor_kind,
                vit_layer=vit_layer,
                clip_layer=clip_layer,
            )
        )
    return pairs


def _row_matches_branch(row: dict[str, str], *, model_name: str, pair: BranchPair) -> bool:
    layer = pair.vit_layer if model_name == "vit_base_patch16_224" else pair.clip_layer
    return (
        row["category"] == pair.category
        and row["model_name"] == model_name
        and row["pooling"] == pair.pooling
        and row["anchor_kind"] == pair.anchor_kind
        and row["layer"] == layer
    )


def summarize_encoder_head_to_head(output_dir: Path, categories: list[str]) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for category in categories:
        csv_path = output_dir / f"{category}_combined.csv"
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))

    pairs = build_branch_pairs(rows)
    comparison_rows: list[dict[str, object]] = []
    frontier_rows: list[dict[str, object]] = []

    for pair in pairs:
        vit_candidates = [row for row in rows if _row_matches_branch(row, model_name="vit_base_patch16_224", pair=pair)]
        clip_candidates = [row for row in rows if _row_matches_branch(row, model_name="vit_base_patch16_clip_224.openai", pair=pair)]
        vit_best = select_best_row(vit_candidates)
        clip_best = select_best_row(clip_candidates)
        outcome = classify_comparison_outcome(vit_best, clip_best)

        comparison_rows.append(
            {
                "category": pair.category,
                "pooling": pair.pooling,
                "anchor_kind": pair.anchor_kind,
                "vit_layer": pair.vit_layer,
                "clip_layer": pair.clip_layer,
                "outcome": outcome,
                "vit_gen_accuracy_delta": float(vit_best["gen_accuracy_delta"]) if vit_best else float("nan"),
                "clip_gen_accuracy_delta": float(clip_best["gen_accuracy_delta"]) if clip_best else float("nan"),
                "vit_probe_rdm_delta": float(vit_best["probe_rdm_spearman_delta"]) if vit_best else float("nan"),
                "clip_probe_rdm_delta": float(clip_best["probe_rdm_spearman_delta"]) if clip_best else float("nan"),
                "vit_best_config": (
                    f"n={vit_best['dam_n']},beta={vit_best['dam_beta']},alpha={vit_best['dam_alpha']},steps={vit_best['dam_steps_multiplier']}"
                    if vit_best
                    else ""
                ),
                "clip_best_config": (
                    f"n={clip_best['dam_n']},beta={clip_best['dam_beta']},alpha={clip_best['dam_alpha']},steps={clip_best['dam_steps_multiplier']}"
                    if clip_best
                    else ""
                ),
            }
        )

        for encoder_name, best in (("vit", vit_best), ("clip", clip_best)):
            if best is None:
                continue
            frontier_rows.append(
                {
                    "category": pair.category,
                    "pooling": pair.pooling,
                    "anchor_kind": pair.anchor_kind,
                    "encoder": encoder_name,
                    "model_name": best["model_name"],
                    "layer": best["layer"],
                    "qualifying_win": str(best["qualifying_win"]).lower() == "true",
                    "gen_accuracy_delta": float(best["gen_accuracy_delta"]),
                    "gen_avg_margin_delta": float(best["gen_avg_margin_delta"]),
                    "gen_avg_human_similarity_regret_delta": float(best["gen_avg_human_similarity_regret_delta"]),
                    "probe_rdm_spearman_delta": float(best["probe_rdm_spearman_delta"]),
                }
            )

    summary = {
        "categories": categories,
        "branch_pairs": [pair.__dict__ for pair in pairs],
        "comparison_rows": comparison_rows,
        "frontier_rows": frontier_rows,
    }
    csv_path = output_dir / "vit_vs_clip_head_to_head.csv"
    if comparison_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)
    frontier_csv = output_dir / "vit_vs_clip_frontier.csv"
    if frontier_rows:
        with frontier_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(frontier_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frontier_rows)
    with (output_dir / "vit_vs_clip_head_to_head.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def get_animals_frontier_configs() -> tuple[dict[str, float | int], ...]:
    return tuple(
        {
            "n": n,
            "beta": beta,
            "alpha": alpha,
            "lmbda": 0.0,
            "steps_multiplier": steps_multiplier,
        }
        for n in (2, 4)
        for beta in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
        for alpha in (0.02, 0.05, 0.1)
        for steps_multiplier in (1, 2, 3, 5, 8)
    )


def get_hard_ident_tradeoff_configs() -> tuple[dict[str, float | int], ...]:
    return tuple(
        {
            "n": n,
            "beta": beta,
            "alpha": alpha,
            "lmbda": lmbda,
            "steps_multiplier": steps_multiplier,
        }
        for n in (2, 4)
        for beta in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
        for alpha in (0.02, 0.05, 0.1)
        for lmbda in (0.0, 0.05)
        for steps_multiplier in (1, 2, 3, 5, 8)
    )


def get_frontier_branch_specs() -> tuple[FrontierBranchSpec, ...]:
    return (
        FrontierBranchSpec(
            category="animals",
            model_name="vit_base_patch16_224",
            pooling="cls",
            layer="layer_11",
            split_mode="random_storage_curve",
            storage_sizes=(40, 80),
            n_seeds=5,
        ),
        FrontierBranchSpec(
            category="fruits",
            model_name="vit_base_patch16_224",
            pooling="cls",
            layer="layer_11",
            split_mode="balanced_exemplar_folds",
            storage_sizes=None,
            n_seeds=3,
        ),
    )


def pareto_frontier_rows(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    z_key: str | None = None,
    maximize_z: bool = False,
    filters: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    filtered = rows
    if filters:
        filtered = [
            row
            for row in rows
            if all(row.get(key) == value for key, value in filters.items())
        ]
    frontier: list[dict[str, object]] = []
    for row in filtered:
        dominated = False
        x = float(row[x_key])
        y = float(row[y_key])
        z = float(row[z_key]) if z_key is not None else 0.0
        for other in filtered:
            if other is row:
                continue
            ox = float(other[x_key])
            oy = float(other[y_key])
            oz = float(other[z_key]) if z_key is not None else 0.0
            z_better = oz >= z if maximize_z else oz <= z
            z_strict = oz > z if maximize_z else oz < z
            if (
                ox >= x
                and oy >= y
                and z_better
                and (ox > x or oy > y or (z_key is not None and z_strict))
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def classify_frontier_shape(rows: list[dict[str, object]]) -> str:
    clean = [row for row in rows if bool(row.get("qualifying_win", False))]
    if not clean:
        return "no_useful_dam_regime"

    aligned = [row for row in clean if float(row["probe_rdm_spearman_delta"]) >= 0.0]
    if not aligned:
        return "hard_tradeoff"

    best_retrieval = max(clean, key=lambda row: float(row["gen_accuracy_delta"]))
    if float(best_retrieval["probe_rdm_spearman_delta"]) >= 0.0:
        if all(float(row["probe_rdm_spearman_delta"]) >= 0.0 for row in aligned):
            return "no_tradeoff"
        return "soft_tradeoff"
    return "hard_tradeoff"


def summarize_frontier_runs(output_dir: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"branches": {}}
    branch_keys = sorted(
        {
            (
                str(row["category"]),
                str(row["model_name"]),
                str(row["pooling"]),
                str(row["layer"]),
            )
            for row in rows
        }
    )
    for category, model_name, pooling, layer in branch_keys:
        branch_rows = [
            row
            for row in rows
            if row["category"] == category
            and row["model_name"] == model_name
            and row["pooling"] == pooling
            and row["layer"] == layer
        ]
        key = f"{category}::{model_name}::{pooling}::{layer}"
        clean_frontier = pareto_frontier_rows(
            branch_rows,
            x_key="gen_accuracy_delta",
            y_key="probe_rdm_spearman_delta",
            z_key="gen_avg_human_similarity_regret_delta",
            maximize_z=False,
            filters={"qualifying_win": True},
        )
        retrieval_frontier = pareto_frontier_rows(
            branch_rows,
            x_key="gen_accuracy_delta",
            y_key="gen_avg_margin_delta",
            z_key="gen_avg_human_similarity_regret_delta",
            maximize_z=False,
        )
        alignment_frontier = pareto_frontier_rows(
            branch_rows,
            x_key="gen_accuracy_delta",
            y_key="probe_rdm_spearman_delta",
            z_key="gen_avg_human_similarity_regret_delta",
            maximize_z=False,
        )
        summary["branches"][key] = {
            "classification": classify_frontier_shape(branch_rows),
            "n_rows": len(branch_rows),
            "n_qualifying_wins": int(sum(bool(row["qualifying_win"]) for row in branch_rows)),
            "best_clean_win": select_best_row(
                [
                    {
                        **row,
                        "qualifying_win": "True" if bool(row["qualifying_win"]) else "False",
                    }
                    for row in branch_rows
                    if bool(row["qualifying_win"])
                ]
            ),
            "clean_frontier": clean_frontier,
            "retrieval_frontier": retrieval_frontier,
            "alignment_frontier": alignment_frontier,
        }
    with (output_dir / "frontier_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
