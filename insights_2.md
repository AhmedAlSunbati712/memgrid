# Vision Layerwise Baseline: Notes

## Summary

The current milestone is a synthetic vision baseline, not a DAM retrieval
study. The implemented workflow generates simple colored-square stimuli,
extracts layerwise vision features, builds identification and generalization
tasks, and evaluates direct nearest-neighbor performance layer by layer using
`evaluate_layerwise_baseline(...)`.

This gives a first readout of what the raw feature space already supports
before adding any retrieval dynamics.

## What the baseline does

- Identification baseline:
  - stored items are selected from the synthetic stimulus set,
  - probes are perturbed versions of those same stored items,
  - correctness means retrieving the exact corresponding stored item.
- Generalization baseline:
  - probes are held-out items not present in the stored set,
  - correctness means retrieving the stored item nearest in synthetic stimulus
    space: weighted normalized RGB distance plus weighted normalized xy
    position distance.

The baseline uses direct nearest-neighbor lookup in feature space, with cosine
similarity as the default metric.

## Key functions

- `generate_square_stimuli(...)`:
  creates synthetic images and aligned metadata.
- `build_balanced_splits_by_color(...)`:
  chooses a color-balanced stored set and held-out probe set.
- `build_identification_task(...)`:
  creates the exact-match task targets.
- `build_generalization_task_synthetic(...)`:
  creates the held-out generalization targets using stimulus-space distance.
- `extract_feature_bundle(...)`:
  pulls per-layer feature matrices out of a cached `.npz` payload.
- `subset_feature_bundle(...)`:
  slices the same item indices from every layer.
- `evaluate_layerwise_baseline(...)`:
  performs direct nearest-neighbor evaluation for each layer.

## Current saved baseline readout

From the saved baseline summary before rerunning after the preprocessing and
diagnostic fixes:

- Identification baseline:
  - `layer_0`: `66.7%`
  - `layer_6`: `16.7%`
  - `layer_11`: `8.3%`
- Generalization baseline:
  - `layer_0`: `41.7%`
  - `layer_6`: `41.7%`
  - `layer_11`: `50.0%`

This means the baseline already shows non-trivial structure in the raw feature
space. Early layers are strongest for the current identification setup, while
generalization is more mixed across the selected layers.

These values came from `PREPROCESS_MODE = "zscore_l2"`. They should now be
treated as a diagnostic failure case, not the preferred baseline.

## Why identification similarity looked too low

The surprising pattern was:

- identification probes were noisy versions of stored images,
- generalization probes were clean held-out images,
- but identification `avg_sim` was much lower than generalization `avg_sim`.

The sanity checks isolate the cause. Exact stored-image probes and clean
re-extracted stored-image probes both produce `100%` accuracy and target cosine
near `1.0`. The failure appears when noisy identification probes are transformed
with z-score statistics fit on the clean synthetic set. This makes small
off-distribution perturbations dominate some feature dimensions and collapses
the target cosine.

With `raw` or `l2_only` preprocessing, noisy identification target cosine stays
high. Therefore the default baseline now uses `l2_only`, and `zscore_l2` should
be presented only as a preprocessing artifact or ablation. The standalone
feature-cache script also defaults to l2-only; pass `--zscore` only for an
explicit ablation.

## Sanity checks now in the notebook

The notebook now reports:

- exact-image sanity: the stored vectors are used as their own probes,
- clean re-extraction sanity: the same stored images are passed through the
  model again and compared to the stored vectors,
- baseline identification/generalization with target similarity and margin,
- one-factor ablations for mixed color-position, color-only, and position-only
  conditions.

The exact-image and clean re-extraction checks should be `100%` and near `1.0`
target similarity. If either fails, the issue is implementation/preprocessing,
not a scientific result.

## Variable glossary

- `stored_features`: layerwise vectors used as the memory bank.
- `probe_features`: layerwise vectors used as queries.
- `ground_truth_idx`: local stored-set index of the correct answer for each
  probe.
- `metric`: similarity rule used for nearest-neighbor lookup. Supported values
  are `cosine`, `dot`, and `euclidean`.
- `synthetic_metadata_distance`: target-definition distance for synthetic
  generalization: `color_weight * rgb_distance + position_weight *
  position_distance`.
- `rgb_distance`: Euclidean distance between `color_rgb` values, normalized so
  black-to-white is `1.0`.
- `position_distance`: Euclidean distance between square xy positions,
  normalized so opposite valid corners are `1.0`.
- `color_weight`: scales color distance. Set to `0.0` to ignore color.
- `position_weight`: scales position distance. Set to `0.0` to ignore position.
- `accuracy`: percent of probes whose nearest stored vector matches the correct
  target.
- `avg_sim`: average cosine similarity between the probe and the correct stored
- `avg_target_sim`: explicit name for average cosine similarity between the
  probe and the correct stored target.
- `avg_sim`: backward-compatible alias for `avg_target_sim`.
- `avg_best_wrong_sim`: average cosine similarity between the probe and the
  closest incorrect stored item.
- `avg_margin`: `avg_target_sim - avg_best_wrong_sim`; positive margins mean
  the correct target is separated from competitors on average.
- `std_sim`: standard deviation of those target similarities.
- `avg_probe_norm`: average norm of the probe vectors.
- `avg_target_norm`: average norm of the correct stored vectors.
- `PREPROCESS_MODE`: feature normalization choice. The current default is
  `l2_only`; `zscore_l2` is useful as a diagnostic ablation but distorted the
  noisy identification probes.
- `COLOR_ONLY_JITTER_STD`: per-sample RGB jitter used to make fixed-position
  color-only samples non-identical.

## Current interpretation

The baseline is the cleanest thing to talk about right now because it isolates
the representation itself. The most defensible current statement is:

- the scorer is sane because exact-image and clean re-extraction retrieval pass,
- the old low identification similarity was caused by z-score preprocessing of
  noisy probes,
- `l2_only` is the cleaner default for the synthetic baseline,
- the color-only and position-only ablations should be used to determine
  whether color, location, or their combination drives the tradeoff.

It does not yet claim anything about DAM retrieval behavior. That is now out of
the active milestone.

## Reduced sweep result

A reduced synthetic sweep was run over:

- `task_mode`: `mixed_color_position`, `color_only`, `position_only`
- `pooling`: `cls`, `mean_tokens`
- `n_stored_per_color`: `2`, `4`, `6`
- `square_size`: `56`
- `ident_noise_std`: `0.0`, `5.0`
- `ident_max_shift`: `0`, `2`
- `color_only_jitter_std`: `8.0` for `color_only`

The result was:

- `0` qualifying tradeoff candidates under the current rule:
  - generalization must peak at an interior storage level,
  - identification must continue improving after that peak,
  - generalization must then drop by at least `10` points,
  - at least one point on the curve must have positive margin.

The strongest patterns from the sweep are:

- `color_only` has the best generalization:
  - up to `93.3%`
- `position_only` can have very strong identification in some early/mid-layer
  settings:
  - up to `100%`
- `mixed_color_position` does not produce the desired parabola-like tradeoff in
  the reduced grid.

One additional technical observation from the reduced sweep is that `cls` and
`mean_tokens` produced identical metrics for every reduced-sweep condition.
This suggests the current intermediate-extraction path is not exposing a
meaningful pooling difference for this model, so pooling is not yet a useful
lever in this code path.

The practical conclusion is:

- the synthetic baseline is technically working,
- the reduced sweep did not reveal the target tradeoff,
- the next logical step is either:
  - a slightly broader synthetic sweep focused on feature readout and model
    variation, or
  - moving to naturalistic images with psychological similarity labels.
