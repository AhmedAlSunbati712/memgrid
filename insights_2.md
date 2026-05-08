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

From the saved baseline summary before rerunning after the distance-metric
fix:

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

Because the generalization ground-truth labels now use corrected RGB-plus-xy
stimulus distance, the notebook should be rerun before treating these
generalization numbers as final.

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
  target.
- `std_sim`: standard deviation of those target similarities.
- `avg_probe_norm`: average norm of the probe vectors.
- `avg_target_norm`: average norm of the correct stored vectors.

## Current interpretation

The baseline is the cleanest thing to talk about right now because it isolates
the representation itself. It answers:

- can these layerwise vision features support exact-match identification?
- can these layerwise vision features support held-out synthetic
  generalization?

It does not yet claim anything about DAM retrieval behavior. That is now out of
the active milestone.
