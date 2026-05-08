# AGENT_MEMORY_2

Last updated: baseline metric fixed to use direct stimulus-space distance.

## Current objective

Present the synthetic vision baseline as the current milestone.

## What exists now

- Core research code remains at the repo root:
  - `DAM.py`, `experiments.py`, `utils.py`, `plotting.py`
- Vision baseline code lives under `vision/`
- Baseline smoke and task-definition tests live under `tests/`

## Current milestone

The active vision workflow now stops at `evaluate_layerwise_baseline(...)`.

It includes:

- synthetic stimulus generation,
- feature extraction or cache loading,
- preprocessing,
- task construction for identification and synthetic generalization,
- direct nearest-neighbor baseline evaluation layer by layer.

It does not include active DAM retrieval, sweep logic, or retrieval-diagnostic
code in the current vision experiment path.

## Work completed in this baseline milestone

- Added `vision/image_generator.py`:
  - `generate_square_stimuli(...)`
  - `perturb_image(...)`
  - `build_preview_grid(...)`
  - `metadata_to_dicts(...)`
- Added `vision/model_wrapper.py`:
  - `VisionEmbeddingWrapper`
  - extraction of selected intermediate vision layers
- Added `vision/feature_preprocess.py`:
  - `LayerwisePreprocessor`
- Added `vision/tasks.py`:
  - `TaskBatch`
  - `build_balanced_splits_by_color(...)`
  - `build_identification_task(...)`
  - `build_generalization_task_synthetic(...)`
  - `synthetic_metadata_distance(...)`
- Updated `synthetic_metadata_distance(...)` so synthetic generalization labels
  use a simple weighted stimulus-space distance:
  - normalized RGB distance for color,
  - normalized xy distance for square position.
- Reduced `vision/vision_experiments.py` to the baseline surface:
  - `load_feature_cache(...)`
  - `extract_feature_bundle(...)`
  - `subset_feature_bundle(...)`
  - metric helpers
  - `evaluate_layerwise_baseline(...)`
- Replaced the working notebook with `VisionLayerwiseBaseline.ipynb`
  - baseline-only orchestration
  - baseline-only summary output
- Updated `vision/__init__.py` exports to match the baseline-only surface.
- Rewrote `insights_2.md` as a baseline note instead of a DAM-debugging note.

## Tests and verification

- Project venv currently lacks `pytest`, so validation used the available
  Anaconda pytest executable.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_vision_task_definitions.py`
  passed: `6 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_vision_experiments_smoke.py`
  passed: `3 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_vision_smoke.py`
  passed: `3 passed`.
- `VisionLayerwiseBaseline.ipynb` JSON parses successfully.

## Baseline result summary to reference

- Identification baseline:
  - `layer_0`: `66.7%`
  - `layer_6`: `16.7%`
  - `layer_11`: `8.3%`
- Generalization baseline:
  - `layer_0`: `41.7%`
  - `layer_6`: `41.7%`
  - `layer_11`: `50.0%`

## Next step after this milestone

- Keep the meeting framing focused on the baseline pipeline and baseline
  results.
- Re-run `VisionLayerwiseBaseline.ipynb` after the metric change because
  generalization targets and reported baseline values may change.
- If work resumes later, inspect whether the corrected stimulus-space labels
  reveal the intended identification/generalization tradeoff before revisiting
  DAM retrieval.
