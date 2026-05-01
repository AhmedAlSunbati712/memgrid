# AGENT_MEMORY_2

Last updated: retrieval-stability sweep implemented.

## Current objective

Diagnose why vision-DAM retrieval similarities are unexpectedly low and add tools
to distinguish feature-geometry problems from DAM-dynamics problems.

## What exists now

- Existing DAM/grid framework remains in place (`DAM.py`, `experiments.py`, `utils.py`, `plotting.py`).
- Vision frontend and DAM integration code now live under `vision/`, with tests
  under `tests/`.

## Decisions made for tonight

- Use one timm model: `vit_base_patch16_224`.
- Extract three layers (early/mid/late) by block index.
- Start with color-only synthetic stimuli (fixed square position by default).
- Preprocessing tonight: `float32` conversion, optional z-score, L2 normalization.
- Keep cosine as the default scoring convention because `experiments.py` uses
  cosine nearest-neighbor and cosine target similarity throughout.
- Add metric variants and no-DAM baselines as diagnostics before changing the
  main experiment interpretation.

## Work completed this session

- Created this memory file as required by the plan.
- Added `vision/image_generator.py`:
  - `generate_square_stimuli(...)`
  - `perturb_image(...)`
  - preview-grid and metadata helpers.
- Added `vision/model_wrapper.py`:
  - `VisionEmbeddingWrapper` around timm models,
  - extraction from early/mid/late transformer blocks,
  - intermediate API + hook fallback,
  - pooling to fixed vectors.
- Added `vision/feature_preprocess.py`:
  - `LayerwisePreprocessor` with fit/transform and L2 normalization.
- Added `vision/run_synthetic_features.py`:
  - full generate -> extract -> preprocess -> cache pipeline.
- Added package marker `vision/__init__.py`.
- Added `tests/test_vision_smoke.py` for generator/wrapper/preprocess smoke coverage.
- Installed dependencies needed for this track: `torch`, `torchvision`, `timm`.
- Ran smoke tests successfully:
  - `pytest -q tests/test_vision_smoke.py` -> `3 passed`.
- Ran script successfully (no pretrained weights to avoid network model download):
  - `python -m vision.run_synthetic_features --model-name vit_base_patch16_224 --batch-size 8 --n-per-color 2 --output-dir results/vision --no-pretrained`
  - Outputs generated:
    - `results/vision/features_synth_vit_base_patch16_224.npz`
    - `results/vision/metadata_synth_vit_base_patch16_224.json`
    - `results/vision/preview_synth_vit_base_patch16_224.png`

## Work completed after frontend sprint

- Added `vision/tasks.py`:
  - `TaskBatch`
  - balanced color splits,
  - identification task construction,
  - synthetic metadata-distance generalization task construction.
- Added `vision/vision_experiments.py`:
  - cached feature loading helpers,
  - layerwise identification and generalization runners,
  - seeded update schedules for deterministic DAM runs.
- Renamed the working notebook to `VisionDAMPipeline.ipynb` and converted it
  from a frontend sandbox to an orchestration notebook.
- Extended preprocessing/cache behavior:
  - `LayerwisePreprocessor.to_state_dict(...)`
  - `LayerwisePreprocessor.from_state_dict(...)`
  - cache export now includes per-layer preprocessing statistics.
- Added tests:
  - `tests/test_vision_task_definitions.py`
  - `tests/test_vision_experiments_smoke.py`
- Verification run:
  - `pytest -q tests/test_vision_experiments_smoke.py` -> `2 passed`.
  - `pytest -q tests/test_vision_task_definitions.py tests/test_vision_smoke.py` -> `7 passed`.

## Current low-similarity finding

- Example notebook results showed low retrieved-to-target cosine similarities:
  near zero or slightly negative for identification, despite some above-chance
  accuracies.
- `DenseAM.ipynb` suggests a plausible cause: DAM dynamics can push continuous
  vectors away from their original manifold unless regularization and scaling
  are appropriate.
- `experiments.py` confirms that cosine is the inherited metric convention, so
  the next step should be diagnostic comparison rather than immediately
  replacing cosine.

## Work completed in low-similarity diagnostic pass

- Updated `AGENT_MEMORY_2.MD` per `AGENTS.md` after noticing the previous
  pipeline implementation had not been recorded here.
- Extended `vision/vision_experiments.py`:
  - added `score_against_stored(...)`,
  - added `find_nearest_by_metric(...)`,
  - added `evaluate_layerwise_baseline(...)`,
  - added metric options: `cosine`, `dot`, `euclidean`,
  - added DAM diagnostics: baseline accuracy, initial similarity, final
    similarity, delta similarity, retrieved norm, probe norm, target norm,
    winner agreement, baseline winners, and DAM winners.
- Updated `vision/__init__.py` to export the new diagnostic helpers.
- Updated `VisionDAMPipeline.ipynb`:
  - added `TASK_MODE`,
  - added `PREPROCESS_MODE`,
  - added `METRIC`,
  - added `NORMALIZE_RETRIEVED_FOR_SCORING`,
  - added no-DAM baseline evaluation,
  - changed summary rows to compare baseline-vs-DAM accuracy, initial/final
    similarity, delta similarity, and retrieved norm.
- Updated tests:
  - added scorer helper coverage,
  - added no-DAM baseline smoke coverage,
  - added position-only metadata-distance coverage.
- Verification run:
  - `pytest -q tests/test_vision_experiments_smoke.py` -> `5 passed`.
  - `pytest -q tests/test_vision_task_definitions.py` -> `5 passed`.
  - `pytest -q tests/test_vision_smoke.py` -> `3 passed`.

## Small follow-up fix

- Fixed notebook import robustness after `evaluate_layerwise_baseline` was added
  to `vision.__init__`.
- Cause: an already-running notebook kernel can hold an older imported `vision`
  module object that does not include newly exported names.
- Updated `VisionDAMPipeline.ipynb` import cell to invalidate import caches and
  reload `vision.vision_experiments` and `vision` before importing the exported
  helpers.
- Verification:
  - compiled the notebook import cell successfully,
  - ran `from vision import evaluate_layerwise_baseline` in a fresh Python
    process successfully.

## Documentation update

- Added `insights_2.md` with a detailed explanation of the current
  low-similarity behavior.
- The note is grounded in:
  - `results/vision/vision_dam_summary_vit_base_patch16_224.json`
  - `results/vision/vision_dam_metadata_vit_base_patch16_224.json`
  - `experiments.py`
  - `DenseAM.ipynb`
- Core conclusion recorded there:
  - the raw feature space already contains useful task signal,
  - DAM retrieval is collapsing the state norm toward zero,
  - low cosine similarity is largely a consequence of that collapse.
- Next step after this note:
  - run the notebook with larger `DAM_BETA` and smaller `DAM_ALPHA`,
    then inspect whether `avg_retrieved_norm` stays near the probe/target norm
    instead of collapsing to approximately `0.0003`.

## Work completed in retrieval-stability sweep pass

- Extended `vision/vision_experiments.py` with:
  - `DEFAULT_COLLAPSE_THRESHOLD`
  - `is_retrieval_collapsed(...)`
  - `rank_sweep_rows(...)`
  - `sweep_layerwise_dam_configs(...)`
- The sweep helper:
  - runs multiple DAM parameter configurations against the same stored/probe
    split,
  - recomputes the matching baseline per metric,
  - emits one row per `(config, layer)` with `baseline_accuracy`,
    `dam_accuracy`, `avg_initial_sim`, `avg_final_sim`, `avg_delta_sim`,
    `avg_retrieved_norm`, `winner_agreement`, and `retrieval_collapse`,
  - sorts rows so non-collapsed runs with better similarity deltas and better
    DAM accuracy rise to the top.
- Updated `vision/__init__.py` exports for the new sweep and collapse helpers.
- Updated `VisionDAMPipeline.ipynb`:
  - imported sweep helpers and collapse threshold,
  - added default sweep grids for `beta`, `alpha`, `lmbda`,
    `steps_multiplier`, and retrieved-state normalization,
  - added a sweep cell that builds `ident_sweep_rows` and `gen_sweep_rows`,
  - added sweep results into the saved summary JSON.
- Updated tests:
  - retrieval-collapse threshold check,
  - sweep-row ranking check,
  - deterministic sweep helper smoke test with expected output fields.
- Verification run:
  - `pytest -q tests/test_vision_experiments_smoke.py` -> `8 passed`.
  - `pytest -q tests/test_vision_task_definitions.py tests/test_vision_smoke.py` -> `8 passed`.
  - compiled the relevant notebook cells successfully.

## Next step after sweep implementation

- Re-run `VisionDAMPipeline.ipynb` so the new sweep cell writes
  `identification_sweep` and `generalization_sweep` into the summary artifact.
- Inspect the top-ranked rows first:
  - look for `retrieval_collapse = False`,
  - then compare `avg_delta_sim`,
  - then compare `dam_accuracy` against `baseline_accuracy`.
- If no configuration avoids collapse, the next pass should target the DAM
  update dynamics directly rather than adding more notebook diagnostics.

## Remaining tasks tonight

- Run `VisionDAMPipeline.ipynb` with `TASK_MODE="mixed_color_position"` and
  inspect whether no-DAM baseline accuracy is higher than DAM accuracy.
- Run the notebook with `TASK_MODE="color_only_fixed_position"` as a sanity
  check for color separability.
- Compare `PREPROCESS_MODE` values: `zscore_l2`, `l2_only`, `zscore_only`,
  and `raw`.
- Compare `METRIC` values: `cosine`, `dot`, and `euclidean`.
- If baseline is high but DAM is low, sweep `DAM_LAMBDA`, `DAM_BETA`,
  `DAM_ALPHA`, and `STEPS_MULTIPLIER`.
- If baseline is also low, revisit stimulus construction, feature extraction,
  or preprocessing before tuning DAM.

## Tomorrow handoff (expected)

- Use baseline-vs-DAM diagnostics to decide whether to tune DAM parameters,
  adjust preprocessing, or change task construction.

## Exact next commands (tomorrow)

1) Reproduce synthetic features quickly:

`python -m vision.run_synthetic_features --model-name vit_base_patch16_224 --batch-size 8 --n-per-color 2 --output-dir results/vision --no-pretrained`

2) Verify vision smoke tests:

`pytest -q tests/test_vision_smoke.py`

3) Run low-similarity diagnostic tests:

`pytest -q tests/test_vision_task_definitions.py tests/test_vision_experiments_smoke.py tests/test_vision_smoke.py`

## Work completed in calibrated-retrieval refinement pass

- Extended `vision/vision_experiments.py` with norm-calibration helpers:
  - `DEFAULT_OVERSHOOT_RATIO`
  - `compute_retrieved_norm_ratio(...)`
  - `is_retrieval_overshooting(...)`
  - `is_retrieval_calibrated(...)`
- Updated sweep-row generation so each row now includes:
  - `dam_accuracy`
  - `baseline_delta_accuracy`
  - `retrieved_norm_ratio`
  - `retrieval_overshoot`
  - `retrieval_calibrated`
- Updated `rank_sweep_rows(...)` to prioritize:
  - calibrated runs first,
  - collapsed runs last,
  - then positive `avg_delta_sim`,
  - then higher `dam_accuracy`,
  - then norm ratio closer to `1.0`.
- Updated `vision/__init__.py` exports for the new calibration helpers.
- Updated `VisionDAMPipeline.ipynb`:
  - imported `DEFAULT_OVERSHOOT_RATIO`,
  - narrowed the default sweep grid to the healthier region:
    - `beta = [1.0, 1.5, 2.0, 2.5]`
    - `alpha = [0.05, 0.1, 0.2]`
    - `lmbda = [0.0, 0.01, 0.05, 0.1]`
    - `steps_multiplier = [5, 10, 20]`
  - passed `overshoot_ratio` into the sweep helper,
  - added `ident_norm_ratio` and `gen_norm_ratio` to the single-run summary,
  - added saved top-calibrated and top-accuracy-improving sweep slices to the
    summary JSON.
- Fixed the sweep-result schema mismatch by standardizing on
  `dam_accuracy` instead of relying on a generic `accuracy` key in sweep rows.

## Verification for calibrated-retrieval refinement pass

- `pytest -q tests/test_vision_experiments_smoke.py` -> `9 passed`
- `pytest -q tests/test_vision_task_definitions.py tests/test_vision_smoke.py` -> `8 passed`
- Parsed `VisionDAMPipeline.ipynb` as JSON successfully after editing.

## Next step after calibrated-retrieval refinement

- Re-run `VisionDAMPipeline.ipynb` and inspect the new calibrated top rows:
  - `identification_sweep_top_calibrated`
  - `generalization_sweep_top_calibrated`
  - `identification_sweep_top_accuracy`
  - `generalization_sweep_top_accuracy`
- The immediate decision rule:
  - if calibrated generalization rows preserve most of the previous gain, keep
    tuning within this narrow region;
  - if only overshooting rows achieve the large gains, the next pass should
    target the DAM update rule itself rather than expanding the sweep again.

## Notebook result after calibrated sweep rerun

- Re-ran `VisionDAMPipeline.ipynb` with the calibrated sweep settings.
- Saved summary now shows:
  - `identification_sweep_top_calibrated`
  - `generalization_sweep_top_calibrated`
  - `identification_sweep_top_accuracy`
  - `generalization_sweep_top_accuracy`

### Main outcome

- The broad-sweep generalization gains do **not** survive in the calibrated
  regime.
- The large generalization improvements came from overshooting runs, not from
  well-calibrated retrieval.

### Key findings from the saved summary

- Top calibrated identification rows:
  - mostly `layer_0`
  - strong positive `avg_delta_sim`
  - but still below or at the strongest baseline
  - example:
    - `b2.5_a0.1_l0_s20`
    - `baseline_accuracy = 66.7%`
    - `dam_accuracy = 58.3%`
    - `retrieved_norm_ratio = 0.155`
  - one calibrated config ties the baseline on `layer_0`:
    - `66.7% -> 66.7%`
- Top calibrated identification accuracy improvements exist only for weaker
  layers:
  - `layer_6`: `16.7% -> 25.0%`
- Top calibrated generalization rows:
  - improve similarity,
  - keep norm ratio near the calibration band,
  - but do **not** improve accuracy over baseline.
- `generalization_sweep_top_accuracy` is empty.
- Direct inspection of the full sweep confirmed:
  - calibrated generalization rows: `24`
  - calibrated generalization rows with positive
    `baseline_delta_accuracy`: `0`

### Interpretation

- Collapse was a real failure mode and is now understood.
- Overshoot is also a real regime and is where the large generalization gains
  currently appear.
- Under the current update rule, the system does not achieve all of the
  following at once:
  - stable norms,
  - identification improvement over the best baseline,
  - generalization improvement over the best baseline.

## What was updated after this result

- Updated `insights_2.md` with the calibrated sweep outcome and the revised
  diagnosis.

## Verification context for this stage

- Earlier code changes were already verified with:
  - `pytest -q tests/test_vision_experiments_smoke.py` -> `9 passed`
  - `pytest -q tests/test_vision_task_definitions.py tests/test_vision_smoke.py` -> `8 passed`
- This stage was primarily notebook-result interpretation, not new code.

## Recommended next step after this pause

- Pause notebook parameter sweeps on the current DAM update rule.
- If work resumes, inspect `DAM.py` and test retrieval-rule changes aimed at:
  - preventing collapse,
  - preventing overshoot,
  - preserving probe direction while sharpening toward useful attractors.
