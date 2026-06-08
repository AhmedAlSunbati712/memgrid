# AGENT_MEMORY_2

Last updated: naturalistic corruption path added to break the identification ceiling, with spot-check runs showing non-ceiling identification regimes.

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
- Added `vision/baseline_diagnostics.py`:
  - exact-image sanity evaluation
  - clean re-extraction sanity evaluation
  - perturbation-sweep summary over prepared feature bundles
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
- Updated `generate_square_stimuli(...)`:
  - optional `color_jitter_std` for non-degenerate fixed-position color-only
    ablations.
- Updated `perturb_image(...)`:
  - shifted pixels now use background-filled translation instead of `np.roll`
    wraparound.
- Reduced `vision/vision_experiments.py` to the baseline surface:
  - `load_feature_cache(...)`
  - `extract_feature_bundle(...)`
  - `subset_feature_bundle(...)`
  - metric helpers
  - `evaluate_layerwise_baseline(...)`, now including target similarity,
    best-wrong similarity, and margin diagnostics.
- Replaced the working notebook with `VisionLayerwiseBaseline.ipynb`
  - baseline-only orchestration
  - baseline-only summary output
  - sanity-check block
  - mixed/color-only/position-only ablation block
- Updated `vision/run_synthetic_features.py`:
  - repo-root direct execution now works,
  - l2-only is the default preprocessing,
  - `--zscore` explicitly enables the old z-score behavior for ablations.
- Added `vision/synthetic_sweep.py`:
  - reusable sweep-config generation
  - reduced CPU-friendly sweep generation
  - tradeoff-candidate detection
- Added `vision/run_synthetic_sweep.py`:
  - executes the synthetic sweep
  - saves JSON / CSV / summary artifacts under `results/vision/sweeps/`
- Updated `vision/__init__.py` exports to match the baseline-only surface.
- Rewrote `insights_2.md` as a baseline note instead of a DAM-debugging note.

## Current diagnosis

The low identification baseline similarity was mainly a preprocessing artifact.
Exact-image retrieval and clean re-extraction retrieval pass at `100%` with
target cosine near `1.0`. The drop appears when noisy identification probes are
transformed with z-score statistics fit on the clean synthetic set.

The notebook default is now `PREPROCESS_MODE = "l2_only"`. Keep `zscore_l2` as
an ablation/failure mode, not as the default baseline.

## Reduced sweep result

A reduced CPU-friendly synthetic sweep was executed across:

- `task_mode`: `mixed_color_position`, `color_only`, `position_only`
- `pooling`: `cls`, `mean_tokens`
- `n_stored_per_color`: `2`, `4`, `6`
- `square_size`: `56`
- `ident_noise_std`: `0.0`, `5.0`
- `ident_max_shift`: `0`, `2`
- `color_only_jitter_std`: `8.0`

Outcome:

- `0` qualifying tradeoff candidates were found.
- Best identification row:
  - `position_only`, `cls`, `layer_6`, `n_stored_per_color=2`,
    `ident_noise_std=0.0`, `ident_max_shift=2`
  - `ident_accuracy = 100.0%`
  - `gen_accuracy = 51.4%`
- Best generalization row:
  - `color_only`, `cls`, `layer_11`, `n_stored_per_color=2`,
    `ident_noise_std=0.0`, `ident_max_shift=0`
  - `gen_accuracy = 93.3%`
- `cls` and `mean_tokens` produced identical metrics across the reduced sweep,
  so the current code path does not expose a meaningful pooling difference for
  this model.

## Model comparison milestone

Implemented:

- Added `vision/model_comparison.py`:
  - fixed 4-model comparison config generation
  - pooling policy by model family
  - summary reduction across all compared models
- Added `vision/run_model_comparison.py`:
  - executes the synthetic baseline comparison across:
    - `vit_base_patch16_224`
    - `deit_small_patch16_224`
    - `convnext_tiny`
    - `resnet50`
  - saves:
    - per-model JSON
    - combined CSV
    - sanity JSON
    - summary JSON
  - output path: `results/vision/model_comparison/`
- Added `VisionModelComparison.ipynb`:
  - notebook entry point for the multi-model comparison
  - reads the saved summary and combined CSV
- Added `results.md`:
  - short running log of completed experiment passes and main findings
- Updated `vision/model_wrapper.py`:
  - transformer pooling now uses prefix tokens for `cls` / `auto`
  - `mean_tokens` now uses patch-map averaging for transformers
  - conv backbones now use the `forward_intermediates(...)` path cleanly
- Updated `vision/__init__.py` exports for the new comparison utilities
- Added `tests/test_model_comparison_smoke.py`
- Extended `tests/test_vision_smoke.py` to cover:
  - conv backbone extraction
  - real `cls` vs `mean_tokens` divergence for ViT

Executed result:

- Full comparison completed across `108` conditions.
- All sanity checks passed.
- `0` qualifying tradeoff candidates were found.
- Best generalization rows remained `color_only`:
  - `vit_base_patch16_224`: `93.3%`
  - `deit_small_patch16_224`: `91.7%`
  - `convnext_tiny`: `93.3%`
  - `resnet50`: `88.3%`
- Best margin / best identification rows were usually in `position_only` or
  very easy mixed settings.
- Changing backbone did not recover the desired tradeoff.

## Naturalistic fruits milestone

Implemented:

- Added `vision/io_naturalistic.py`:
  - Peterson dataset loading
  - category integrity checks
  - concept extraction from filenames
  - leave-one-exemplar-out fold construction
- Extended `vision/tasks.py`:
  - `build_generalization_task_human_similarity(...)`
  - `build_identification_task(...)` now also accepts plain string IDs
- Added `vision/run_naturalistic_baseline.py`:
  - naturalistic fruits runner
  - two-model pass:
    - `vit_base_patch16_224`
    - `convnext_tiny`
  - fold-level identification/generalization evaluation
  - human-similarity retrieval metrics
  - human RDM Spearman alignment metrics
  - output path: `results/vision/naturalistic/`
- Added `VisionNaturalisticBaseline.ipynb`
- Updated `vision/__init__.py` exports for the naturalistic loader/task path
- Added tests:
  - `tests/test_naturalistic_smoke.py`
  - `tests/test_naturalistic_runner_smoke.py`

Executed result:

- Category integrity checks passed for:
  - `animals`
  - `fruits`
  - `vegetables`
- `fruits` was used as the first MVP category because it has:
  - `40` concepts
  - exactly `3` exemplars per concept
  - exact filename alignment with the Peterson matrix
- Fruits protocol:
  - `3` leave-one-exemplar-out folds
  - `80` stored images per fold
  - `40` held-out generalization probes per fold
  - chance accuracy = `1.25%`
- Identification was effectively solved:
  - almost all rows were `100%`
- Best average generalization:
  - `vit_base_patch16_224`, `mean_tokens`, `layer_11`
  - `48.3%`
- Best ConvNeXt average generalization:
  - `convnext_tiny`, `layer_3`
  - `45.8%`
- Best human alignment:
  - `vit_base_patch16_224`, `mean_tokens`, `layer_6`
  - `human_rdm_spearman ≈ 0.379`
- Important pattern:
  - later layers retrieve better,
  - mid layers align better to human similarity,
  - generalization margins remain negative even in the best rows.

## Naturalistic vegetables expansion

Implemented:

- Extended `vision/io_naturalistic.py`:
  - optional `required_exemplars` filtering for balanced category subsets
  - integrity metadata now records dropped concepts
- Extended `vision/run_naturalistic_baseline.py`:
  - supports `--required-exemplars`
  - adds cross-category summary generation via `summarize_cross_category(...)`
- Updated `VisionNaturalisticBaseline.ipynb`:
  - category switch now supports vegetables
  - notebook can show the cross-category summary once both categories exist
- Extended tests:
  - trimmed vegetables loader behavior
  - trimmed vegetables fold construction
  - cross-category summary smoke test

Executed result:

- Vegetables was run on the balanced subset:
  - `39` concepts
  - `117` images
  - dropped concepts: `taisoi`, `tatsoi`
  - `3` leave-one-exemplar-out folds
  - `78` stored and `39` held-out probes per fold
  - chance accuracy ≈ `1.28%`

## Animals-first DAM frontier analysis

Implemented:

- Extended `vision/naturalistic_dam.py` with focused frontier helpers:
  - `FrontierBranchSpec`
  - `get_animals_frontier_configs()`
  - `get_frontier_branch_specs()`
  - `pareto_frontier_rows(...)`
  - `classify_frontier_shape(...)`
  - `summarize_frontier_runs(...)`
- Added `vision/run_animals_dam_frontier.py`:
  - fixed main branch:
    - `animals / vit_base_patch16_224 / cls / layer_11`
  - fixed failure-control branch:
    - `fruits / vit_base_patch16_224 / cls / layer_11`
  - reuses the naturalistic DAM evaluation path and writes:
    - `results/vision/naturalistic_dam/frontier/frontier_rows.csv`
    - `results/vision/naturalistic_dam/frontier/frontier_summary.json`
- Updated `VisionNaturalisticDAM.ipynb`:
  - added dedicated sections for:
    - animals frontier
    - fruits control frontier
    - frontier classification summary
- Extended `tests/test_naturalistic_dam_smoke.py`:
  - frontier config generation
  - branch selection
  - Pareto frontier extraction
  - frontier classification

Tested:

- `pytest -q tests/test_naturalistic_dam_smoke.py`
  - `14 passed`
- `python vision/run_animals_dam_frontier.py --device cpu --batch-size 16 --output-dir results/vision/naturalistic_dam/frontier --max-configs 6`
  - completed successfully
- full uncapped frontier run was launched on the same branch set for the overnight pass

Validated capped-pass result:

- `animals` branch classification:
  - `no_tradeoff`
- Best clean animals row:
  - baseline generalization: `56.25%`
  - DAM generalization: `67.5%`
  - `gen_accuracy_delta = +11.25%`
  - `gen_avg_margin_delta = +0.1176`
  - `gen_avg_human_similarity_regret_delta = -0.0645`
  - `probe_rdm_spearman_delta = +0.1483`
- `animals` had `37` qualifying wins across `60` evaluated rows in the capped pass.
- `fruits` branch classification:
  - `no_useful_dam_regime`
- On `fruits`, retrieval can improve, but no clean win appeared in the capped pass.

Current interpretation:

- The project still does not show the original identification-vs-generalization
  tradeoff.
- The current stronger result is branch-dependent:
  - `animals` supports a useful DAM regime where retrieval and alignment improve
    together,
  - `fruits` remains fragile and does not support a clean DAM regime.

Next step:

- Let the full frontier run finish and use the completed artifact to confirm
  whether the fixed `animals` branch stays `no_tradeoff` across the full DAM
  strength curve.

## Advisor-facing report

Implemented:

- Added `advisor_report.md` at repo root.
- The report is written as a presentation-ready narrative rather than a raw
  experiment log.
- It references the main notebooks directly:
  - `VisionLayerwiseBaseline.ipynb`
  - `VisionModelComparison.ipynb`
  - `VisionNaturalisticBaseline.ipynb`
  - `VisionNaturalisticDAM.ipynb`

What the report covers:

- the original project goal,
- why the synthetic benchmark was useful but not sufficient,
- the naturalistic baseline as the main result,
- the CLIP comparison,
- the DAM results,
- the current interpretation:
  - no original identification-vs-generalization tradeoff yet,
  - but a strong retrieval-vs-alignment story on naturalistic images,
  - with animals as the strongest DAM case.

Tested:

- checked the current notebook inventory and headings before writing
- matched the narrative against `results.md`
- ensured all notebook references point to files that currently exist in the
  repo

Next step:

- use `advisor_report.md` as the main discussion document with the advisor,
  alongside the naturalistic notebooks for live result inspection if needed

## Notebook explanation pass

Implemented:

- Updated the three main presentation notebooks with inline comment blocks at
  the top of each code cell:
  - `VisionModelComparison.ipynb`
  - `VisionNaturalisticBaseline.ipynb`
  - `VisionNaturalisticDAM.ipynb`
- The comments now explain:
  - what the cell output is showing,
  - what the main table fields mean,
  - how to read the rows,
  - and what finding the cell supports.

Special attention was added to table cells, including:

- synthetic comparison summary tables,
- naturalistic per-category aggregation tables,
- cross-category baseline comparison tables,
- DAM cross-category summaries,
- ViT-vs-CLIP DAM tables,
- frontier and control-branch tables.

Tested:

- checked the updated code-cell headers directly after patching
- verified notebook JSON validity for:
  - `VisionModelComparison.ipynb`
  - `VisionNaturalisticBaseline.ipynb`
  - `VisionNaturalisticDAM.ipynb`

Next step:

- if needed, do one more readability pass later based on how the explanations
  feel during the advisor meeting, but the notebooks are now self-explanatory
  enough to present from directly

## Naturalistic identification-hardening pass

Implemented:

- Extended `vision/image_generator.py` with named corruption modes:
  - `noise_shift`
  - `occlusion`
  - `multi_cutout`
  - `warp`
  - via a new `corrupt_image(...)` entry point
- Extended `vision/vision_experiments.py`:
  - optional decision noise after similarity computation
  - `decision_noise_std`
  - `decision_noise_seed`
- Extended `vision/run_naturalistic_baseline.py`:
  - configurable corruption settings for identification probes
  - corruption labels written into result rows
  - optional preview PNG export
- Added `vision/run_naturalistic_corruption_sweep.py`
- Updated `VisionNaturalisticBaseline.ipynb`:
  - corruption sweep section
  - corruption spot-check section
- Exported `corrupt_image` from `vision/__init__.py`

Tested:

- `pytest -q tests/test_vision_smoke.py tests/test_vision_experiments_smoke.py tests/test_naturalistic_runner_smoke.py`
  - `24 passed`
- `python vision/run_naturalistic_corruption_sweep.py --help`
- notebook JSON validity still needs to be checked once the final notebook pass is complete

Executed spot checks:

- `animals`, ViT `cls`, `noise30_shift16_dnoise0`
- `animals`, ViT `cls`, `occ0.5_dnoise0`
- `animals`, ViT `cls`, `occ0.5_dnoise0.01`
- `fruits`, ViT `cls`, `occ0.5_dnoise0`

Useful findings:

- The identification ceiling was real and came from using probes that were too
  easy.
- On `animals`, strong corruption can now break the ceiling:
  - `occ0.5_dnoise0`: strongest retrieval row `ident_accuracy = 90.0%`,
    `gen_accuracy = 70.0%`
  - `occ0.5_dnoise0.01`: strongest retrieval row `ident_accuracy = 67.5%`,
    `gen_accuracy = 65.0%`
- On `fruits`, `occ0.5_dnoise0` also breaks the ceiling:
  - strongest retrieval row `ident_accuracy = 80.0%`,
    `gen_accuracy = 47.5%`

Interpretation:

- The old naturalistic identification task was too weak to meaningfully stress
  retrieval.
- Large occlusion is the cleanest simple corruption so far.
- A small amount of decision noise can lower identification further while
  preserving a usable generalization signal.

Next step:

- run DAM on the new non-ceiling animal regime first:
  - `animals`
  - `vit_base_patch16_224`
  - `cls`
  - `layer_11`
  - `occ0.5`
  - compare `decision_noise_std = 0.0` vs `0.01`
- Identification was again effectively solved.
- Best average generalization:
  - `vit_base_patch16_224`, `cls`, `layer_11`
  - `49.6%`
- Best ConvNeXt average generalization:
  - `convnext_tiny`, `layer_3`
  - `45.3%`
- Best vegetables human alignment:
  - `vit_base_patch16_224`, `mean_tokens`, `layer_6`
  - `human_rdm_spearman ≈ 0.462`
- Cross-category pattern now replicated:
  - late layers retrieve best in both fruits and vegetables
  - mid layers align best to human similarity in both fruits and vegetables
- Vegetables slightly improved best retrieval and clearly improved best
  human alignment relative to fruits.

## Naturalistic animals expansion

Implemented:

- Extended `vision/run_naturalistic_baseline.py` again:
  - supports `split_mode="random_storage_curve"`
  - supports repeated-seed storage curves via `storage_sizes` and `n_seeds`
  - generalizes cross-category summaries beyond two categories
- Updated `VisionNaturalisticBaseline.ipynb`:
  - discovers the latest multi-category comparison artifact automatically
- Extended `tests/test_naturalistic_runner_smoke.py`:
  - deterministic random-storage split coverage

Executed result:

- Animals used the random storage-curve protocol:
  - `n_stored = 40, 80`
  - `5` seeds each
  - `120` total images
- Identification was again effectively solved.
- Best average retrieval was much stronger than in fruits or vegetables:
  - `vit_base_patch16_224`, `cls`, `layer_11`, `n_stored=80`
  - `gen_accuracy = 63.0%`
- Best mean-tokens retrieval:
  - `vit_base_patch16_224`, `mean_tokens`, `layer_11`, `n_stored=80`
  - `gen_accuracy = 57.5%`
- Best ConvNeXt retrieval:
  - `convnext_tiny`, `layer_3`, `n_stored=80`
  - `gen_accuracy = 57.5%`
- Best animal alignment:
  - `vit_base_patch16_224`, `mean_tokens`, `layer_6`
  - `human_rdm_spearman ≈ 0.423`
- Important difference from fruits/vegetables:
  - best animal late-layer rows achieved positive generalization margins.
- Animal same-concept accuracy is now intentionally treated as undefined / `NaN`
  because the `stim-xxxx` filenames do not encode a reliable concept grouping.

## Three-category naturalistic comparison

Executed result:

- Added `animals` into the naturalistic comparison summary.
- The replicated pattern now holds across:
  - `fruits`
  - `vegetables`
  - `animals`
- Category-level picture:
  - best retrieval:
    - fruits: `48.3%`
    - vegetables: `49.6%`
    - animals: `63.0%`
  - best alignment:
    - fruits: `0.379`
    - vegetables: `0.462`
    - animals: `0.423`
- Interpretation:
  - late layers are consistently best for retrieval,
  - mid layers are consistently best for human alignment,
  - animals gives the strongest retrieval regime,
  - vegetables gives the strongest alignment regime.

## Tests and verification

- Project venv currently lacks `pytest`, so validation used the available
  Anaconda pytest executable.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_vision_task_definitions.py`
  passed: `6 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_vision_experiments_smoke.py`
  passed: `5 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_vision_smoke.py`
  passed: `5 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_vision_experiments_smoke.py tests/test_vision_smoke.py tests/test_vision_task_definitions.py`
  passed: `16 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_synthetic_sweep_smoke.py`
  passed: `5 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_model_comparison_smoke.py`
  passed: `3 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_vision_smoke.py tests/test_vision_experiments_smoke.py tests/test_synthetic_sweep_smoke.py tests/test_model_comparison_smoke.py`
  passed: `19 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_naturalistic_smoke.py tests/test_naturalistic_runner_smoke.py`
  passed before vegetables expansion.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_vision_task_definitions.py tests/test_naturalistic_smoke.py tests/test_naturalistic_runner_smoke.py`
  passed before vegetables expansion.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_naturalistic_smoke.py tests/test_naturalistic_runner_smoke.py`
  passed after vegetables expansion: `7 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_naturalistic_runner_smoke.py`
  passed after animals expansion: `3 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q`
  passed after animals expansion: `49 passed`.
- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_synthetic_features.py --help`
  now works from the repo root.
- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_synthetic_sweep.py --reduced --batch-size 32 --device cpu`
  completed and saved:
  - `results/vision/sweeps/synthetic_sweep_vit_base_patch16_224.json`
  - `results/vision/sweeps/synthetic_sweep_vit_base_patch16_224.csv`
  - `results/vision/sweeps/synthetic_sweep_summary_vit_base_patch16_224.json`
- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_model_comparison.py --device cpu --batch-size 16`
  completed and saved:
  - `results/vision/model_comparison/model_comparison_summary.json`
  - `results/vision/model_comparison/model_comparison_combined.csv`
  - `results/vision/model_comparison/model_comparison_sanity.json`
  - per-model JSON files for the four backbones
- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_naturalistic_baseline.py --category fruits --device cpu --batch-size 16`
  completed and saved:
  - `results/vision/naturalistic/dataset_integrity.json`
  - `results/vision/naturalistic/fruits_combined.csv`
  - `results/vision/naturalistic/fruits_summary.json`
  - per-model JSON files for the fruits MVP
- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_naturalistic_baseline.py --category vegetables --required-exemplars 3 --device cpu --batch-size 16`
  completed and saved:
  - `results/vision/naturalistic/vegetables_combined.csv`
  - `results/vision/naturalistic/vegetables_summary.json`
  - per-model JSON files for vegetables
- Cross-category summary saved:
  - `results/vision/naturalistic/fruits_vs_vegetables_summary.csv`
  - `results/vision/naturalistic/fruits_vs_vegetables_summary.json`
- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_naturalistic_baseline.py --category animals --split-mode random_storage_curve --storage-sizes 40 80 --n-seeds 5 --device cpu --batch-size 16`
  completed and saved:
  - `results/vision/naturalistic/animals_combined.csv`
  - `results/vision/naturalistic/animals_summary.json`
  - per-model JSON files for animals
- Three-category comparison saved:
  - `results/vision/naturalistic/animals_vs_fruits_vs_vegetables_summary.csv`
  - `results/vision/naturalistic/animals_vs_fruits_vs_vegetables_summary.json`
- `VisionLayerwiseBaseline.ipynb` JSON parses successfully.
- `VisionModelComparison.ipynb` JSON parses successfully.
- `VisionNaturalisticBaseline.ipynb` JSON parses successfully.

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
- Re-run `VisionLayerwiseBaseline.ipynb` after the preprocessing and diagnostic
  changes. Treat older `zscore_l2` numbers as stale.
- Since the reduced synthetic sweep found no qualifying tradeoff candidate,
  and the multi-model comparison also found no qualifying tradeoff candidate,
  the work has now moved to the naturalistic path.
- Next step should be:
  - use the replicated 3-category naturalistic baseline as the fixed reference
    point for DAM integration,
  - specifically test whether DAM can improve late-layer retrieval while
    preserving the mid-layer human-alignment structure.

## 2026-05-21 - CLIP naturalistic integration

### What I implemented

- Extended `vision/run_naturalistic_baseline.py` so the naturalistic runner can
  include CLIP image encoders without changing the task definitions:
  - added `BASELINE_MODEL_SPECS`
  - added `CLIP_MODEL_SPECS`
  - added `get_model_specs(include_clip=...)`
  - added CLI flag `--include-clip`
  - added optional `model_specs=` argument to `run_naturalistic_baseline(...)`
- Kept CLIP image-only:
  - used `vit_base_patch16_clip_224.openai`
  - no text prompts
  - no zero-shot scoring
- Updated `VisionNaturalisticBaseline.ipynb`:
  - imported `get_model_specs`
  - added `INCLUDE_CLIP = True`
  - routed fruits / vegetables / animals cells through `model_specs=MODEL_SPECS`
- Added smoke coverage:
  - `tests/test_vision_smoke.py`
    - added CLIP wrapper pooling smoke test
  - `tests/test_naturalistic_runner_smoke.py`
    - added CLIP model-spec selection smoke test

### What I ran

- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_naturalistic_baseline.py --category fruits --device cpu --batch-size 16 --include-clip`
- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_naturalistic_baseline.py --category vegetables --required-exemplars 3 --device cpu --batch-size 16 --include-clip`
- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_naturalistic_baseline.py --category animals --split-mode random_storage_curve --storage-sizes 40 80 --n-seeds 5 --device cpu --batch-size 16 --include-clip`
- Rebuilt the cross-category summary with:
  - `summarize_cross_category(Path("results/vision/naturalistic"), ["fruits", "vegetables", "animals"])`

### What the results say

- All CLIP runs passed exact-image and clean re-extraction sanity.
- CLIP improved human alignment strongly on all three categories.
- Best alignment rows are now CLIP `cls` at `layer_11`:
  - fruits: `0.5796`
  - vegetables: `0.4936`
  - animals: `0.5497`
- CLIP did not replace the original ViT as the best retrieval model overall.
- Best retrieval rows stayed with `vit_base_patch16_224`:
  - fruits: `48.3%`
  - vegetables: `49.6%`
  - animals: `57.9%`
- CLIP remained competitive on retrieval:
  - fruits: `46.7%`
  - vegetables: `45.3%`
  - animals: `54.9%`
- Animals is the most promising CLIP category:
  - CLIP `cls` `layer_11` gives strong retrieval, positive generalization
    margin, and the best alignment score in the whole naturalistic benchmark.

### Files and artifacts

- Updated:
  - `vision/run_naturalistic_baseline.py`
  - `VisionNaturalisticBaseline.ipynb`
  - `tests/test_vision_smoke.py`
  - `tests/test_naturalistic_runner_smoke.py`
  - `results.md`
- Naturalistic artifacts were refreshed in:
  - `results/vision/naturalistic/`
  - category CSV / JSON files now include CLIP rows
  - `animals_vs_fruits_vs_vegetables_summary.csv`
  - `animals_vs_fruits_vs_vegetables_summary.json`

### How it was tested

- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_vision_smoke.py tests/test_naturalistic_runner_smoke.py`
  passed: `11 passed`
- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_naturalistic_baseline.py --help`
  works and shows `--include-clip`
- `timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True)`
  loads successfully and exposes transformer intermediates
- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q`
  passed after CLIP integration: `52 passed`
- `VisionNaturalisticBaseline.ipynb` still parses as JSON

### Next steps

- Run the same CLIP image-tower comparison on the synthetic benchmark.
- After that, decide whether DAM should be tested on:
  - original ViT only,
  - CLIP only,
  - or both naturalistic reference encoders.

## 2026-05-21 - CLIP synthetic integration

### What I implemented

- Updated `vision/model_comparison.py` so the synthetic comparison now centers on:
  - `vit_base_patch16_224`
  - `convnext_tiny`
  - `vit_base_patch16_clip_224.openai`
- Treated CLIP as transformer-like in the same path as the base ViT:
  - pooling options are `cls` and `mean_tokens`
- Updated `tests/test_model_comparison_smoke.py`:
  - replaced the old DeiT expectation with CLIP
  - asserted that CLIP appears in `DEFAULT_MODEL_NAMES`
- Updated `VisionModelComparison.ipynb`:
  - notebook description now mentions the CLIP image encoder
  - added one compact `ViT vs CLIP` summary cell using the saved comparison CSV

### What I ran

- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_model_comparison.py --device cpu --batch-size 16`

### What the results say

- Synthetic sanity checks still passed.
- CLIP did not produce any tradeoff candidates.
- Best `color_only` generalization stayed `93.3%` and CLIP matched it.
- CLIP had the strongest `color_only` generalization margin:
  - about `0.00229`
  - larger than the base ViT row for the same task family
- CLIP did not improve the harder synthetic regimes:
  - `mixed_color_position`
    - ViT best: `73.3%`
    - CLIP best: `65.0%`
  - `position_only`
    - ViT best: `82.9%`
    - CLIP best: `71.4%`
- So CLIP behaves like a stronger version of the same easy synthetic signal:
  - it is best when color dominates,
  - it does not recover the missing synthetic tradeoff.

### Files and artifacts

- Updated:
  - `vision/model_comparison.py`
  - `tests/test_model_comparison_smoke.py`
  - `VisionModelComparison.ipynb`
  - `results.md`
- Refreshed:
  - `results/vision/model_comparison/model_comparison_combined.csv`
  - `results/vision/model_comparison/model_comparison_summary.json`
  - `results/vision/model_comparison/model_comparison_sanity.json`
  - per-model JSON files, including CLIP

### How it was tested

- `/Volumes/youwhat/anac/anaconda3/bin/pytest -q tests/test_model_comparison_smoke.py tests/test_vision_smoke.py`
  passed: `10 passed`
- `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_model_comparison.py --device cpu --batch-size 16`
  completed and saved refreshed comparison artifacts
- `python -m json.tool VisionModelComparison.ipynb`
  passed
- `python -m json.tool results/vision/model_comparison/model_comparison_summary.json`
  passed

### Next steps

- Use the naturalistic benchmark as the main DAM testbed.
- If DAM is compared across encoders, prioritize:
  - `vit_base_patch16_224`
  - `vit_base_patch16_clip_224.openai`
- Do not spend more time widening the synthetic backbone search unless a later DAM result specifically needs a synthetic control rerun.

## 2026-05-21 - Naturalistic DAM benchmark

### What I implemented

- Added `vision/naturalistic_dam.py`
  - staged DAM config grids
  - baseline-anchor layer selection from saved naturalistic CSVs
  - probe-RDM alignment scoring before vs after DAM
  - cross-category DAM summary helper
- Added `vision/run_naturalistic_dam.py`
  - reuses the existing naturalistic split logic and feature extraction
  - runs baseline-vs-DAM comparisons with delta metrics
- Added `VisionNaturalisticDAM.ipynb`
  - separate fruits / vegetables / animals / cross-category sections
- Added `tests/test_naturalistic_dam_smoke.py`

### Important implementation note

- The first DAM sweep failed badly.
- Exact-image retrieval collapsed under the original aggressive settings.
- The stable DAM regime required:
  - mean-centering the stored/probe features inside the DAM path
  - L2 renormalizing after centering
  - much smaller `beta`, `alpha`, and `steps_multiplier`

### What I ran

- `python vision/run_naturalistic_dam.py --category fruits --device cpu --batch-size 16 --baseline-dir results/vision/naturalistic --output-dir results/vision/naturalistic_dam --include-clip`
- `python vision/run_naturalistic_dam.py --category vegetables --device cpu --batch-size 16 --baseline-dir results/vision/naturalistic --output-dir results/vision/naturalistic_dam --include-clip`
- `python vision/run_naturalistic_dam.py --category animals --device cpu --batch-size 16 --baseline-dir results/vision/naturalistic --output-dir results/vision/naturalistic_dam --include-clip`

### What the results say

- `fruits`
  - no qualifying DAM win
  - DAM can improve retrieval, but it damages probe-geometry alignment too much
- `vegetables`
  - qualifying DAM wins exist
  - best clean win: `vit_base_patch16_224` `cls` `layer_11`
  - generalization improved from `53.85%` to `56.41%`
- `animals`
  - strongest clean DAM result on the validated ViT/CLIP branch set
  - best clean win: `vit_base_patch16_224` `cls` `layer_11`
  - generalization improved from `56.25%` to `67.5%`
  - identification stayed at `100%`

### Files and artifacts

- Added:
  - `vision/naturalistic_dam.py`
  - `vision/run_naturalistic_dam.py`
  - `tests/test_naturalistic_dam_smoke.py`
  - `VisionNaturalisticDAM.ipynb`
- Updated:
  - `results.md`
- Refreshed:
  - `results/vision/naturalistic_dam/fruits_combined.csv`
  - `results/vision/naturalistic_dam/fruits_summary.json`
  - `results/vision/naturalistic_dam/vegetables_combined.csv`
  - `results/vision/naturalistic_dam/vegetables_summary.json`
  - `results/vision/naturalistic_dam/animals_combined.csv`
  - `results/vision/naturalistic_dam/animals_summary.json`
  - `results/vision/naturalistic_dam/fruits_vs_vegetables_vs_animals_summary.json`

### How it was tested

- `python vision/run_naturalistic_dam.py --help`
  passed
- `pytest -q tests/test_naturalistic_dam_smoke.py tests/test_naturalistic_runner_smoke.py`
  passed
- `pytest -q`
  passed: `58 passed`
- `python -m json.tool VisionNaturalisticDAM.ipynb`
  passed

### Next steps

- Compare DAM behavior on the same naturalistic anchor layers more directly between:
  - `vit_base_patch16_224`
  - `vit_base_patch16_clip_224.openai`
- Decide whether to widen the animals DAM pass beyond the validated branch set or focus on a tighter DAM tradeoff analysis around the strongest late-layer retrieval regime.

## 2026-05-21 - Naturalistic DAM ViT vs CLIP head-to-head

### What I implemented

- Extended `vision/naturalistic_dam.py` with:
  - matched branch-pair construction
  - clean-win vs retrieval-only outcome classification
  - head-to-head summary generation
  - retrieval/alignment frontier artifact generation
- Updated `vision/run_naturalistic_dam.py` so multi-category runs also emit the head-to-head summary.
- Updated `VisionNaturalisticDAM.ipynb` so it now shows:
  - cross-category DAM summary
  - direct `ViT vs CLIP` comparison table
  - retrieval-vs-alignment frontier table

### What I ran

- `python -m json.tool VisionNaturalisticDAM.ipynb`
- `pytest -q tests/test_naturalistic_dam_smoke.py tests/test_naturalistic_runner_smoke.py`
- `pytest -q`
- generated the comparison artifacts directly from `results/vision/naturalistic_dam/*.csv`

### What the results say

- `animals`:
  - `ViT` is the clean DAM winner on the matched `cls` retrieval anchor
  - `ViT` delta: `+11.25%`
  - `CLIP` delta: `+7.5%`
- `vegetables`:
  - `ViT` also owns the clean win on the matched `cls` retrieval anchor
  - `CLIP` can improve retrieval, but is weaker under the full clean-win criterion
- `fruits`:
  - neither encoder gives a clean DAM win
  - `CLIP` often gets the larger raw retrieval gain
  - but its alignment preservation is less reliable

### Files and artifacts

- Updated:
  - `vision/naturalistic_dam.py`
  - `vision/run_naturalistic_dam.py`
  - `tests/test_naturalistic_dam_smoke.py`
  - `VisionNaturalisticDAM.ipynb`
  - `results.md`
- Added / refreshed:
  - `results/vision/naturalistic_dam/vit_vs_clip_head_to_head.json`
  - `results/vision/naturalistic_dam/vit_vs_clip_head_to_head.csv`
  - `results/vision/naturalistic_dam/vit_vs_clip_frontier.csv`

### How it was tested

- `pytest -q tests/test_naturalistic_dam_smoke.py tests/test_naturalistic_runner_smoke.py`
  passed
- `pytest -q`
  passed: `62 passed`
- `python -m json.tool VisionNaturalisticDAM.ipynb`
  passed

### Next steps

- The strongest next move is to push on the `animals` late-layer `ViT` branch to see whether a clearer retrieval-vs-alignment tradeoff frontier exists there.
- In parallel, use the matched comparison to understand why `fruits` remains fragile even when `CLIP` can improve raw retrieval.

## 2026-05-28 - Hard-identification DAM hypothesis test

### What I implemented

- Extended `vision/naturalistic_dam.py` so DAM evaluation can optionally use decision noise at the final winner-selection stage for:
  - identification probes
  - generalization probes
- Added a dedicated runner:
  - `vision/run_naturalistic_dam_hard_ident.py`
- The new runner hard-codes a narrow hypothesis-driven comparison instead of a broad search:
  - `animals` as the positive case
  - `fruits` as the control
  - `vit_base_patch16_224`
  - `cls`
  - `layer_11`
  - `occlusion_frac = 0.5`
  - `decision_noise_std in {0.0, 0.01}`
- Added `tests/test_naturalistic_hard_ident_smoke.py`
- Updated `VisionNaturalisticDAM.ipynb` with a new:
  - `## Hard Identification DAM Test`
  section that loads and explains the new artifacts

### What I ran

- `pytest -q tests/test_naturalistic_hard_ident_smoke.py tests/test_naturalistic_dam_smoke.py tests/test_naturalistic_runner_smoke.py tests/test_vision_experiments_smoke.py`
- `python vision/run_naturalistic_dam_hard_ident.py --help`
- `python vision/run_naturalistic_dam_hard_ident.py --device cpu --batch-size 16 --output-dir results/vision/naturalistic_dam_hard_ident`

### What the results say

- `animals` remains the strong positive case after breaking the identification ceiling
- Best `animals` row with `decision_noise_std = 0.0`:
  - baseline identification: `92.5%`
  - baseline generalization: `56.25%`
  - DAM identification: `97.5%`
  - DAM generalization: `68.75%`
  - generalization delta: `+12.5%`
  - margin delta: `+0.118`
  - regret delta: `-0.0659`
  - probe-RDM delta: `+0.152`
- Best `animals` row with `decision_noise_std = 0.01`:
  - baseline identification: `85.0%`
  - baseline generalization: `47.5%`
  - DAM identification: `97.5%`
  - DAM generalization: `65.0%`
  - generalization delta: `+17.5%`
  - margin delta: `+0.121`
  - regret delta: `-0.0888`
  - probe-RDM delta: `+0.149`
- `fruits` still does not yield a clean win under the same hard-identification setup:
  - raw generalization accuracy can improve
  - but the generalization margin worsens on the best rows

### Files and artifacts

- Added:
  - `vision/run_naturalistic_dam_hard_ident.py`
  - `tests/test_naturalistic_hard_ident_smoke.py`
- Updated:
  - `vision/naturalistic_dam.py`
  - `VisionNaturalisticDAM.ipynb`
  - `results.md`
- Generated:
  - `results/vision/naturalistic_dam_hard_ident/hard_ident_combined.csv`
  - `results/vision/naturalistic_dam_hard_ident/hard_ident_summary.json`

### How it was tested

- targeted pytest run passed
- hard-ident runner help path passed
- hard-ident experiment run completed successfully

### Next steps

- If only one more DAM branch is pursued, keep it on:
  - `animals`
  - `vit_base_patch16_224`
  - `cls`
  - `layer_11`
  - `occ0.5`
- Use the hard-identification regime as the default for any remaining narrow DAM comparisons, since it removes the old identification ceiling and still preserves a strong positive signal on `animals`.

## 2026-05-28 - Hard-cue tradeoff search and cue-fidelity analysis

### What I implemented

- Extended `vision/naturalistic_dam.py` with new cue-fidelity metrics for DAM generalization probes:
  - `cue_recovered_cosine_mean`
  - `cue_recovered_cosine_std`
  - `cue_displacement_mean`
  - `target_pull_gain_mean`
  - `target_pull_gain_std`
- Added `rowwise_cosine(...)`.
- Added `get_hard_ident_tradeoff_configs()` for the broader tradeoff-search grid.
- Upgraded `vision/run_naturalistic_dam_hard_ident.py` so it now:
  - supports `--config-set {stage_a,tradeoff}`
  - writes split-level rows and aggregated config summaries
  - computes a simple recoverability analysis from split-level best gains
  - writes a hypothesis summary into the JSON artifact
- Updated `tests/test_naturalistic_dam_smoke.py` with coverage for:
  - `rowwise_cosine`
  - `get_hard_ident_tradeoff_configs`

### What I ran

- `pytest -q tests/test_naturalistic_dam_smoke.py tests/test_naturalistic_hard_ident_smoke.py`
- `python vision/run_naturalistic_dam_hard_ident.py --help`
- capped tradeoff sweep:
  - `python vision/run_naturalistic_dam_hard_ident.py --device cpu --batch-size 16 --config-set tradeoff --max-configs 120 --output-dir results/vision/naturalistic_dam_tradeoff_capped`

### What the results say

- The expected retrieval-vs-cue-fidelity tradeoff did not appear on the strongest `animals` branch.
- Best aggregated `animals` row under `decision_noise_std = 0.01`:
  - `mean_gen_accuracy_delta = +7.875%`
  - `mean_gen_avg_margin_delta = +0.0910`
  - `mean_probe_rdm_spearman_delta = +0.1447`
  - `mean_cue_displacement = 0.0093`
- The top animal retrieval rows all stayed in a very low-displacement regime, roughly `0.002` to `0.009`.
- Higher-displacement `animals` configs, around `0.028` to `0.054`, had weaker retrieval gains, roughly `+4.5%` to `+6.4%`.
- `fruits` remained the weak control:
  - best mean generalization delta was only `+2.5%`
  - but mean generalization margin stayed negative on the best rows

### Supported conclusion

- The stronger supported hypothesis is:
  - **useful DAM gains come from gentle cue-preserving stabilization, not strong cue overwriting**
- This is a better-supported result than the original “stronger completion creates a useful tradeoff” story.
- In the current capped search, stronger cue displacement was associated with weaker, not stronger, retrieval gains on `animals`.

### Files and artifacts

- Updated:
  - `vision/naturalistic_dam.py`
  - `vision/run_naturalistic_dam_hard_ident.py`
  - `tests/test_naturalistic_dam_smoke.py`
  - `results.md`
- Generated:
  - `results/vision/naturalistic_dam_tradeoff_capped/hard_ident_combined.csv`
  - `results/vision/naturalistic_dam_tradeoff_capped/hard_ident_aggregated.csv`
  - `results/vision/naturalistic_dam_tradeoff_capped/hard_ident_summary.json`

### How it was tested

- targeted pytest run passed
- runner help path passed
- capped tradeoff sweep completed successfully

### Next steps

- If one more pass is worth doing, stay narrow:
  - `animals / vit_base_patch16_224 / cls / layer_11 / occ0.5`
- Search only near the mild-completion regime:
  - low `beta`
  - short step counts
  - `n=2`
- Do not widen to stronger completion unless the explicit goal is to demonstrate failure.

## 2026-05-28 - Animals image-transformer graph suite

### What I implemented

- Added `vision/run_animals_image_transformer_graphs.py`.
- The runner builds a graph suite for the naturalistic `animals` dataset only, using:
  - `vit_base_patch16_224`
  - `vit_base_patch16_clip_224.openai`
  - `cls` pooling only
  - all transformer layers `layer_0` through `layer_11`
- For each setting, it generates:
  - baseline layerwise `(gen_accuracy, ident_accuracy)` graph
  - DAM layerwise graphs for energy orders `n = 2, 4, 6, 8`
  - raw CSVs
  - aggregated CSVs
  - `config.txt`
  - `insights.md`
- Output root:
  - `result/image_transformers/`
- Settings generated:
  - `easy_s40`
  - `easy_s80`
  - `occ50_s40`
  - `occ50_s80`
  - `occ50_dnoise001_s40`
  - `occ50_dnoise001_s80`

### What I ran

- `pytest -q tests/test_animals_image_transformer_graphs_smoke.py tests/test_naturalistic_dam_smoke.py tests/test_naturalistic_runner_smoke.py`
- `python vision/run_animals_image_transformer_graphs.py --help`
- capped validation run:
  - `MPLCONFIGDIR=/private/tmp/mplcfg python vision/run_animals_image_transformer_graphs.py --device cpu --batch-size 16 --output-dir result/image_transformers_debug --max-settings 1 --max-seeds 1 --max-dam-orders 1 --max-layers 3`
- full generation run:
  - `MPLCONFIGDIR=/private/tmp/mplcfg python vision/run_animals_image_transformer_graphs.py --device cpu --batch-size 16 --output-dir result/image_transformers`

### What the results produced

- The full artifact tree was generated under:
  - `result/image_transformers/`
- Each of the 6 setting folders contains:
  - `baseline_layers_gen_vs_ident.png`
  - `dam_n2_layers_gen_vs_ident.png`
  - `dam_n4_layers_gen_vs_ident.png`
  - `dam_n6_layers_gen_vs_ident.png`
  - `dam_n8_layers_gen_vs_ident.png`
  - `baseline_raw.csv`
  - `baseline_aggregated.csv`
  - `dam_raw.csv`
  - `dam_aggregated.csv`
  - `config.txt`
  - `insights.md`
- A root `README.md` was also written to `result/image_transformers/`.

### Notes

- The graph titles and figure text include the evaluated setting details.
- `config.txt` includes:
  - corruption parameters
  - decision-noise level
  - storage sizes and probe counts
  - number of seeds
  - DAM energy orders
  - fixed DAM parameters
  - feature dimensionality / neuron count per layer

### How it was tested

- targeted pytest run passed
- runner help path passed
- capped validation run succeeded and produced the expected file layout
- full generation run completed successfully

### Next steps

- If the next step is interpretation only, use the per-setting `insights.md` files and the aggregated CSVs directly.
- If the next step is a cleaner comparison view, build one top-level summary table across the 6 setting folders from the generated aggregated CSVs rather than rerunning the experiments.

## 2026-06-04 - Similarity-ordered animals tradeoff suite

### What I implemented

- Added `vision/run_animals_image_transformer_similarity_ordered.py`.
- The runner builds a second animals-only result package under:
  - `result/image_transformers_similarity_ordered/`
- It re-runs the ViT/CLIP `cls`-pooling layerwise analysis across:
  - all prior settings
  - plus new `s100` settings:
    - `easy_s100`
    - `occ50_s100`
    - `occ50_dnoise001_s100`
- For every split/model/layer row it computes stored-pattern geometry:
  - `mean_pairwise_cosine_similarity`
  - `mean_pairwise_cosine_distance`
  - their standard deviations
- It emits two plot families for baseline and each DAM order:
  - accuracy vs pairwise-distance plots
  - `(gen_accuracy, ident_accuracy)` scatter plots colored and connected by pairwise-distance order
- It writes:
  - per-setting `tradeoff_summary.csv`
  - root `combined_baseline_aggregated.csv`
  - root `combined_dam_aggregated.csv`
  - root `tradeoff_summary.csv`
  - root `overall_insights.md`
- Added smoke coverage in:
  - `tests/test_animals_image_transformer_similarity_ordered_smoke.py`

### Important implementation defaults

- The animals dataset used in this pipeline has 120 total images.
- Because generalization requires held-out probes, `n_stored=120` is invalid here.
- The large-storage extension therefore uses `n_stored=100`, leaving `20` probes.

### What I ran

- `pytest -q tests/test_animals_image_transformer_similarity_ordered_smoke.py tests/test_animals_image_transformer_graphs_smoke.py`
- runner help:
  - `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_animals_image_transformer_similarity_ordered.py --help`
- capped validation run:
  - `MPLCONFIGDIR=/private/tmp/mplcfg /Volumes/youwhat/anac/anaconda3/bin/python vision/run_animals_image_transformer_similarity_ordered.py --device cpu --batch-size 16 --output-dir result/image_transformers_similarity_ordered_debug --max-settings 1 --max-seeds 1 --max-dam-orders 1 --max-layers 3`
- full generation run:
  - `MPLCONFIGDIR=/private/tmp/mplcfg /Volumes/youwhat/anac/anaconda3/bin/python vision/run_animals_image_transformer_similarity_ordered.py --device cpu --batch-size 16 --output-dir result/image_transformers_similarity_ordered`
- full repo tests:
  - `pytest -q`

### What the results produced

- The full artifact tree was generated under:
  - `result/image_transformers_similarity_ordered/`
- Condition folders generated:
  - `easy_s40`
  - `easy_s80`
  - `easy_s100`
  - `occ50_s40`
  - `occ50_s80`
  - `occ50_s100`
  - `occ50_dnoise001_s40`
  - `occ50_dnoise001_s80`
  - `occ50_dnoise001_s100`
- Root-level summaries produced:
  - `README.md`
  - `combined_baseline_aggregated.csv`
  - `combined_dam_aggregated.csv`
  - `tradeoff_summary.csv`
  - `overall_insights.md`

### What the new tradeoff analysis found

- No condition met the full `tradeoff_supported` rule.
- ViT was consistently `tradeoff_partial` across all 9 baseline settings and all 36 DAM rows.
- CLIP was weaker:
  - baseline:
    - `tradeoff_partial` in 6 settings
    - `tradeoff_not_supported` in 3 settings
  - DAM:
    - `tradeoff_partial` in 24 rows
    - `tradeoff_not_supported` in 12 rows
- The dominant pattern in the generated `tradeoff_summary.csv` is:
  - identification tends to increase with stored-pattern pairwise distance for ViT
  - best generalization still tends to occur at the high-distance end, not at the middle-distance sweet spot required for the full tradeoff rule

### How it was tested

- new smoke tests passed
- capped debug generation produced the expected file set and summaries
- full similarity-ordered sweep completed successfully
- full repo pytest passed:
  - `84 passed`

### Next steps

- If the next step is interpretation, read:
  - `result/image_transformers_similarity_ordered/overall_insights.md`
  - `result/image_transformers_similarity_ordered/tradeoff_summary.csv`
- If the next step is experimentation, the most direct follow-up is:
  - compare the `tradeoff_summary.csv` across storage sizes `40`, `80`, and `100`
  - identify whether any corruption regime moves best-generalization away from the high-distance end and toward the middle tercile
- If the goal remains the original tradeoff, the present result suggests that pairwise-distance ordering reveals a monotonic separability trend more clearly than a true intermediate-distance generalization peak.

## 2026-06-04 - Fixed distance-ordered scatter suite

### What I implemented

- Added `vision/run_animals_image_transformer_similarity_ordered_fixed.py`.
- This is a post-processing runner over the existing:
  - `result/image_transformers_similarity_ordered/`
- It does not rerun models or DAM.
- It reads the aggregated CSVs, computes a deterministic `distance_rank` per:
  - setting
  - model
  - DAM order where applicable
- It regenerates the scatter plots so the sequence is explicitly ordered by ascending mean pairwise cosine distance rank, rather than merely shading the old layerwise plots.
- It writes a new corrected root:
  - `result/image_transformers_similarity_ordered_fixed/`
- Each setting folder now includes:
  - corrected baseline plot:
    - `baseline_layers_gen_vs_ident_ordered_by_distance.png`
  - corrected DAM plots:
    - `dam_n2_layers_gen_vs_ident_ordered_by_distance.png`
    - `dam_n4_layers_gen_vs_ident_ordered_by_distance.png`
    - `dam_n6_layers_gen_vs_ident_ordered_by_distance.png`
    - `dam_n8_layers_gen_vs_ident_ordered_by_distance.png`
  - `distance_order.csv`
  - updated `insights.md`

### What I ran

- targeted smoke tests:
  - `pytest -q tests/test_animals_image_transformer_similarity_ordered_smoke.py tests/test_animals_image_transformer_graphs_smoke.py`
- runner help:
  - `/Volumes/youwhat/anac/anaconda3/bin/python vision/run_animals_image_transformer_similarity_ordered_fixed.py --help`
- capped validation run:
  - `MPLCONFIGDIR=/private/tmp/mplcfg /Volumes/youwhat/anac/anaconda3/bin/python vision/run_animals_image_transformer_similarity_ordered_fixed.py --input-dir result/image_transformers_similarity_ordered --output-dir result/image_transformers_similarity_ordered_fixed_debug --max-settings 1 --max-dam-orders 1`
- full post-processing run:
  - `MPLCONFIGDIR=/private/tmp/mplcfg /Volumes/youwhat/anac/anaconda3/bin/python vision/run_animals_image_transformer_similarity_ordered_fixed.py --input-dir result/image_transformers_similarity_ordered --output-dir result/image_transformers_similarity_ordered_fixed`
- full repo tests:
  - `pytest -q`

### What the results produced

- The corrected output tree was generated under:
  - `result/image_transformers_similarity_ordered_fixed/`
- Root-level files present:
  - `README.md`
  - `combined_baseline_aggregated.csv`
  - `combined_dam_aggregated.csv`
  - `overall_insights.md`
  - `tradeoff_summary.csv`
- All 9 setting folders are present in the fixed root and include the corrected plot names plus `distance_order.csv`.

### How it was tested

- new and existing smoke tests passed
- capped debug post-processing run produced the expected corrected artifacts
- full fixed-root post-processing run completed successfully
- full repo pytest passed:
  - `85 passed`

### Next steps

- If the next step is interpretation, use:
  - `result/image_transformers_similarity_ordered_fixed/overall_insights.md`
  - and per-setting `distance_order.csv`
- If another visualization revision is needed, it should build on the fixed root rather than the original similarity-ordered root, since the fixed root is now the first version where distance rank is the explicit ordering variable in the scatter plots.
