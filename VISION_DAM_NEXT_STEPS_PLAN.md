# Vision Models + DAM: Full Next-Step Plan

## Why this plan (context from current repo)

The current codebase already has a strong DAM experiment backbone:

- `DAM.py`: retrieval dynamics (`retrieve_differential`) and memory model parameters.
- `experiments.py`: identification/generalization loops and result schema.
- `utils.py`: nearest-neighbor and cosine-based winner selection.
- `plotting.py`: tradeoff summaries/plots.

That means the main gap is not DAM logic. The main gap is a new **vision representation frontend** and explicit **task ground-truth definitions** for visual stimuli.

This plan keeps DAM core code mostly unchanged and adds a modular vision pipeline that plugs into the same evaluation style.

---

## External grounding used for this plan

- `timm` feature extraction docs (`forward_features`, `features_only`, `forward_intermediates`, `feature_info`):
  - [HF timm feature extraction](https://huggingface.co/docs/timm/v1.0.19/feature_extraction)
  - [timm feature extraction guide](https://pprp.github.io/timm/feature_extraction/)
- `timm` model-specific preprocessing (`resolve_data_config`, `create_transform`):
  - [Using timm on Hugging Face](https://huggingface.co/docs/hub/timm)
  - [timm quickstart / transforms](https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/quickstart.mdx)
- Hook caveats in PyTorch (transformer fastpath and hook behavior):
  - [PyTorch issue on hooks + fastpath](https://github.com/pytorch/pytorch/issues/132700)
  - [Related PyTorch issue](https://github.com/pytorch/pytorch/issues/128413)
- Human similarity and representational alignment methods (for naturalistic phase):
  - [Peterson et al. 2018](https://cocosci.princeton.edu/papers/Peterson_et_al-2018-Cognitive_Science.pdf)
  - [NeurIPS 2023: human similarity alignment](https://arxiv.org/abs/2306.04507)

---

## Project goal

Test whether pretrained vision model layers show an identification-generalization tradeoff when used as DAM inputs, with:

1. A controlled synthetic task (colored square stimuli).
2. A naturalistic task using within-category human similarity judgments.

---

## Critical design decisions (decide first)

1. **Representation unit**  
   DAM stores layer activation vectors, not images.

2. **Task definitions**
   - Identification: probe is a perturbed version of a stored item; success = retrieved nearest stored vector is the correct stored ID.
   - Generalization: probe is novel (not stored); success = retrieved nearest stored vector matches the ground-truth neighbor defined in metadata space (synthetic) or human similarity space (naturalistic).

3. **Layer extraction method**
   - Preferred order:
     1) use `timm` native intermediate APIs (`forward_intermediates` / `features_only`) when available,
     2) fallback to explicit forward hooks only when needed.

4. **Cross-layer comparability**
   - Use standardized preprocessing + optional dimension alignment to fixed `D`.
   - Recommended default: z-score + L2 norm; then PCA to target `D` with per-layer cap `min(D, n_features, n_samples - 1)`.

---

## Proposed file/module additions

Create a new `vision/` package:

- `vision/image_generator.py`
- `vision/model_wrapper.py`
- `vision/feature_preprocess.py`
- `vision/tasks.py`
- `vision/vision_experiments.py`
- `vision/io_naturalistic.py`
- `vision/configs.py`

And tests:

- `tests/test_vision_smoke.py`
- `tests/test_vision_task_definitions.py`
- `tests/test_vision_preprocess.py`

Optional notebook driver:

- `VisionDAM.ipynb` (or add a clean section in `DenseAM.ipynb`)

---

## End-to-end architecture

1. **Stimuli + metadata**
   - Synthetic: generate image tensors + metadata (color, x, y, etc.).
   - Naturalistic: load images and pairwise human similarity table.

2. **Feature extraction**
   - Select model + selected layers.
   - Extract raw activations per image per layer.
   - Convert each activation to a 1D vector.

3. **Feature preprocessing**
   - Per-layer fit/transform pipeline.
   - Output standardized `X_layer` matrices (stored and probes).

4. **DAM evaluation**
   - For each layer, build DAM from stored vectors.
   - Run identification and generalization probes.
   - Produce results dict keyed by layer (same metric fields: `accuracy`, `avg_sim`, `std_sim`).

5. **Analysis**
   - Plot layerwise ID vs GEN frontier.
   - Report where layers fall (early/mid/late trends).
   - For naturalistic phase, also report representational alignment to human similarities (RSA-style metrics).

---

## Phase plan with concrete tasks and deliverables

## Phase 0: Foundation and contracts (1 day)

### Tasks

- Define canonical data contracts:
  - `StimulusRecord`: `id`, `image_path` or array, and metadata.
  - `FeatureBundle`: `{layer_name: np.ndarray [N, D_layer]}`.
  - `TaskBatch`: stored IDs, probe IDs, and ground-truth index mapping.
- Decide result schema to mirror existing experiments:
  - `results[layer][n][K] = {"accuracy", "avg_sim", "std_sim"}`.
- Decide baseline hyperparameters for first run:
  - one model, 3 layers, small `K`, small trials.

### Deliverables

- `vision/configs.py` with dataclasses and defaults.
- Short README section in this plan file (or `vision/README.md`) on data flow.

### Exit criteria

- One agreed schema for metadata, features, and metrics.
- No ambiguity about what counts as success in ID/GEN.

---

## Phase 1: Synthetic image generator (1-2 days)

### Tasks

- Implement a deterministic generator for square stimuli:
  - image size, background, square size.
  - factors: color and position (`x`, `y`), optional hold-one-factor-constant mode.
- Add perturbation functions for identification probes:
  - small pixel jitter / gaussian noise / slight affine perturbation.
- Add dataset split utility:
  - stored set vs novel probes with disjoint IDs.

### Deliverables

- `vision/image_generator.py`
- metadata tables (DataFrame or dict list) with factor columns.

### Exit criteria

- Can generate reproducible stimuli and probe sets with fixed random seed.
- Can export sample montage for sanity check.

---

## Phase 2: Vision wrapper + layer extraction (2-3 days)

### Tasks

- Build `VisionEmbeddingWrapper` around `timm`:
  - load model by name.
  - auto-build transform via `resolve_data_config` + `create_transform`.
  - layer selection by friendly spec (`early/mid/late` or explicit indices/names).
- Implement extraction backends:
  - Backend A (preferred): `forward_intermediates` / `features_only`.
  - Backend B (fallback): forward hooks.
- Implement tensor-to-vector conversion:
  - ViT default: CLS token and mean-token as configurable options.
  - CNN default: global average pooling over spatial dims.
- Add safe hook lifecycle:
  - register -> forward -> remove handles.
  - avoid storing tensors with grads; use `torch.no_grad()` and detach to CPU.
  - add fastpath guard note for transformer hook edge cases.

### Deliverables

- `vision/model_wrapper.py`

### Exit criteria

- Given a list of images, wrapper returns consistent `{layer_name: [N, D]}`.
- No memory leak from lingering hooks over repeated calls.

---

## Phase 3: Standardized feature preprocessing (1-2 days)

### Tasks

- Implement per-layer preprocessing pipeline:
  - cast `float32`.
  - optional centering/z-score using fit set stats.
  - L2 normalize each sample vector.
- Implement optional dimension alignment:
  - PCA per layer with `D_eff = min(target_D, D_layer, n_fit_samples - 1)`.
  - if strict fixed D required, add zero-pad option.
- Add fit/transform separation:
  - fit only on stored/training set to avoid leakage.
  - transform stored and probe sets with same fitted object.

### Deliverables

- `vision/feature_preprocess.py`

### Exit criteria

- Every selected layer can be transformed into comparable vectors.
- Re-running transform on same inputs yields deterministic outputs.

---

## Phase 4: Task definitions and scoring (1 day)

### Tasks

- Add explicit task builders in `vision/tasks.py`:
  - `build_identification_task(...)`
  - `build_generalization_task_synthetic(...)`
  - `build_generalization_task_human_similarity(...)`
- Synthetic generalization ground truth options:
  - color-only, position-only, weighted color+position distance.
- Naturalistic generalization ground truth:
  - for probe item, choose stored item with highest human-rated similarity (or lowest human distance).
- Keep DAM success criterion fixed:
  - retrieved nearest stored vector index equals ground-truth index.

### Deliverables

- `vision/tasks.py`

### Exit criteria

- Task builder outputs ground-truth index arrays and can be unit tested independently of DAM.

---

## Phase 5: DAM experiment runners for vision layers (2-3 days)

### Tasks

- Implement `vision_experiments.py`:
  - layerwise identification runner.
  - layerwise generalization runner.
  - combined sweep helper (similar style to `run_multiscale_ident_gen_sweep`).
- Keep DAM calls compatible with current `SilentDAM`.
- Add query perturbation controls for identification.
- Add per-layer result dictionary:
  - `results[layer][n][K] -> metrics`.

### Deliverables

- `vision/vision_experiments.py`

### Exit criteria

- First complete end-to-end run on synthetic data produces layerwise ID/GEN metrics.

---

## Phase 6: Naturalistic dataset integration (2-3 days)

### Tasks

- Implement loaders:
  - image folders by category (animals/fruits/vegetables).
  - human similarity ratings file parsing.
- Build within-category splits (consistent with provided ratings setup).
- Add mapping from image IDs to rating matrix indices.
- Validate matrix integrity:
  - symmetric checks, diagonal handling, missing pair handling.

### Deliverables

- `vision/io_naturalistic.py`

### Exit criteria

- Can run one category end-to-end and compute ID/GEN + alignment metrics.

---

## Phase 7: Analysis and reporting (1-2 days)

### Tasks

- Add plotting helpers for layerwise results:
  - layer order vs ID accuracy.
  - layer order vs GEN accuracy.
  - ID vs GEN tradeoff scatter with layer labels.
- For naturalistic track:
  - compare feature RDM with human similarity matrix using rank correlation (Spearman).
  - optionally relate human-alignment score to GEN score per layer.
- Summarize model-wise and category-wise findings.

### Deliverables

- `vision/analysis.py` (optional) or additions to `plotting.py`
- notebook report cells with reproducible figures.

### Exit criteria

- Clear figure set ready for advisor discussion.

---

## Minimum viable experiment (MVP) spec

Run this first before expanding:

- Model: one timm model (e.g., ViT or ResNet).
- Layers: exactly 3 (early/mid/late).
- Synthetic data:
  - color-only variation first (position fixed).
  - then add position variation.
- DAM settings:
  - small `K` values, one or two `n` values.
  - fixed retrieval backend (`numba` if available and valid settings).
- Output:
  - ID/GEN per layer.
  - one tradeoff plot.

If MVP works, then scale:

1) more layers, 2) more models, 3) naturalistic dataset.

---

## Suggested wrapper API (concrete)

```python
class VisionEmbeddingWrapper:
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        device: str = "cpu",
        extraction_mode: str = "auto",  # auto|intermediates|features_only|hooks
        pooling: str = "auto",          # auto|cls|mean_tokens|gap
    ):
        ...

    def list_available_layers(self) -> list[str]:
        ...

    def extract(
        self,
        images,                          # PIL list or tensor batch iterable
        layer_spec,                      # list[str] or list[int] or "early_mid_late"
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        ...
```

And preprocessing:

```python
class LayerwisePreprocessor:
    def fit(self, layer_to_features: dict[str, np.ndarray]) -> None: ...
    def transform(self, layer_to_features: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
```

---

## Experiment metric definitions (final wording)

- **Stored in DAM:** activation vectors from chosen layer for stored stimuli.
- **Identification probe:** activation vector from perturbed version of a stored stimulus.
- **Generalization probe:** activation vector from novel stimulus.

Success:

- **Identification success:** retrieved nearest stored vector index equals target stored index.
- **Generalization success:** retrieved nearest stored vector index equals metadata/human-similarity ground-truth nearest index.

Keep this wording everywhere (code comments, docs, meeting slides) to avoid ambiguity.

---

## Reuse plan for existing repo functions

- Reuse `SilentDAM` directly from `DAM.py`.
- Reuse nearest-cosine logic from `utils.py` or add vision-local equivalent with same semantics.
- Mirror return structure from `experiments.py` so existing plot/report code is easy to adapt.

Avoid touching existing grid experiments unless needed. Keep vision track additive.

---

## Test strategy

## Unit tests

- image generator determinism and metadata correctness.
- wrapper returns expected number of layers and sample counts.
- preprocessing fit/transform shape and finite-value checks.
- task builders produce valid ground-truth indices.

## Integration smoke tests

- one mini synthetic run:
  - tiny image set, 1 model, 2 layers, very small `K`.
  - assert metrics returned and finite.

## Regression/stability checks

- fixed-seed run should keep coarse layer ranking stable.
- cache read/write should reproduce feature arrays exactly.

---

## Risk register and mitigations

1. **Hook instability across model families**  
   Mitigation: prefer `timm` intermediate APIs first; fallback hooks only if needed.

2. **Cross-layer dimensionality confound**  
   Mitigation: enforce preprocessing + optional fixed-dimensional projection.

3. **Probe leakage into fit stats**  
   Mitigation: fit preprocessor on stored/train split only.

4. **Naturalistic label noise / missing pairs**  
   Mitigation: add robust parser with strict validation and explicit missing-pair policy.

5. **Runtime blowup for many layers/models**  
   Mitigation: feature caching (`.npz`/parquet), batch extraction, quick mode configs.

---

## Execution order (recommended)

1. Phase 0 contracts
2. Phase 1 synthetic generator
3. Phase 2 wrapper extraction
4. Phase 3 preprocessing
5. Phase 4 task builders
6. Phase 5 end-to-end synthetic layerwise DAM run
7. Phase 6 naturalistic loader + run
8. Phase 7 plots and advisor-ready summary

Do not start naturalistic modeling before synthetic MVP is stable.

---

## Immediate next 3 coding tasks (start here)

1. Create `vision/model_wrapper.py` with a working extractor for one timm model and 3 selected layers.
2. Create `vision/image_generator.py` with color-only square stimuli + perturbation function.
3. Create `vision/tasks.py` and `vision/vision_experiments.py` for a tiny synthetic ID/GEN run using `SilentDAM`.

If these three work end-to-end, the rest is mostly scale and evaluation design.
