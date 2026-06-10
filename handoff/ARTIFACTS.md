# Bundled Results Map

## Policy

- **Bundled deliverables** live under `handoff/results/` as real files (not symlinks).
- `handoff/results/MANIFEST.json` stores SHA256 checksums for key CSV deliverables; `scripts/verify_artifacts.py` checks integrity.
- Notebook reload (`RUN_EXPERIMENTS=False`) and live regen (`RUN_EXPERIMENTS=True`) both read/write the same paths under `handoff/results/`.

## Layout under `handoff/results/`

| Path | Contents |
|------|----------|
| `image_transformers/` | Animals ViT/CLIP **layer-index** graphs (6 legacy settings) |
| `image_transformers_similarity_ordered/` | Intermediate similarity-ordered CSVs (regen step 1) |
| `image_transformers_similarity_ordered_fixed/` | Animals **distance-rank** graphs (same 6 settings) |
| `vision/` | Synthetic, naturalistic, DAM CSVs/JSONs |
| `grid/` | Grid DAM multiscale + break-it JSON/PNG outputs |

## Key deliverables (checksums in MANIFEST.json)

### `results/image_transformers/` (animals graph suite)

Six legacy settings:

- `easy_s40`, `easy_s80` — noise+shift corruption, 40/80 stored
- `occ50_s40`, `occ50_s80` — 50% occlusion
- `occ50_dnoise001_s40`, `occ50_dnoise001_s80` — occlusion + decision noise

Per setting: `baseline_aggregated.csv`, `dam_aggregated.csv`, `config.txt`, and **5 PNGs**.

### `results/image_transformers_similarity_ordered_fixed/` (distance-rank graphs)

Post-processed similarity-ordered scatter plots. Same six settings; **5 PNGs** per setting with `*_ordered_by_distance.png` suffix.

Intermediate runs go to `handoff/results/image_transformers_similarity_ordered/`.

### `results/vision/` (vision pipeline)

| Subpath | Source |
|---------|--------|
| `naturalistic/*_combined.csv` | Naturalistic baseline |
| `naturalistic_dam/vit_vs_clip_head_to_head.csv` | DAM ViT vs CLIP |
| `model_comparison/model_comparison_combined.csv` | Synthetic multi-model |
| `synthetic/baseline_summary_*.json` | Synthetic layerwise baseline |

## Handoff outputs

| Path | Producer | Files |
|------|----------|-------|
| `handoff/results/grid/multiscale/` | `run_grid_multiscale.py` | `ident_results.json`, `gen_results.json`, `config.json`, `tradeoff_by_n.png` |
| `handoff/results/grid/breakit/{experiment}/` | `run_grid_breakit.py` / notebook | `{experiment}_raw.json`, tradeoff PNGs |
| `handoff/results/vision/` | `run_vision_*.py`, notebook regen cells | model-specific JSON/PNG |

## Refreshing bundled results

After regenerating results locally, update the manifest:

```bash
# from memgrid repo root
python -c "
from handoff.scripts._bootstrap import bootstrap_script_imports
bootstrap_script_imports('handoff/scripts/verify_artifacts.py')
from handoff.lib.artifact_manifest import write_manifest
write_manifest()
"
python handoff/scripts/verify_artifacts.py
```
