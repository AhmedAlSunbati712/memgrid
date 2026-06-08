# Results

## 2026-05-21 - Reduced ViT synthetic sweep

What was run:
- Reduced synthetic sweep on `vit_base_patch16_224`
- Task modes: `mixed_color_position`, `color_only`, `position_only`
- Storage curve: `n_stored_per_color = 2, 4, 6`

Main findings:
- No qualifying tradeoff candidate was found.
- `color_only` gave the best generalization.
- `position_only` could give very strong identification.
- The desired pattern, where generalization peaks and then drops while identification keeps improving, did not appear.

Next step:
- Compare other non-VLM vision backbones on the same synthetic baseline.

## 2026-05-21 - Multi-model synthetic baseline comparison

What was run:
- `VisionModelComparison.ipynb`
- `python vision/run_model_comparison.py --device cpu --batch-size 16`
- Models: `vit_base_patch16_224`, `deit_small_patch16_224`, `convnext_tiny`, `resnet50`
- Same synthetic tasks and sanity checks as the baseline notebook

Main findings:
- All sanity checks passed.
- No model produced a qualifying tradeoff candidate.
- Best generalization still came from `color_only`.
- Best identification usually came from `position_only` or from very easy mixed settings with no probe noise.
- `convnext_tiny` was the strongest non-ViT model for position-only generalization.
- `deit_small_patch16_224` was weaker than the baseline ViT on the best generalization row.
- `resnet50` did not improve the overall picture.

Short take:
- Changing backbone alone did not reveal the tradeoff we want.
- The synthetic task is still easy in color space and not giving a clean identification-generalization curve.

Next step:
- Move to the naturalistic image dataset with psychological similarity labels.

## 2026-05-21 - Naturalistic dataset integrity

What was checked:
- Peterson similarity file and local image folders
- Categories: `animals`, `fruits`, `vegetables`

Main findings:
- All three categories match the Peterson filenames on disk.
- `fruits` and `vegetables` each have `120` rated images at `500x500`.
- `animals` has `120` rated images at `300x300`.
- `fruits` is the cleanest first MVP:
  - `40` concepts
  - `3` exemplars per concept
  - exact leave-one-exemplar-out folds are easy to define

Next step:
- Run the naturalistic fruits baseline first.

## 2026-05-21 - Naturalistic fruits baseline

What was run:
- `VisionNaturalisticBaseline.ipynb`
- `python vision/run_naturalistic_baseline.py --category fruits --device cpu --batch-size 16`
- Models: `vit_base_patch16_224` and `convnext_tiny`
- Fruits protocol: `40` concepts, `3` folds, `80` stored and `40` held-out probes per fold

Main findings:
- All sanity checks passed.
- Identification is essentially solved:
  - almost all rows are `100%`.
- Generalization is far above chance:
  - chance is `1.25%`
  - best average generalization was `48.3%` for `vit_base_patch16_224` with `mean_tokens` at `layer_11`
  - `convnext_tiny` reached `45.8%` at `layer_3`
- Late layers gave the best retrieval accuracy.
- Mid layers gave the best human-alignment scores:
  - best `human_rdm_spearman` was about `0.379` for `vit_base_patch16_224` `mean_tokens` at `layer_6`
  - `convnext_tiny` `layer_2` was close at about `0.378`
- Generalization margins are still negative even in the best rows.
- Same-concept retrieval is high in the best late-layer rows:
  - around `80-82.5%`

Short take:
- Naturalistic data is much more informative than the synthetic squares.
- We now have a real separation between retrieval accuracy and human-structure alignment:
  - late layers retrieve better,
  - mid layers align better to human similarity.
- This is a better foundation for the next phase than continuing synthetic sweeps.

Next step:
- Expand the same naturalistic baseline to `vegetables`, then compare whether the same late-vs-mid layer pattern holds.

## 2026-05-21 - Naturalistic vegetables baseline

What was run:
- `python vision/run_naturalistic_baseline.py --category vegetables --required-exemplars 3 --device cpu --batch-size 16`
- Same naturalistic baseline protocol as fruits
- Trimmed to the balanced `39` concepts with `3` exemplars each

Main findings:
- All sanity checks passed.
- Identification is again essentially solved.
- Generalization stayed far above chance:
  - chance is about `1.28%`
  - best average generalization was `49.6%` for `vit_base_patch16_224` with `cls` at `layer_11`
  - `convnext_tiny` reached `45.3%` at `layer_3`
- Late layers again gave the best retrieval accuracy.
- Mid layers again gave the best human-alignment scores:
  - best `human_rdm_spearman` was about `0.462` for `vit_base_patch16_224` `mean_tokens` at `layer_6`
- Generalization margins are still negative in the best rows.

Short take:
- The naturalistic pattern replicates in a second category.
- Vegetables is not just a noisy repeat of fruits; it gives even stronger human alignment in the mid ViT layer.

Next step:
- Compare fruits vs vegetables directly and decide whether the same pattern is stable enough to justify DAM on naturalistic data.

## 2026-05-21 - Fruits vs Vegetables comparison

What was added:
- Cross-category summary from the saved naturalistic artifacts

Main findings:
- The same split holds in both categories:
  - late layers retrieve best,
  - mid layers align best to human similarity.
- Vegetables slightly improved the best late-layer generalization:
  - fruits best average generalization: `48.3%`
  - vegetables best average generalization: `49.6%`
- Vegetables improved the best human-alignment score more clearly:
  - fruits best `human_rdm_spearman`: about `0.379`
  - vegetables best `human_rdm_spearman`: about `0.462`
- `convnext_tiny` remained competitive on retrieval, but the strongest alignment rows still came from `vit_base_patch16_224`.

Short take:
- The late-layer retrieval vs mid-layer alignment split now looks replicated, not accidental.
- This is a much better basis for the next DAM phase than the synthetic setup was.

Next step:
- Use these naturalistic baselines as the reference point for any DAM integration, instead of going back to synthetic-first tuning.

## 2026-05-21 - Naturalistic animals baseline

What was run:
- `python vision/run_naturalistic_baseline.py --category animals --split-mode random_storage_curve --storage-sizes 40 80 --n-seeds 5 --device cpu --batch-size 16`
- Same models as the other naturalistic runs
- Random stored-set curves instead of exemplar folds

Main findings:
- All sanity checks passed.
- Identification is again essentially solved.
- Generalization is much stronger than in fruits or vegetables:
  - best average generalization was `63.0%` for `vit_base_patch16_224` `cls` at `layer_11` with `80` stored images
  - `vit_base_patch16_224` `mean_tokens` reached `57.5%`
  - `convnext_tiny` `layer_3` also reached `57.5%`
- Unlike fruits and vegetables, the best animal late-layer rows have positive generalization margins.
- Best human alignment still comes from a mid ViT layer:
  - `vit_base_patch16_224` `mean_tokens` at `layer_6`
  - `human_rdm_spearman` about `0.423`

Short take:
- The animal category did not break the protocol.
- It actually strengthened the main story:
  - late layers are best for retrieval,
  - mid layers are best for alignment,
  - and late-layer retrieval can now become positively separated from competitors.

Next step:
- Fold animals into the cross-category comparison and use that as the baseline reference for any DAM work.

## 2026-05-21 - Fruits vs Vegetables vs Animals comparison

What was added:
- 3-category naturalistic comparison summary

Main findings:
- The late-layer retrieval / mid-layer alignment split now appears in all three naturalistic categories.
- Best retrieval:
  - fruits: `48.3%`
  - vegetables: `49.6%`
  - animals: `63.0%`
- Best alignment:
  - fruits: about `0.379`
  - vegetables: about `0.462`
  - animals: about `0.423`
- Animals is the strongest retrieval category by a clear margin.
- Vegetables is the strongest human-alignment category.
- `vit_base_patch16_224` remains the strongest model family for alignment; `convnext_tiny` stays competitive on retrieval.

Short take:
- The naturalistic baseline is now replicated across three categories.
- This is strong enough to stop searching for a baseline and start asking what DAM changes relative to this reference.

Next step:
- Use the 3-category naturalistic baseline as the fixed benchmark for DAM integration and evaluate whether DAM improves retrieval without destroying the mid-layer human-alignment structure.

## 2026-05-21 - Naturalistic CLIP image-tower comparison

What was run:
- `python vision/run_naturalistic_baseline.py --category fruits --device cpu --batch-size 16 --include-clip`
- `python vision/run_naturalistic_baseline.py --category vegetables --required-exemplars 3 --device cpu --batch-size 16 --include-clip`
- `python vision/run_naturalistic_baseline.py --category animals --split-mode random_storage_curve --storage-sizes 40 80 --n-seeds 5 --device cpu --batch-size 16 --include-clip`
- Same naturalistic benchmark as before
- Added `vit_base_patch16_clip_224.openai` as an image-only encoder with `cls` and `mean_tokens`

Main findings:
- All sanity checks still passed.
- CLIP helped human alignment a lot on all three categories.
- Best alignment rows are now CLIP:
  - fruits: `0.580` with `vit_base_patch16_clip_224.openai` `cls` `layer_11`
  - vegetables: `0.494` with `vit_base_patch16_clip_224.openai` `cls` `layer_11`
  - animals: `0.550` with `vit_base_patch16_clip_224.openai` `cls` `layer_11`
- CLIP did not become the consistent best retrieval model.
- Best generalization still came from the original ViT on all three categories:
  - fruits: `48.3%` with `vit_base_patch16_224` `mean_tokens` `layer_11`
  - vegetables: `49.6%` with `vit_base_patch16_224` `cls` `layer_11`
  - animals: `57.9%` with `vit_base_patch16_224` `cls` `layer_11`
- CLIP was still competitive on retrieval:
  - fruits: `46.7%`
  - vegetables: `45.3%`
  - animals: `54.9%`
- On animals, CLIP `cls` at `layer_11` gave both strong retrieval and strong alignment:
  - `54.9%` generalization
  - positive average margin
  - `0.550` human RDM Spearman

Short take:
- CLIP changes the benchmark in a meaningful way.
- It is strongest as a human-alignment encoder, not as a pure retrieval winner.
- The original ViT still gives the best top retrieval rows, but CLIP gives a much better retrieval-alignment tradeoff, especially on animals.

Next step:
- Run the same CLIP image-tower comparison on the synthetic benchmark, then decide whether DAM should be tested on top of the original ViT, CLIP, or both.

## 2026-05-21 - Synthetic CLIP image-tower comparison

What was run:
- `python vision/run_model_comparison.py --device cpu --batch-size 16`
- Synthetic comparison now includes:
  - `vit_base_patch16_224`
  - `convnext_tiny`
  - `vit_base_patch16_clip_224.openai`
- Same synthetic tasks as before:
  - `mixed_color_position`
  - `color_only`
  - `position_only`

Main findings:
- All sanity checks still passed.
- CLIP did not produce any tradeoff candidates.
- CLIP matched the best `color_only` generalization:
  - `93.3%`, same as the original ViT
- CLIP improved the `color_only` margin noticeably:
  - ViT best `gen_avg_margin`: about `0.00034`
  - CLIP best `gen_avg_margin`: about `0.00229`
- CLIP did not improve the hard synthetic settings:
  - `mixed_color_position`: ViT best `73.3%`, CLIP best `65.0%`

## 2026-05-21 - Naturalistic DAM benchmark

What was added:
- `vision/run_naturalistic_dam.py`
- `vision/naturalistic_dam.py`
- `VisionNaturalisticDAM.ipynb`

What was run:
- `python vision/run_naturalistic_dam.py --category fruits --device cpu --batch-size 16 --baseline-dir results/vision/naturalistic --output-dir results/vision/naturalistic_dam --include-clip`
- `python vision/run_naturalistic_dam.py --category vegetables --device cpu --batch-size 16 --baseline-dir results/vision/naturalistic --output-dir results/vision/naturalistic_dam --include-clip`
- `python vision/run_naturalistic_dam.py --category animals --device cpu --batch-size 16 --baseline-dir results/vision/naturalistic --output-dir results/vision/naturalistic_dam --include-clip`

Main implementation note:
- DAM collapsed badly with the first aggressive sweep.
- Fixing that required using a DAM-only centered feature space and much smaller update settings.
- The stable regime was low-gain and short:
  - small `beta`
  - small `alpha`
  - `steps_multiplier` of `1` or `2`

Main findings:
- `fruits`:
  - no qualifying DAM win
  - best DAM row improved generalization from `37.5%` to `45.0%`
  - but it hurt probe-geometry alignment too much, so it does not count as a clean win
- `vegetables`:
  - DAM produced qualifying wins
  - cleanest win: `vit_base_patch16_224` `cls` `layer_11`
  - generalization improved from `53.85%` to `56.41%`
  - probe RDM alignment also improved
- `animals`:
  - strongest DAM result
  - validated on the narrowed ViT/CLIP smoke branch set
  - best clean win: `vit_base_patch16_224` `cls` `layer_11`
  - generalization improved from `56.25%` to `67.5%`
  - identification stayed at `100%`
  - human-similarity regret improved
  - probe RDM alignment improved

Short take:
- DAM is not universally helpful on the naturalistic benchmark.
- But it is clearly not dead either.
- The strongest pattern so far is:
  - late ViT retrieval layers can benefit from DAM,
  - especially on `animals`,
  - while `fruits` is more fragile because retrieval gains come with geometry damage.

Next step:
- Use the current naturalistic DAM benchmark to compare ViT vs CLIP more directly on the same anchor layers, then decide whether to widen the animals pass or move to a more focused DAM tradeoff analysis.

## 2026-05-21 - Naturalistic DAM ViT vs CLIP head-to-head

What was added:
- matched `ViT` vs `CLIP` branch comparison on top of the saved DAM artifacts
- `vit_vs_clip_head_to_head.json`
- `vit_vs_clip_head_to_head.csv`
- `vit_vs_clip_frontier.csv`
- updated `VisionNaturalisticDAM.ipynb` to show the comparison and frontier tables

What was compared:
- same category
- same pooling
- same anchor role
- ViT branch vs CLIP branch on the matched DAM benchmark

Main findings:
- `animals`:
  - clear `ViT` clean win
  - matched `cls` retrieval anchor at `layer_11`
  - `ViT` DAM delta: `+11.25%`
  - `CLIP` DAM delta: `+7.5%`
  - `ViT` also preserved alignment better on the winning row
- `vegetables`:
  - `ViT` clean win on the `cls` retrieval anchor
  - `CLIP` could improve retrieval too, but the clean win still belonged to `ViT`
- `fruits`:
  - neither encoder produced a clean DAM win
  - both can improve retrieval on some rows
  - `CLIP` often gets the larger retrieval delta
  - but its probe-geometry alignment degrades more

Short take:
- `ViT` is currently the better DAM substrate.
- `CLIP` can improve retrieval, but it is less reliable when the criterion includes preserving human-structure geometry.
- The strongest positive result is still the `animals` `ViT` late-layer branch.

Next step:
- Use the matched comparison to decide whether to:
  - deepen the `animals` ViT tradeoff search, or
  - analyze why `fruits` remains fragile while `animals` and `vegetables` support cleaner DAM gains.
  - `position_only`: ViT best `82.9%`, CLIP best `71.4%`
- `convnext_tiny` stayed much weaker on `mixed_color_position` and `position_only`.

Short take:
- CLIP does not fix the synthetic benchmark.
- It behaves best on the same easy axis as the other models: color.
- The synthetic conclusion stays the same:
  - color is easy,
  - position is weaker,
  - no useful tradeoff appears.

Next step:
- For DAM, prioritize the naturalistic benchmark over the synthetic one.
- If DAM is tested on multiple encoders, the strongest candidates are the original ViT and CLIP, not more synthetic-only backbone widening.

## 2026-05-21 - Animals-first DAM frontier analysis

What was added:
- `vision/run_animals_dam_frontier.py`
- frontier helpers in `vision/naturalistic_dam.py`
- advisor-facing frontier sections in `VisionNaturalisticDAM.ipynb`

What was run:
- `python vision/run_animals_dam_frontier.py --device cpu --batch-size 16 --output-dir results/vision/naturalistic_dam/frontier --max-configs 6`
- a full uncapped frontier run was also launched on the same branch set for the overnight pass

Frontier setup:
- main branch: `animals / vit_base_patch16_224 / cls / layer_11`
- control branch: `fruits / vit_base_patch16_224 / cls / layer_11`
- swept DAM strength over:
  - `n in {2, 4}`
  - `beta in {0.02, 0.05, 0.1, 0.2, 0.35, 0.5}`
  - `alpha in {0.02, 0.05, 0.1}`
  - `steps_multiplier in {1, 2, 3, 5, 8}`

Main findings from the validated capped pass:
- `animals` already looks like `no_tradeoff`, not `soft_tradeoff` or `hard_tradeoff`.
- On the best clean animals row:
  - baseline generalization: `56.25%`
  - DAM generalization: `67.5%`
  - generalization delta: `+11.25%`
  - margin delta: `+0.118`
  - regret delta: `-0.0645`
  - probe-RDM delta: `+0.148`
- `animals` had many clean wins even in the capped pass:
  - `37` qualifying wins across `60` evaluated rows
- `fruits` remained the failure control:
  - retrieval could improve
  - but no clean win appeared
  - branch classification was `no_useful_dam_regime`

Short take:
- The current evidence still does not show the original identification-vs-generalization tradeoff.
- It instead points to a sharper result:
  - on `animals`, useful DAM settings improve retrieval and alignment together
  - on `fruits`, DAM stays fragile and does not produce a clean regime

Next step:
- Let the full frontier run finish and use that completed artifact as the final answer to whether the `animals` branch stays `no_tradeoff` across the full strength curve.

## 2026-05-28 - Naturalistic identification-hardening pass

What was added:
- configurable naturalistic probe corruptions:
  - `noise_shift`
  - `occlusion`
  - `multi_cutout`
  - `warp`
- optional decision noise after similarity computation
- `vision/run_naturalistic_corruption_sweep.py`
- corruption preview images and spot-check artifacts under:
  - `results/vision/naturalistic_spotcheck/`

Why this was needed:
- the naturalistic identification task was too easy
- the old probes only used mild pixel noise plus a tiny translation
- that is why many identification rows stayed near `100%`

Main spot-check findings:
- `animals`, ViT `cls`, `noise30_shift16_dnoise0`
  - strongest retrieval row still had `100%` identification
  - but the alignment-oriented `layer_6` row dropped to `95%`
- `animals`, ViT `cls`, `occ0.5_dnoise0`
  - strongest retrieval row dropped to `90%` identification
  - generalization on that same row stayed high at `70.0%`
- `animals`, ViT `cls`, `occ0.5_dnoise0.01`
  - strongest retrieval row dropped further to `67.5%` identification
  - generalization still stayed strong at `65.0%`
- `fruits`, ViT `cls`, `occ0.5_dnoise0`
  - strongest retrieval row dropped to `80.0%` identification
  - generalization stayed at `47.5%`

Short take:
- the identification ceiling was real and came from probe difficulty
- large occlusion is the cleanest fix so far
- adding a small amount of decision noise pushes identification lower without destroying generalization
- this creates a much better entry point for the next DAM pass than the old ceilinged setup

Next step:
- run a narrow DAM comparison on the strongest non-ceiling regime first:
  - `animals`
  - ViT
  - `layer_11`
  - `occ0.5`
  - with and without a small decision-noise level

## 2026-05-28 - Hard-identification DAM hypothesis test

Hypothesis:
- once identification is no longer trivial, DAM should still help most clearly on the strongest `animals` late-layer ViT branch
- `fruits` should remain the weaker control under the same hard-identification setup

Fixed regime:
- model: `vit_base_patch16_224`
- pooling: `cls`
- layer: `layer_11`
- corruption: `occlusion`
- `occlusion_frac = 0.5`
- decision-noise levels:
  - `0.0`
  - `0.01`

Why this test matters:
- it directly addresses the earlier ceiling problem in identification
- it turns the next step into a concrete hypothesis test instead of another broad exploratory run

Main results:
- `animals`, `occ0.5`, `decision_noise_std = 0.0`
  - baseline identification: `92.5%`
  - baseline generalization: `56.25%`
  - DAM identification: `97.5%`
  - DAM generalization: `68.75%`
  - generalization delta: `+12.5%`
  - margin delta: `+0.118`
  - regret delta: `-0.0659`
  - probe-RDM delta: `+0.152`
  - this is a qualifying hard-ident win
- `animals`, `occ0.5`, `decision_noise_std = 0.01`
  - baseline identification: `85.0%`
  - baseline generalization: `47.5%`
  - DAM identification: `97.5%`
  - DAM generalization: `65.0%`
  - generalization delta: `+17.5%`
  - margin delta: `+0.121`
  - regret delta: `-0.0888`
  - probe-RDM delta: `+0.149`
  - this is also a qualifying hard-ident win
- `fruits`, same setup
  - DAM can improve generalization accuracy
  - but it worsens generalization margin on the best rows
  - so it still does not produce a clean win under the hard-ident criterion

Short take:
- breaking the identification ceiling did not erase the positive `animals` result
- if anything, it made the category contrast clearer
- `animals` remains the strong positive case for DAM
- `fruits` remains the weaker control where raw accuracy can go up but the retrieval quality stays less clean

Next step:
- use this hard-identification regime as the default DAM testbed for any remaining focused comparisons
- if only one branch is explored further, keep it on:
  - `animals`
  - `vit_base_patch16_224`
  - `cls`
  - `layer_11`
  - `occ0.5`

## 2026-05-28 - Tradeoff search under hard partial cues

Question:
- under partial-input probes, does stronger pattern completion improve retrieval at the cost of cue fidelity?

What was added:
- cue-fidelity metrics for DAM generalization probes:
  - `cue_recovered_cosine_mean`
  - `cue_displacement_mean`
  - `target_pull_gain_mean`
- aggregated config summaries over splits, not just single best rows
- a capped hard-ident tradeoff sweep under:
  - `animals`
  - `fruits`
  - `vit_base_patch16_224`
  - `cls`
  - `layer_11`
  - `occ0.5`
  - `decision_noise_std in {0.0, 0.01}`

Main finding:
- the expected retrieval-vs-cue-fidelity tradeoff did **not** appear on the strongest `animals` branch
- the best `animals` gains came from **small cue displacement**, not large cue overwriting

Key aggregated results from `animals`, `decision_noise_std = 0.01`:
- best retrieval config:
  - `n=2, beta=0.02, alpha=0.1, lmbda=0.05, steps=2`
  - mean baseline generalization: `51.5%`
  - mean DAM generalization: `59.375%`
  - mean generalization delta: `+7.875%`
  - mean margin delta: `+0.0910`
  - mean probe-RDM delta: `+0.1447`
  - mean cue displacement: only `0.0093`
- top-retrieval animal configs overall:
  - mean cue displacement stayed very small, around `0.002` to `0.009`
  - mean qualifying fraction was high, around `0.8`
- higher-displacement animal configs:
  - cue displacement rose to `0.028` to `0.054`
  - retrieval deltas were lower, typically around `+4.5%` to `+6.4%`

What this means:
- stronger completion did not help by aggressively overwriting the cue
- the best animal gains came from **gentle stabilization**
- in other words, DAM behaves more like a cue-preserving denoiser than a hard attractor snap in the useful regime

Control result on `fruits`:
- `fruits` still failed to produce a clean regime
- best mean generalization delta under `decision_noise_std = 0.01` was only `+2.5%`
- mean generalization margin stayed negative on the best fruits rows, around `-0.047` to `-0.059`
- so raw accuracy can improve a bit, but retrieval quality remains fragile

Supported conclusion:
- the original completion-vs-fidelity tradeoff is weak on the strongest positive branch
- the stronger supported hypothesis is:
  - **useful DAM gains come from gentle cue-preserving stabilization, not strong cue overwriting**
- `animals` supports this regime
- `fruits` does not

Why this is a better story than the original tradeoff:
- it is grounded directly in the new cue-fidelity metrics
- it explains why the strongest gains happen at low `beta` and short step counts
- it is consistent with the current category split:
  - `animals` has a recoverable cue structure that mild completion can stabilize
  - `fruits` remains too fragile for the same mechanism to produce clean gains

Next step:
- if only one final branch is explored, stay on:
  - `animals / vit_base_patch16_224 / cls / layer_11 / occ0.5`
- vary only the mild-completion regime around the winning configs:
  - very low `beta`
  - short step counts
  - `n=2`
- do not widen to stronger completion unless the explicit goal is to show failure.
