# Vision Retrieval Project Report

## Goal

The main goal of this project was to test whether a vision-based memory pipeline could show a meaningful tradeoff between **identification** and **generalization**, and then test whether **Dense Associative Memory (DAM)** could improve retrieval on top of that baseline.

In simple terms:

- **Identification** asks: if I store an image, can I retrieve that exact image from a noisy version of it?
- **Generalization** asks: if I show a new image, can I retrieve the stored image that is most similar to it?

The original hope was to find a curve where generalization improves at first, then drops, while identification continues to improve.

## What We Built

The work ended up in four main experiment stages:

1. **Synthetic vision baseline** using colored squares.
   Notebook: [VisionLayerwiseBaseline.ipynb](/Volumes/youwhat/projects/memgrid/VisionLayerwiseBaseline.ipynb:1)
2. **Synthetic model comparison** across several non-language vision backbones.
   Notebook: [VisionModelComparison.ipynb](/Volumes/youwhat/projects/memgrid/VisionModelComparison.ipynb:1)
3. **Naturalistic baseline** using the Peterson similarity dataset for fruits, vegetables, and animals.
   Notebook: [VisionNaturalisticBaseline.ipynb](/Volumes/youwhat/projects/memgrid/VisionNaturalisticBaseline.ipynb:1)
4. **Naturalistic DAM benchmark** plus a matched ViT-vs-CLIP comparison and a focused animals frontier analysis.
   Notebook: [VisionNaturalisticDAM.ipynb](/Volumes/youwhat/projects/memgrid/VisionNaturalisticDAM.ipynb:1)

The naturalistic benchmark became the strongest part of the project because it uses real images together with human similarity judgments.

## Synthetic Baseline: Useful, but Not the Main Result

The synthetic setup used simple images of colored squares at different positions. This stage was useful because it let us debug the pipeline and make sure the basic evaluation logic was correct.

One important issue was found early: the low identification similarity was mostly caused by **z-score preprocessing** on a very small clean synthetic dataset. After switching the default preprocessing to **L2-only normalization**, the sanity checks behaved correctly.

Main synthetic takeaways:

- The pipeline itself was not broken.
- The synthetic task did **not** show the desired identification/generalization tradeoff.
- Color was easy for the models.
- Position was weaker and less reliably represented.
- Changing the backbone alone did not fix the problem.

Even after testing models like ViT, DeiT, ConvNeXt, ResNet, and later CLIP on the synthetic benchmark, the main picture stayed the same: this setup did not give a convincing tradeoff story.

That is why the project moved to naturalistic images.

## Naturalistic Baseline: Main Scientific Result

The naturalistic benchmark used the Peterson dataset, which includes:

- real images from three categories: **fruits, vegetables, animals**
- a human similarity matrix for each category

This made the generalization task much more meaningful.

For these experiments:

- **Identification** used noisy versions of stored images as probes.
- **Generalization** used held-out real images as probes.
- The correct generalization target was defined as the stored image with the **highest human-rated similarity** to the probe.

### What Happened

Across all three categories, a clear pattern appeared:

- **Late layers** of the vision model were best for retrieval accuracy.
- **Mid layers** were best for alignment with human similarity structure.

This pattern was much stronger and more stable than anything found in the synthetic task.

### Best Baseline Generalization by Category

- **Fruits:** `48.3%`
- **Vegetables:** `49.6%`
- **Animals:** `63.0%`

Animals was the strongest retrieval category by a large margin.

### Best Human-Alignment Scores

These were measured with `human_rdm_spearman`, which asks how well the model’s representational geometry matches the human similarity geometry.

- **Fruits:** about `0.379`
- **Vegetables:** about `0.462`
- **Animals:** about `0.423`

So the baseline result was already interesting even before DAM:

- retrieval and alignment do not peak at the same layer
- late layers retrieve best
- mid layers align best to human judgments

This became the main result that motivated the DAM phase.

## CLIP Comparison

CLIP was then added as an **image encoder only**. No text prompts or zero-shot classification were used. The goal was to see whether CLIP changed the baseline picture.

Main CLIP result:

- **CLIP improved human alignment strongly**
- but **base ViT still gave the strongest top retrieval rows**

Best CLIP alignment rows:

- **Fruits:** `0.580`
- **Vegetables:** `0.494`
- **Animals:** `0.550`

So CLIP was especially useful as an alignment model. However, it did not replace the original ViT as the strongest retrieval backbone.

The most important interpretation here is:

- **ViT** was the stronger retrieval substrate
- **CLIP** was the stronger alignment substrate

That made both of them important candidates for DAM.

## DAM Results

The DAM experiments were run on top of the naturalistic benchmark, using the same task definitions as the baseline.

The key question was not just whether DAM helps, but whether it can improve retrieval **without destroying the human-similarity structure**.

### Category-Level DAM Results

- **Fruits:** no clean DAM win
- **Vegetables:** modest clean DAM gain
- **Animals:** strongest clean DAM gain

Best animals DAM result:

- baseline generalization: `56.25%`
- DAM generalization: `67.5%`

This was the strongest clear improvement in the project.

### Main Insight From DAM

The original identification-vs-generalization tradeoff still did **not** appear.

Instead, the project found a more realistic tension:

- sometimes DAM improves retrieval
- but in some categories it can also distort the human-like geometry of the representation

This was especially clear in **fruits**, where retrieval could improve but the geometry often became less aligned with human similarity.

In **animals**, the story was much better: useful DAM settings improved both retrieval and alignment at the same time.

## ViT vs CLIP Under DAM

A matched DAM comparison was then run between ViT and CLIP on the same anchor branches.

Main result:

- **ViT is currently the better DAM substrate**

Why:

- On **animals**, ViT gave a stronger clean DAM win than CLIP.
- On **vegetables**, ViT also owned the clean DAM win.
- On **fruits**, neither encoder gave a clean DAM win.

So even though CLIP helped the baseline alignment story, it did not become the best encoder once DAM was added.

## Current Takeaway

At this point, the clearest summary is:

1. The original synthetic benchmark was useful for debugging, but it did not produce the target tradeoff.
2. The naturalistic benchmark was much more informative.
3. On naturalistic data, there is a stable split:
   - late layers are best for retrieval
   - mid layers are best for human alignment
4. CLIP improves alignment, but base ViT remains the stronger retrieval backbone.
5. DAM can help, but the effect is category-dependent:
   - strongest on animals
   - modest on vegetables
   - fragile on fruits

Most importantly:

**We did not recover the original identification-vs-generalization tradeoff.**

But we did find a more defensible and interesting result:

**a retrieval-vs-alignment story on naturalistic images, with DAM improving retrieval most clearly on animals.**

## What the Frontier Analysis Suggests

The latest focused DAM frontier analysis fixed attention on the strongest branch:

- `animals / vit_base_patch16_224 / cls / layer_11`

It also used:

- `fruits / vit_base_patch16_224 / cls / layer_11`

as the failure-control branch.

The current capped frontier pass suggests:

- **Animals** looks like a `no_tradeoff` branch so far:
  useful DAM settings improve retrieval and alignment together.
- **Fruits** looks like a `no_useful_dam_regime` branch:
  retrieval can improve, but not in a clean, stable way.

This means the next important question is no longer “does DAM ever work?” because it clearly can. The better question is:

**Why does animals support a useful DAM regime while fruits stays fragile?**

## Two-Minute Presentation Version

We started with a synthetic vision task, mainly to debug the pipeline, but it never gave the identification-generalization tradeoff we were hoping for. After fixing a preprocessing issue, the synthetic results were still too limited: color was easy, position was weak, and changing the backbone did not solve that. The project became much stronger once we moved to naturalistic images with human similarity judgments. There, a stable pattern emerged across fruits, vegetables, and animals: late layers retrieve best, while mid layers align best with human similarity. CLIP improved alignment a lot, but the original ViT stayed better for top retrieval. When we added DAM, the strongest result appeared on animals, where generalization improved from `56.25%` to `67.5%`. Fruits remained fragile, and vegetables showed smaller clean gains. So the current story is not the original identification-generalization tradeoff, but a retrieval-vs-alignment story on naturalistic images, with animals as the strongest case for useful DAM behavior.
