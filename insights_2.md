# Vision DAM Low-Similarity Diagnostic: Insights

## Executive Summary

The low similarity values are not primarily a metric bug and not primarily a task-definition bug. The notebook outputs show that the **raw vision features already carry useful task signal**, but the current DAM retrieval dynamics are collapsing the query state toward the origin and destroying that signal.

The strongest evidence is the retrieved-state norm:

- `avg_probe_norm` is approximately `1.0`
- `avg_target_norm` is approximately `1.0`
- `avg_retrieved_norm` is approximately `0.0003`

That means the retrieval process is not refining the probe into a better attractor. It is shrinking it into a near-zero vector. Once that happens, cosine to the target necessarily becomes very small.

This is consistent with the current DAM settings being poorly matched to the scale and geometry of the ViT feature vectors.

---

## Which Run This Refers To

The analysis below refers to:

- [results/vision/vision_dam_summary_vit_base_patch16_224.json](/Volumes/youwhat/projects/memgrid/results/vision/vision_dam_summary_vit_base_patch16_224.json:1)
- [results/vision/vision_dam_metadata_vit_base_patch16_224.json](/Volumes/youwhat/projects/memgrid/results/vision/vision_dam_metadata_vit_base_patch16_224.json:1)

Notebook configuration recorded in the summary:

- `task_mode = "mixed_color_position"`
- `preprocess_mode = "zscore_l2"`
- `metric = "cosine"`

This means:

- synthetic squares varied in both color and position,
- features were z-scored and L2-normalized,
- winner selection and similarity reporting used cosine,
- identification probes came from perturbed images,
- generalization probes were held-out stimuli scored against metadata-defined neighbors.

---

## Variable Glossary

The notebook and summary JSON use a mix of task, preprocessing, retrieval, and
diagnostic variables. This section defines the important ones.

### Task and data variables

- `stored_features`: the feature vectors stored in memory for each layer.
- `probe_features`: the query feature vectors sent into retrieval.
- `ground_truth_idx`: for each probe, the index of the correct stored target.
- `layer_0`, `layer_6`, `layer_11`: selected feature layers from the vision
  model. Lower numbers are earlier layers; higher numbers are later layers.
- `task_mode`:
  - `mixed_color_position`: both color and position vary.
  - `color_only_fixed_position`: position is fixed; color carries the task.
  - `position_only`: color is ignored in task truth.
- `preprocess_mode`:
  - `zscore_l2`: z-score each layer, then L2-normalize vectors.
  - `l2_only`: only L2-normalize.
  - `zscore_only`: only z-score.
  - `raw`: no z-score and no L2 normalization.

### DAM hyperparameters

- `DAM_ORDER` or `n`: order of the dense associative memory energy function.
  Larger values typically make the memory landscape sharper.
- `DAM_BETA` or `beta`: gain on the local field before the `tanh` nonlinearity.
  Small values make updates weak; large values make updates stronger.
- `DAM_ALPHA` or `alpha`: update mixing coefficient in the retrieval step.
  Higher values move the state more aggressively each step.
- `DAM_LAMBDA` or `lmbda`: regularization term used by the DAM update.
- `steps_multiplier`: number of retrieval updates scaled by feature dimension.
  Actual steps are `steps_multiplier * feature_dim`.
- `retrieval_backend`: implementation used for retrieval, such as `numba` or
  `numpy`.
- `normalize_retrieved`: whether the final retrieved vector is normalized before
  scoring it against stored vectors.

### Scoring and baseline variables

- `metric`: the similarity rule used to choose the nearest stored item.
  Supported values are `cosine`, `dot`, and `euclidean`.
- `baseline_accuracy`: accuracy from direct nearest-neighbor lookup on the raw
  probe features, without DAM retrieval.
- `dam_accuracy`: accuracy after running the DAM retrieval dynamics first.
- `baseline_delta_accuracy`: `dam_accuracy - baseline_accuracy`.
  Positive values mean DAM improved accuracy; negative values mean it hurt.
- `baseline_avg_sim`: average target similarity for the no-DAM baseline.

### Similarity diagnostics

- `avg_initial_sim`: average cosine similarity between the incoming probe and
  the correct stored target before retrieval.
- `avg_final_sim` or `avg_sim`: average cosine similarity between the retrieved
  state and the correct stored target after retrieval.
- `avg_delta_sim`: `avg_final_sim - avg_initial_sim`.
  Positive values mean retrieval moved the state toward the correct target in
  direction; negative values mean it moved away.
- `std_sim`: standard deviation of final target similarities across probes.

### Norm diagnostics

- `avg_probe_norm`: average norm of the probe vectors before retrieval.
- `avg_target_norm`: average norm of the correct stored target vectors.
- `avg_retrieved_norm`: average norm of the final retrieved states.
- `retrieved_norm_ratio`: `avg_retrieved_norm / avg_target_norm`.
  This is the main scale-comparison diagnostic.

### Retrieval health labels

- `retrieval_collapse`: true when `avg_retrieved_norm` falls below the collapse
  threshold. This means retrieval is shrinking the state toward zero.
- `collapse_threshold`: the numeric threshold used for the collapse label.
- `retrieval_overshoot`: true when `retrieved_norm_ratio` exceeds the overshoot
  threshold. This means retrieval is expanding the state far beyond the target
  scale.
- `overshoot_ratio`: the numeric threshold used for the overshoot label.
- `retrieval_calibrated`: true when the run is neither collapsed nor
  overshooting.

### Sweep outputs

- `identification_sweep`: all sweep rows for the identification task.
- `generalization_sweep`: all sweep rows for the generalization task.
- `identification_sweep_top_calibrated`: top-ranked identification rows after
  prioritizing calibrated retrieval.
- `generalization_sweep_top_calibrated`: top-ranked generalization rows after
  prioritizing calibrated retrieval.
- `identification_sweep_top_accuracy`: calibrated identification rows with
  positive `baseline_delta_accuracy`.
- `generalization_sweep_top_accuracy`: calibrated generalization rows with
  positive `baseline_delta_accuracy`.
- `winner_agreement`: how often the DAM winner matches the no-DAM baseline
  winner.

### Metadata-based task truth

- `color_weight`: how much color contributes to synthetic generalization truth.
- `position_weight`: how much position contributes to synthetic generalization
  truth.
- metadata distance: the handcrafted distance over color and position used to
  define the correct stored neighbor for synthetic generalization.

---

## What The Outputs Say

## 1. The raw feature space is not hopeless

The no-DAM baseline already performs meaningfully above chance in several cases.

### Identification baseline

- `layer_0`: `66.7%`
- `layer_6`: `16.7%`
- `layer_11`: `8.3%`

### Generalization baseline

- `layer_0`: `41.7%`
- `layer_6`: `41.7%`
- `layer_11`: `50.0%`

This matters because it tells us the features themselves are usable:

- early layer features (`layer_0`) separate perturbed stored items reasonably well,
- late layer features (`layer_11`) support the best generalization baseline,
- the representation already contains a meaningful early-vs-late pattern.

So the current low similarities should not be blamed on “ViT features are useless” or “the task is impossible.”

---

## 2. DAM retrieval usually makes the results worse

Comparing baseline vs DAM:

### Identification

- `layer_0`: `66.7% -> 33.3%`
- `layer_6`: `16.7% -> 8.3%`
- `layer_11`: `8.3% -> 16.7%`

### Generalization

- `layer_0`: `41.7% -> 16.7%`
- `layer_6`: `41.7% -> 33.3%`
- `layer_11`: `50.0% -> 50.0%`

This is not what an effective attractor memory should do. If retrieval were helping, we would expect:

- better winner accuracy than baseline, or at least no consistent degradation,
- higher cosine to the ground-truth target after retrieval than before retrieval.

Instead, the summary shows the opposite.

---

## 3. Similarity drops sharply during retrieval

The diagnostic fields added to the runner show that the model starts from a better state than it ends with.

### Identification

- `layer_0`: `avg_initial_sim = 0.146`, `avg_sim = 0.034`, `avg_delta_sim = -0.112`
- `layer_11`: `avg_initial_sim = 0.127`, `avg_sim = 0.008`, `avg_delta_sim = -0.118`
- `layer_6`: `avg_initial_sim = 0.097`, `avg_sim = -0.005`, `avg_delta_sim = -0.102`

### Generalization

- `layer_0`: `avg_initial_sim = 0.419`, `avg_sim = 0.067`, `avg_delta_sim = -0.351`
- `layer_11`: `avg_initial_sim = 0.617`, `avg_sim = 0.114`, `avg_delta_sim = -0.503`
- `layer_6`: `avg_initial_sim = 0.414`, `avg_sim = 0.091`, `avg_delta_sim = -0.322`

This is the clearest sign that retrieval is actively damaging the query.

The query enters the DAM with a moderate or strong target alignment and exits with much weaker alignment.

---

## 4. The state norm is collapsing

This is the strongest diagnostic result.

For every layer and both tasks:

- probe norm is approximately `1`
- target norm is approximately `1`
- retrieved norm is approximately `0.0003`

That means the state update is shrinking the vector almost to zero.

This alone explains much of the observed behavior:

- a near-zero vector has weak cosine relationship to any target,
- nearest-neighbor winner selection becomes unstable,
- winner agreement with the baseline drops,
- even when accuracy does not collapse completely, the retrieved representation is no longer meaningful in the original feature geometry.

The issue is therefore not just “cosine values happen to be low.” The retrieved object itself is degenerate.

---

## Why This Is Happening

## 1. Current DAM settings are too weak for this feature scale

The notebook used:

- `DAM_BETA = 0.01`
- `DAM_ALPHA = 0.5`
- `DAM_LAMBDA = 0.0`

The probes are L2-normalized to norm `1`, and the targets are also norm `1`.

With such a small `beta`, the `tanh(beta * local_field)` response is extremely small unless local fields are very large. Combined with repeated relaxation:

```python
s_new = (1 - alpha) * s_i + alpha * tanh(beta * local_field)
```

the update can repeatedly contract the state toward a small-magnitude fixed point rather than pulling it toward a stored attractor.

The observed retrieved norms strongly suggest that this is exactly what is happening.

---

## 2. `DenseAM.ipynb` already hinted at this failure mode

`DenseAM.ipynb` includes a clear warning: for continuous-valued patterns, DAM dynamics can drive states toward undesirable regimes unless scale and regularization are chosen appropriately.

That notebook focused on cases where:

- the model sharpens or distorts continuous vectors,
- regularization can matter,
- good performance often depends on the relation between pattern scale, update gain, and the shape of the attractor landscape.

The current vision setup is a textbook case where the representation geometry is very different from binary or grid-coded patterns, so reusing the old low-beta defaults without retuning is unlikely to work.

---

## 3. The metric is not the first problem

It is natural to suspect cosine because the reported similarities are so low. But the outputs do not support that as the primary explanation.

Reasons:

- the repository convention in [experiments.py](/Volumes/youwhat/projects/memgrid/experiments.py:1) already uses cosine for winner selection and target similarity,
- the no-DAM baseline using cosine still extracts useful structure,
- with L2-normalized features, cosine, dot product, and Euclidean ranking are often closely related,
- a vector with norm `0.0003` will score poorly under almost any target-comparison metric.

So changing the metric may still be useful diagnostically, but it is not the main fix.

---

## What These Results Imply

## 1. There is a real representational signal

Ignoring DAM for a moment:

- `layer_0` is strongest for identification,
- `layer_11` is strongest for generalization.

That is already an interesting result. It suggests the vision representation may show a genuine early-vs-late tradeoff pattern in the raw feature space.

## 2. DAM is currently obscuring that signal

The present retrieval dynamics are not revealing a deeper memory effect. They are suppressing the useful structure already present in the features.

## 3. The next tuning target should be state magnitude preservation

Before interpreting any accuracy pattern, the first retrieval-level goal should be:

- keep `avg_retrieved_norm` in a reasonable range,
- ideally near the scale of the probes and stored patterns,
- then test whether retrieval improves or degrades target similarity.

If the retrieved state still collapses, the rest of the analysis is downstream noise.

---

## Recommended Next Experiments

## 1. Increase `DAM_BETA` substantially

Start with:

- `0.1`
- `0.5`
- `1.0`
- `2.0`

The immediate readout to watch is `avg_retrieved_norm`.

## 2. Lower `DAM_ALPHA`

Try:

- `0.1`
- `0.2`

This may reduce aggressive contraction from the repeated relaxation step.

## 3. Compare preprocessing modes

Run the notebook with:

- `zscore_l2`
- `l2_only`
- `zscore_only`
- `raw`

The current `zscore_l2` pipeline may be creating a feature scale that is poorly matched to the DAM dynamics.

## 4. Keep cosine initially

Do not change the main scoring metric yet. First determine whether the retrieval state can be kept off the origin.

## 5. Use fixed-position color-only as a sanity check

This should be treated as a diagnostic:

- it can tell us whether the representation cleanly separates color under
  simpler conditions,
- but it should not be used as the main conclusion for the mixed synthetic
  task.

---

## Calibration Sweep Follow-Up

After the collapse diagnosis, the notebook was updated to distinguish three
retrieval regimes:

- `retrieval_collapse`
- `retrieval_overshoot`
- `retrieval_calibrated`

The refined sweep narrowed the search to the healthier parameter region:

- `beta = [1.0, 1.5, 2.0, 2.5]`
- `alpha = [0.05, 0.1, 0.2]`
- `lmbda = [0.0, 0.01, 0.05, 0.1]`
- `steps_multiplier = [5, 10, 20]`

The goal was no longer just “avoid collapse.” The goal was:

- keep retrieval numerically stable,
- keep `avg_retrieved_norm` near the target scale,
- and preserve the generalization gains seen in the earlier broad sweep.

---

## What The Calibrated Sweep Showed

## 1. The broad-sweep gains depended on overshoot

The earlier broad sweep found large generalization improvements, but those
best-performing rows had very large retrieved norms, often in the low tens up
to the mid-20s.

That meant the improvement was happening in an overshooting regime rather than
in a retrieval regime that stayed close to the original feature manifold.

The calibrated follow-up confirmed this directly.

---

## 2. Calibrated identification improved similarity, but not the main baseline

Top calibrated identification rows were mostly on `layer_0`.

Representative result:

- `beta = 2.5`
- `alpha = 0.1`
- `lmbda = 0.0`
- `steps_multiplier = 20`
- `dam_accuracy = 58.3%`
- `baseline_accuracy = 66.7%`
- `avg_delta_sim = +0.0906`
- `retrieved_norm_ratio = 0.155`

This means:

- DAM can improve target similarity while staying numerically stable,
- but that improvement does not automatically produce better winner accuracy,
- and the strongest identification baseline is still the no-DAM `layer_0`
  baseline.

There was also one calibrated `layer_0` configuration that tied the baseline:

- `66.7% -> 66.7%`

but it still did not improve on it.

Small calibrated identification gains did appear for `layer_6`:

- `16.7% -> 25.0%`

but these are modest and not on the strongest identification layer.

---

## 3. Calibrated generalization did not improve accuracy

This is the most important result of the refined sweep.

Top calibrated generalization rows looked like:

- `layer_6`
- `beta = 1.5`
- `alpha = 0.05`
- `lmbda = 0.0`
- `steps_multiplier = 5`
- `baseline_accuracy = 41.7%`
- `dam_accuracy = 41.7%`
- `avg_delta_sim = +0.1201`
- `retrieved_norm_ratio = 1.99`

So again, similarity improved, but accuracy did not.

More importantly, the saved summary showed:

- `generalization_sweep_top_accuracy = []`

and a direct check across the full sweep confirmed:

- calibrated generalization rows: `24`
- calibrated generalization rows with positive `baseline_delta_accuracy`: `0`

So there are **no calibrated generalization configurations** in this refined
sweep that outperform the no-DAM baseline.

---

## 4. The central tension is now clear

The results across both sweeps suggest a specific tradeoff:

- if retrieval is too weak, the state collapses toward zero and performance
  degrades badly,
- if retrieval is strong enough to produce the large generalization gains, the
  state norm overshoots far beyond the probe/target scale,
- if retrieval is forced into a calibrated norm range, similarity can improve
  but the large generalization gains disappear.

That is a much more useful diagnosis than “similarities are low.”

The problem is now localized to the retrieval dynamics themselves.

---

## Updated Conclusion

The current DAM implementation can be made to avoid collapse, but under the
tested update rule it does not simultaneously achieve all three goals:

- stable norms,
- better identification than the strongest baseline,
- and better generalization than the strongest baseline.

The earlier apparent breakthrough in generalization was real in the sense that
accuracy improved, but it came from an overshooting regime rather than a
well-calibrated attractor regime.

So the next step should **not** be another round of notebook parameter sweeps
over the same update rule.

The next step should be to inspect and possibly modify `DAM.py` so retrieval can:

- preserve useful directionality from the probe,
- avoid collapsing to zero,
- and avoid exploding the state norm while still sharpening the representation.

In short:

- feature space is not the main problem,
- cosine is not the main problem,
- task definition is not the main problem,
- the main unresolved issue is the **shape of the DAM retrieval dynamics** for
  continuous normalized vision features.

---

## Recommended Pause State

If pausing here, the most honest summary is:

- the vision features already contain meaningful identification and
  generalization signal,
- broad DAM sweeps found large gains only in an overshooting regime,
- calibrated DAM sweeps removed that gain,
- the next serious step is a retrieval-rule investigation, not more notebook
  tuning.

- if the baseline is high there, the representation clearly separates color,
- if DAM still collapses there, that is further evidence the issue is in the dynamics, not task ambiguity.

---

## Bottom Line

The notebook outputs strongly support this interpretation:

1. The feature representations already contain useful task signal.
2. DAM retrieval is shrinking the state almost to zero.
3. That collapse causes the low final cosine similarities.
4. The first fix is to retune the retrieval dynamics, not to reinterpret the experiment or abandon cosine.

The single most informative next number to monitor is `avg_retrieved_norm`. If that returns to a reasonable scale after retuning `beta` and `alpha`, then similarity and accuracy should become interpretable again.
