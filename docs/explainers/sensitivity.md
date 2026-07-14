# Parameter Sensitivity

## The question it answers

You chose a specific parameter setting — a specific fast/slow window pair,
a specific lookback, a specific threshold. Is that choice sitting on a
broad plateau of the parameter space, where nearby settings perform
similarly well (a real, robust effect)? Or is it an isolated spike, where
the immediate neighbors perform much worse (a setting that got lucky on
this particular history and would likely disappoint under slightly
different parameters, or slightly different data)?

## The math, in plain English

Each trial is a point in parameter space (one coordinate per swept
parameter) with an associated performance metric (periodic Sharpe by
default). skepsis standardizes each parameter column to a z-score,
\(z = (p - \bar p) / \hat\sigma_p\) (constant columns get \(\hat\sigma_p =
1\) so they contribute zero distance), then finds the chosen
configuration's \(k\) nearest neighbors by Euclidean distance in this
z-scored space, with

$$
k = \min(2d,\, n-1)
$$

where \(d\) is the number of parameters and \(n\) is the number of trials.
For an interior point of a regular 2D grid this recovers exactly the four
orthogonal grid neighbors (\(k = 2 \times 2 = 4\)); the same rule extends
naturally to 1D grids (\(k=2\): the immediate left and right neighbors) and
to irregular or non-grid trial sets, where "neighbor" is defined purely by
distance rather than grid adjacency.

The stability score is the chosen configuration's metric relative to the
median of its neighbors:

$$
\text{stability} = \frac{\text{metric}_{\text{chosen}}}{\operatorname{median}(\text{metric}_{\text{neighbors}})}
$$

A score near 1.0 means the chosen point performs about the same as its
neighbors: a plateau. A score well above 1.0 means the chosen point
dramatically outperforms its neighborhood: an isolated spike, the
fingerprint of a parameter that was fit to noise rather than found because
it captures something real. Two edge cases get special handling rather than
producing a misleading ratio: a non-positive chosen metric makes the ratio
meaningless (returned as `nan`, with a warning), and a non-positive neighbor
median against a positive chosen metric is scored as an extreme spike
(`inf`, flagged) rather than a division degenerating silently.

## When it lies

**Grid resolution changes what "neighbor" means.** A coarse sweep (say,
five values per parameter, widely spaced) makes the nearest neighbors quite
far away in absolute parameter terms; two points that are "adjacent" on a
coarse grid can behave very differently in reality, so a spike detected
against coarse neighbors is a weaker signal than the same spike detected
against a fine grid. Conversely, a very fine grid can make near-duplicate
parameter settings count as separate "neighbors," diluting genuine
structure.

**Z-scoring erases the actual economic scale of each parameter.** Distance
in z-scored space treats a one-standard-deviation step in every parameter
as equally "close," regardless of what that step means in practice. A
sweep of `fast ∈ {5..50}` and `slow ∈ {20..200}` might have very different
sensitivity per unit of each parameter — a 5-day change in a fast window
can flip a signal's timing far more than a 5-day change in a slow one — but
z-scoring doesn't know that; it only knows relative spread within each
column.

**The score is unweighted and doesn't know about grid holes.** All \(k\)
neighbors count equally regardless of how much closer one is than another,
and if the trial set is sparse or irregular, "nearest by Euclidean
distance" can pull in points that aren't meaningfully adjacent at all.
With few trials (skepsis requires at least 4), the neighbor median is a
noisy estimate, and scores near the fail/warn thresholds can flip with the
inclusion or exclusion of a single trial.

**A smooth plateau is necessary, not sufficient, for a real edge.** A
stability score near 1.0 rules out the "isolated lucky spike" failure mode,
but a broad, robust plateau of parameters that are all *equally overfit* to
the same spurious pattern in one dataset will still score well here. Pair
this with DSR and PBO, which look at the same overfitting question from
different angles.

## Worked example

In the "Deflating the Golden Cross" demo, `params_df` carries the
(fast, slow) window pair for each of the 34 trials, so sensitivity runs on
a 2D grid (\(k=4\)).

**Act 1 — raw returns.** The chosen configuration, `f5_s100`, scores a
stability of roughly **1.13** — close to 1.0, unflagged. Its neighbors in
(fast, slow) space perform similarly, which fits Act 1's broader story:
with returns dominated by BTC's overall drift, most parameter choices in
this family produce roughly similar Sharpes, so the surface is a plateau,
not a spike.

**Act 2 — excess over buy-and-hold.** The picture flips completely. The
chosen configuration's neighbors have a *negative* median excess Sharpe
(around −0.13), while the chosen point itself is barely positive (0.084).
That combination — a positive chosen metric surrounded by a negative
neighborhood median — is scored as an extreme isolated spike: stability
**∞**, flagged, and one of the two failing rules that push Act 2's verdict
to **LIKELY_OVERFIT**. The "best" fast/slow pair for beating buy-and-hold
is not sitting on any kind of robust structure; it is a single lucky point
surrounded by parameter settings that would have lost you money relative
to just holding.

## References

- Sharpe, W. F. (1994). "The Sharpe Ratio." *Journal of Portfolio
  Management*, 21(1). (The default per-trial metric.) The neighborhood
  stability method itself is not drawn from a single external paper; it is
  skepsis's own formalization of the "is the peak a plateau or a spike"
  check that parameter sweeps are commonly eyeballed for informally.
