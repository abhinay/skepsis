# Probability of Backtest Overfitting

## The question it answers

Say you swept a grid of parameters and picked the best in-sample performer.
How much of that "best" is a real, generalizable edge, and how much is the
in-sample winner simply being the trial that happened to fit that
particular slice of history? The Probability of Backtest Overfitting (PBO),
via the Combinatorially Symmetric Cross-Validation (CSCV) procedure, answers
a very specific, empirical version of this: across every plausible way of
splitting your history into an in-sample half and an out-of-sample half,
how often does the strategy you'd have picked in-sample turn out to be a
below-median performer out-of-sample?

## The math, in plain English

Start with the \(T \times N\) matrix of trial returns: \(T\) periods, \(N\)
trial variants as columns. Split the \(T\) rows into \(S\) equal, contiguous
time blocks (\(S\) even, default 16). Now consider every way of choosing
\(S/2\) of those blocks to serve as the in-sample (IS) set, leaving the
other \(S/2\) as out-of-sample (OOS) — there are \(\binom{S}{S/2}\) such
combinations (12,870 of them at the default \(S=16\)). For each combination:

1. Rank the \(N\) trials by their in-sample performance (default metric:
   periodic Sharpe) and find the IS winner, trial \(n^*\).
2. Compute the same trials' OOS performance, and find \(n^*\)'s relative
   rank \(\omega\) among them — a value in \((0, 1)\), where 1 means
   \(n^*\) was also the best OOS and near 0 means it was near-worst.
3. Convert to a logit: \(\lambda = \ln\!\left(\dfrac{\omega}{1-\omega}\right)\).

$$
\mathrm{PBO} = \frac{1}{\binom{S}{S/2}} \sum_{c} \mathbb{1}[\lambda_c \le 0]
$$

\(\lambda_c \le 0\) means \(\omega \le 0.5\): the in-sample winner landed in
the *bottom half* out-of-sample for that split. PBO is simply the fraction
of all splits where that happens. If parameter selection carried no real
information, PBO should sit near 0.5 — the in-sample winner is no better
than a coin flip at holding up out-of-sample. A low PBO (well under 0.5)
means the in-sample winner tends to keep winning OOS too — real,
generalizing structure. A high PBO means the opposite: your selection
process is closer to picking noise.

## When it lies

**PBO is only as honest as the trial set you feed it.** The whole method
works by re-ranking trials across resampled splits, so it needs the *actual*
population of variants you considered — every parameter combination you
swept, not just a curated shortlist of "the ones that looked promising."
Feed it three cherry-picked survivors instead of the full 34-trial grid and
PBO will report on the overfitting risk of *that* three-way selection, which
tells you nothing about the real search you ran. Garbage in, garbage out is
not a caveat here, it's the entire failure mode: skepsis cannot detect that
your `trials=` argument is missing half your search.

**It needs a real time-block split, and enough data to make one.** CSCV
requires at least \(2S\) observations so each of the \(S\) blocks has at
least two rows; skepsis raises rather than guessing at a smaller default.
If \(T\) isn't evenly divisible by \(S\), the tail is trimmed and a warning
is raised — a handful of observations are silently excluded from every
block, which matters more the shorter your series is.

**Block boundaries are chronological, not regime-aware.** The \(S\) blocks
are equal-sized contiguous chunks of calendar time, not equal-sized chunks
of "market regime." If your strategy's edge, or lack of one, is
concentrated in one specific regime (a single volatile quarter, say), how
that regime happens to fall across the block boundaries can move PBO around
in ways that have more to do with calendar alignment than with genuine
overfitting.

**A good PBO is not proof of skill, and a bad one is not proof of its
absence.** PBO measures whether the selection *procedure* generalizes; it
says nothing about whether the returns beat a sensible benchmark (that's
what excess-return framing plus DSR is for) or whether the edge is
economically meaningful after costs. Read it alongside DSR and the
sensitivity map, not instead of them.

## Worked example

In the "Deflating the Golden Cross" demo, the same 34-trial BTC moving
average sweep is evaluated with CSCV at the default \(S=16\) (12,870
combinations) in both acts.

**Act 1 — raw returns.** PBO comes back **0.448**: close to the 0.5
coin-flip line. The in-sample Sharpe winner, `f5_s100`, lands in the bottom
half out-of-sample on nearly half of all splits. This is the tell, even
before deflation: the parameter choice carries almost no real selection
information, it's close to arbitrary which of the 34 "wins." Combined with
the fact that raw DSR survives (see the
[Deflated Sharpe Ratio explainer](deflated-sharpe.md)), PBO ≈ 0.448 is what
correctly keeps Act 1's verdict at **MODERATE** rather than **STRONG** — it
warns (PBO's warn threshold is 0.2) even while the headline Sharpe looks
fine.

**Act 2 — excess over buy-and-hold.** PBO is **0.405** on the excess-return
version of the same sweep — still comfortably above the 0.2 warn threshold,
still telling the same story: picking the "best" excess-return variant
in-sample is a weak predictor of which variant will do best out-of-sample.
Paired with a failing DSR (0.345) and an isolated-spike sensitivity score,
the full picture pushes the verdict to **LIKELY_OVERFIT**.

## References

- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2015).
  "The Probability of Backtest Overfitting." *Journal of Computational
  Finance*.
- Sharpe, W. F. (1994). "The Sharpe Ratio." *Journal of Portfolio
  Management*, 21(1). (The default in-sample/out-of-sample ranking metric.)
