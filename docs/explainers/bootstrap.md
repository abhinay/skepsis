# Block Bootstrap

## The question it answers

A single Sharpe ratio and a single max drawdown are point estimates from
one realized path of history. How much would those numbers have wobbled if
history had unfolded slightly differently, and — the sharper question — is
the observed performance actually distinguishable from what a strategy with
*zero* skill would have produced, once you resample in a way that respects
the fact that daily returns aren't independent draws?

## The math, in plain English

Ordinary (iid) bootstrapping resamples individual observations with
replacement, which destroys any autocorrelation in the original series —
fine for genuinely independent data, wrong for returns with momentum,
volatility clustering, or trend persistence. The stationary bootstrap
(Politis & Romano, 1994) instead resamples in *blocks* of random,
geometrically-distributed length, wrapping circularly at the end of the
series, so that local dependence structure survives into each resample.

skepsis's vectorized formulation is mathematically the same generative
process, described differently: each position in the resampled index
independently starts a *new* block with probability \(p = 1/L\), where
\(L\) is the target mean block length (geometric block lengths fall out of
these per-position coin flips, which is exactly the stationary bootstrap's
definition); a new block's starting index is drawn uniformly from
\([0, n)\); positions that don't start a new block continue from wherever
the current block left off, wrapping around the series (mod \(n\)).

The mean block length \(L\) is chosen automatically (Politis & White, 2004,
with the Patton-Politis-White flat-top correction) from the return series'
own autocovariance function \(\hat\gamma(k)\): find the smallest lag
\(\hat m\) beyond which the autocorrelation stays inside a
\(\pm 2\sqrt{\log_{10}n / n}\) significance band for \(k_n\) consecutive
lags, apply a flat-top lag window \(\lambda(\cdot)\) out to \(2\hat m\), and
compute

$$
\hat g = 2\sum_{k=1}^{m}\lambda\!\left(\frac{k}{m}\right) k\,\hat\gamma(k),
\qquad
\hat D = 2\left(\hat\gamma(0) + 2\sum_{k=1}^{m}\lambda\!\left(\frac{k}{m}\right)\hat\gamma(k)\right)^{2}
$$

$$
\hat L = \left(\frac{2\hat g^{2}}{\hat D}\right)^{1/3} n^{1/3}
$$

clipped to stay between 1 and a data-length-dependent ceiling. A larger
\(\hat L\) means the series shows dependence over longer stretches, so
resamples need bigger blocks to preserve it.

From \(n_{\text{resamples}}\) index draws, skepsis computes a distribution
of annualized Sharpe and max drawdown (giving confidence intervals), and
separately a **no-skill p-value**: it demeans the returns (removing any
average outperformance) and resamples the *demeaned* series with the same
block structure, giving a null distribution of Sharpe ratios that share the
original series' autocorrelation shape but have zero true mean. The p-value
is:

$$
p = \frac{1 + \#\{i : SR^{\text{null}}_i \ge SR_{\text{obs}}\}}{n_{\text{resamples}} + 1}
$$

A small \(p\) means the observed Sharpe is higher than almost every
draw from a mean-zero process with the same dependence structure — evidence
against pure noise, on its own terms.

A resample that happens to be exactly constant (every drawn period identical)
has an undefined ordinary Sharpe ratio, so skepsis scores it
\(\operatorname{sign}(\text{mean})\cdot\infty\) instead of dropping it —
except when that constant resample's mean is exactly zero (a zero-return,
zero-risk draw), which scores exactly \(0.0\) so it correctly *ties* a zero
observed Sharpe (\(0.0 \ge 0.0\)) rather than vanishing as
\(\operatorname{sign}(0)\cdot\infty = \text{NaN}\), which would fail every
comparison and silently bias the p-value downward. These draws still count
in the \(p\)-value's exceedance total (on the statistically correct side —
a demeaned, constant-positive or exactly-tying draw counts as evidence for
the null), keeping the \((1 + \#\{\cdot\}) / (n_{\text{resamples}} + 1)\)
denominator exact, while being excluded from the reported confidence
intervals (neither an infinite nor a degenerate zero Sharpe can populate a
quantile the way a genuine resample can). If every resample in a bootstrap
run is degenerate, skepsis raises rather than returning confidence
intervals it cannot actually form.

## When it lies

**Circular wrap stitches the end of history to the beginning.** Blocks that
run past the end of the series continue from index 0. This means resamples
can splice a block ending in, say, a market crash directly into a block
starting during a calm bull run — a transition that never happened in the
real data. The longer the block length relative to the sample size, the
more often this artificial join shows up, and the more it can distort the
tails of the resampled distributions.

**Everything here assumes the return process is roughly stationary.** The
block bootstrap resamples chunks from across the *entire* supplied history
as if the statistical properties (mean, variance, autocorrelation) are
stable through time. A strategy whose edge decayed, or whose target market
changed regime partway through the sample, will have its early "good" years
blended with later "bad" years in every resample, producing confidence
intervals that describe an average-of-all-regimes world rather than the
regime you'd actually be trading in next.

**The p-value tests one specific null: no skill, same autocorrelation.**
It says nothing about selection bias. A strategy chosen as the best of many
trials can post a tiny no-skill p-value — genuinely, robustly better than a
mean-zero process — while still being the product of overfitting, because
"better than mean-zero" and "better than what you'd expect after accounting
for how many things you tried" are different questions. That second
question is DSR's and PBO's job, not this one's.

**Automatic block-length selection is itself a noisy estimate**, especially
on shorter series — it depends on where the empirical autocorrelation first
drops inside a significance band, which can be sensitive to sampling noise.
skepsis enforces a hard floor of 50 observations, but the block-length
heuristic wants considerably more than that to be reliable, and a large
selected block length is itself a signal worth reading directly: it means
the series is far from independent, which strains not just this method's
own confidence intervals but the IID-ish assumptions behind PSR and DSR too
(skepsis raises a separate warning when this happens).

## Worked example

In the "Deflating the Golden Cross" demo, the bootstrap runs with 2,000
resamples on both the raw and excess-return versions of the best trial.

**Act 1 — raw returns.** The no-skill p-value is **0.0005** with a Sharpe
95% CI of **[0.85, 2.07]** — a resounding rejection of the zero-mean null.
But the auto-selected mean block length comes out to 11 periods, above
skepsis's autocorrelation-warning threshold, and correctly so: an
11-day-average dependence structure in daily BTC returns during a sustained
bull run is exactly the kind of persistence that makes "distinguishable
from zero" a much weaker claim than "genuine timing skill." The strategy
easily clears the no-skill bar because it's long an asset that mostly went
up, not because the entry/exit timing adds value.

**Act 2 — excess over buy-and-hold.** Against the harder, fairer null — beat
holding, not beat doing nothing — the no-skill p-value rises to **0.386**
with a CI of **[-0.50, 0.65]**, spanning zero. The bootstrap cannot reject
"this excess return is indistinguishable from noise," which is the correct
read: only 1 of 34 trials shows positive excess Sharpe in the first place.

## References

- Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
  *Journal of the American Statistical Association*, 89(428).
- Politis, D. N., & White, H. (2004). "Automatic Block-Length Selection for
  the Dependent Bootstrap." *Econometric Reviews*, 23(1).
- Sharpe, W. F. (1994). "The Sharpe Ratio." *Journal of Portfolio
  Management*, 21(1).
