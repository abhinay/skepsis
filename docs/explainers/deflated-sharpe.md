# Deflated Sharpe Ratio

## The question it answers

You ran a backtest and got a Sharpe ratio you like. Two things can make that
number lie to you. First, it was estimated from a finite, possibly skewed,
possibly fat-tailed sample, so the raw Sharpe overstates how confident you
should be that the *true* Sharpe is above zero. Second, and far more
dangerous: if you tried more than one strategy variant before picking this
one, the best of a pile of mediocre-to-bad variants can look great by chance
alone, even if every variant has zero real skill. The Probabilistic Sharpe
Ratio (PSR) fixes the first problem. The Deflated Sharpe Ratio (DSR) fixes
the second by asking the harder question: given how many things you tried,
what's the probability that *this* Sharpe reflects real skill rather than
the best draw from a large pile of noise?

## The math, in plain English

PSR asks: what's the probability that the true Sharpe ratio exceeds some
benchmark \(SR^*\), given the observed Sharpe, the sample length, and the
shape of the return distribution?

$$
\mathrm{PSR}(SR^*) = \Phi\!\left(\frac{(\widehat{SR} - SR^*)\,\sqrt{T-1}}
{\sqrt{1 - \hat\gamma_3\,\widehat{SR} + \dfrac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^2}}\right)
$$

\(\Phi\) is the standard normal CDF, \(T\) is the number of return
observations, \(\hat\gamma_3\) is sample skewness, and \(\hat\gamma_4\) is
sample kurtosis under the convention that a normal distribution scores
\(\hat\gamma_4 = 3\) (not the "excess kurtosis" convention where normal
scores 0). Negative skew and fat tails widen the denominator and push PSR
down: the same observed Sharpe earns less confidence when the return
distribution is nastier than normal.

DSR is PSR evaluated against a benchmark that isn't zero: it's the Sharpe
ratio you'd *expect the best of N zero-skill trials to show by chance*.
That expected maximum is:

$$
E[\max_{n=1\ldots N} SR_n] \;\approx\; \hat\sigma_{SR}
\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\frac{1}{N}\right)
+ \gamma\,\Phi^{-1}\!\left(1-\frac{1}{N e}\right)\right]
$$

\(\gamma \approx 0.5772\) is the Euler-Mascheroni constant, \(N\) is the
number of trials, and \(\hat\sigma_{SR}\) is the standard deviation of the
periodic Sharpe ratios across the trials. This is a first-order
approximation from extreme value theory: even under a pure luck null (every
trial has zero true skill), the best of many draws is expected to look good,
and the spread of Sharpes across your trials tells you how good. DSR then
plugs this expected-max Sharpe back into the PSR formula as \(SR^*\):

$$
\mathrm{DSR} = \mathrm{PSR}\big(E[\max SR_n]\big)
$$

More trials, or more spread in trial quality, raise the bar the chosen
strategy has to clear. With a single trial (\(N=1\)) the formula has no
multiple-testing benchmark to apply, so skepsis returns \(E[\max SR] = 0\)
and DSR collapses to plain PSR — flagged elsewhere as an optimistic
special case (see below).

skepsis's implementation reproduces the DSR paper's own published numerical
example to all four printed decimals as a golden test — the formula above is
exactly the code, not a paraphrase.

## When it lies

**skepsis deflates with the raw trial count, not the paper's
effective-trials correction.** The DSR paper's full method estimates an
*effective* number of independent trials by clustering correlated variants
together (e.g. via correlation-based clustering of the trial return series)
so that ten near-duplicate parameter settings count closer to one
independent trial than ten. skepsis does not implement that correction: `N`
in the formula above is literally the number of columns you pass in
`trials=`. This is a deliberate design choice (spec §3.2.8), and it is
conservative — clustering correlated trials down to a smaller effective `N`
would *raise* \(E[\max SR]\) less, which raises DSR. Skipping the correction
means skepsis's DSR is always at least as harsh as, and often harsher than,
the paper's effective-N version. If your 50 "trials" are actually five
strategies each swept across ten near-identical parameter values, expect
skepsis to deflate more than the textbook method would. That's the
trade-off: no hidden clustering heuristic to get wrong, at the cost of
sometimes being pessimistic about genuinely correlated trial families.

**Heavy autocorrelation strains the assumptions.** The \(\sqrt{T-1}\) scaling
and the skew/kurtosis correction come from an asymptotic, close-to-IID
argument. Returns with strong serial correlation (trending strategies on
trending assets are the classic case) violate that, and the effective
sample size is smaller than \(T\) suggests. skepsis surfaces this
indirectly: when the bootstrap's auto-selected block length (Politis-White)
comes out large, it's a sign the series is far from IID, and skepsis emits
a warning that PSR/DSR should be read with extra skepticism.

**A single trial gives you no real deflation.** If you don't pass `trials=`,
skepsis runs DSR with \(N=1\), which is mathematically identical to plain
PSR against a zero benchmark. It says so explicitly, with a warning: this
number is almost certainly optimistic, because you tried more than one
thing to get here even if you didn't record the others.

## Worked example

The "Deflating the Golden Cross" demo evaluates a moving-average crossover
family on 11 years of BTC-USD, sweeping 34 fast/slow window combinations.

**Act 1 — raw returns.** The best combination, `f5_s100`, posts an
annualized Sharpe of **1.464** against a buy-and-hold Sharpe of **1.064**
over the same period. Fed the full 34-trial sweep, DSR does not kill this
result — the verdict comes back **MODERATE**, with the only flagged rule
being PBO (0.448, covered in the [PBO explainer](pbo.md)). The reason DSR
survives is that the returns are real: every one of the 34 variants clears
a Sharpe of 1.05, because they're all long BTC through a decade-long bull
run. High DSR here does not mean "real timing skill" — it means the
deflation correctly recognizes that this family of strategies would have
looked good almost regardless of which fast/slow pair you'd picked, because
the edge is beta, not selection.

**Act 2 — excess over buy-and-hold.** A timing strategy earns its keep only
if it beats doing nothing, so evaluate each variant's return *minus*
buy-and-hold. Only **1 of the 34** trials shows positive excess Sharpe, and
the best of them manages just **0.084**. Deflated against 34 trials, that
becomes a **DSR of 0.345** — worse than a coin flip that the true excess
Sharpe is even positive, let alone attractive — and the verdict flips to
**LIKELY_OVERFIT**. This is the honest version of the question: not "does
this beat zero," but "does this beat the trivially available alternative,
once you account for how many ways you tried to beat it."

## References

- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient
  Frontier." *Journal of Risk*, 15(2).
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  *Journal of Portfolio Management*, 40(5).
- Sharpe, W. F. (1994). "The Sharpe Ratio." *Journal of Portfolio
  Management*, 21(1).
