# The Verdict

## The question it answers

You've run PSR/DSR, PBO, the bootstrap, and the sensitivity map, and you
have four different, only loosely related, statistical opinions. Do you
trust this backtest or not? The verdict is skepsis's attempt at a single
answer: a rule-based aggregate across whichever diagnostics your inputs
allowed skepsis to run, labeled `STRONG`, `MODERATE`, `WEAK`, or
`LIKELY_OVERFIT`.

## The math, in plain English

The verdict is not a new statistical test. It is a fixed set of threshold
comparisons against the outputs of the diagnostics above, plus a simple
counting rule. Every threshold lives in one place
(`skepsis.Thresholds`), is documented, and is overridable.

| Diagnostic | fails if | warns if |
|---|---|---|
| Deflated Sharpe Ratio | `dsr < 0.5` | `dsr < 0.95` |
| Probability of Backtest Overfitting | `pbo > 0.5` | `pbo > 0.2` |
| Bootstrap no-skill p-value | `p > 0.5` | `p > 0.05` |
| Sensitivity stability score | `score > 2.0` | `score > 1.5` |

(A `fail` condition is checked first and takes priority; a diagnostic
that fails does not also count as a warn.)

Any diagnostic that wasn't run — because you didn't pass `trials=` for PBO,
or the series was too short for the bootstrap, or you didn't pass `params=`
for sensitivity — is simply skipped: it contributes neither a fail nor a
warn, and `Result.skipped` records why. Aggregation:

- **Any fail → `LIKELY_OVERFIT`.** One diagnostic crossing its fail line is
  enough to override everything else, regardless of how good the other
  three look.
- **Two or more warns (and no fails) → `WEAK`.**
- **Exactly one warn (and no fails) → `MODERATE`.**
- **Zero warns and zero fails → `STRONG`.**

**The single-trial rule:** if you call `evaluate()` without `trials=`, DSR
still runs, but against a trial count of 1 — which is mathematically
equivalent to no multiple-testing correction at all. Regardless of the DSR
value that produces, skepsis adds an explicit warn: *"DSR computed with
trial count 1 (no trials provided) — almost certainly optimistic."* A
single-trial evaluation can never score better than `MODERATE`, because
that warn alone consumes the "one warn" budget.

Every threshold is overridable per call, for teams with different risk
tolerances or diagnostics they weight differently:

```python
result = skepsis.evaluate(
    returns, trials=trials_df, params=params_df, freq="daily",
    thresholds=skepsis.Thresholds(pbo_warn=0.3),
)
```

## When it lies

**The verdict is a heuristic label, not a statistical guarantee.** There is
no joint hypothesis test being run across the four diagnostics, no combined
error rate, no correction for the fact that DSR, PBO, and the bootstrap's
no-skill p-value are all, to varying degrees, measuring correlated aspects
of "does this generalize." Treating `STRONG` as a certificate is exactly
the kind of overclaiming this project exists to push back against — read
the individual diagnostics and their reasons, always.

**One fail overrides everything, by design, which can feel harsh.** A
result with an excellent DSR, an excellent bootstrap p-value, and an
excellent PBO, but a sensitivity score of 2.01 — marginally over the fail
line — gets the same `LIKELY_OVERFIT` label as a result that fails on every
axis. That's a
deliberate conservative choice (a backtest with even one strong overfitting
signal deserves the loudest label), not evidence that all `LIKELY_OVERFIT`
results are equally bad — read the `reasons` tuple to see which rule
actually fired and by how much.

**The warn/fail boundaries are discrete, and real numbers sit close to
them.** A DSR of 0.501 and a DSR of 0.499 produce different verdict
components (warn vs. fail) despite being nearly identical numbers. Don't
over-read a verdict that changes because a diagnostic crossed a threshold
by a hair; look at the underlying value.

**Skipped diagnostics silently lower the bar.** An evaluation that only
gets `returns=` (no `trials=`, no `params=`) can only ever run PSR/single-trial
DSR and the bootstrap — PBO and sensitivity are skipped, not failed, so they
cannot pull the verdict down. A `STRONG` verdict on a minimal-input call
means "the diagnostics that ran found nothing wrong," not "every possible
check passed."

## Worked example

Both acts of the "Deflating the Golden Cross" demo run the full diagnostic
set (`trials=` and `params=` both provided), so all four rules are live.

**Act 1 — raw returns.** DSR does not fail (the family survives deflation
because the returns are real, if beta-driven), the bootstrap easily rejects
the no-skill null (p = 0.0005), and the stability score (≈1.13) is well
inside the plateau range. The only rule that fires is PBO: **0.448 > 0.2
(warn)**. One warn, zero fails → verdict **MODERATE**.

**Act 2 — excess over buy-and-hold.** Here the diagnostics agree, loudly:

```
- DSR 0.345 < 0.5 (fail)
- stability score ∞ (isolated spike) > 2.0 (fail)
- PBO 0.405 > 0.2 (warn)
- bootstrap no-skill p 0.386 > 0.05 (warn)
```

Two fails are already enough on their own; the two warns confirm the same
conclusion from independent angles. Verdict: **LIKELY_OVERFIT**. This is
the notebook's punchline: the raw-return family looks fine because BTC went
up a lot over 11 years, but the same family, measured against the fair
null of "did the timing add anything over just holding," collapses on
every axis skepsis checks.

## References

- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient
  Frontier." *Journal of Risk*, 15(2).
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  *Journal of Portfolio Management*, 40(5).
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2015).
  "The Probability of Backtest Overfitting." *Journal of Computational
  Finance*.
- Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
  *Journal of the American Statistical Association*, 89(428).
- Politis, D. N., & White, H. (2004). "Automatic Block-Length Selection for
  the Dependent Bootstrap." *Econometric Reviews*, 23(1).
- Sharpe, W. F. (1994). "The Sharpe Ratio." *Journal of Portfolio
  Management*, 21(1).
