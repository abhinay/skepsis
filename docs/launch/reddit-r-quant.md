Title: skepsis: open-source library that tells you whether your backtest is overfit (DSR, PBO/CSCV, block bootstrap, sensitivity maps)

Most backtests published anywhere — blogs, vendor decks, even papers — omit
the one number that determines whether the result means anything: how many
variants were tried. I built skepsis to make accounting for that trivial.

`skepsis.evaluate(returns, trials=..., params=...)` runs: Probabilistic &
Deflated Sharpe Ratio (Bailey & López de Prado), Probability of Backtest
Overfitting via CSCV (Bailey, Borwein, López de Prado & Zhu), a stationary
block bootstrap with Politis-White block length, and a k-NN parameter
sensitivity map. Output: a rule-based verdict (thresholds documented and
overridable) and a self-contained HTML report.

Receipts, because this sub is rightly hostile to claims:
- Golden tests reproduce the DSR paper's numerical example to all four
  printed decimals (SR0 0.1132, DSR 0.9004, and both secondary claims).
- The demo sweeps 34 MA-crossover combos on 11y of daily BTC with shift(1)
  timing and no costs: best combo Sharpe 1.46 / 650x. Against buy-and-hold,
  33/34 variants lose and the winner's DSR is 0.35. PBO says the parameter
  choice flips OOS rank on ~45% of CSCV splits.
- Everything fails loud: NaNs rejected, skipped diagnostics say why,
  strained assumptions warn (e.g., heavy autocorrelation vs PSR).

Apache-2.0, Python ≥3.11, numpy/scipy core. Deliberately NOT a backtester.
Would genuinely value this sub's abuse: what's wrong, what's missing, what
would make you trust it or bin it?

https://github.com/abhinay/skepsis · https://abhinay.github.io/skepsis
