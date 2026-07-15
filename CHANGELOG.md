# Changelog

## 0.1.0 — 2026-07-14

First public release.

- Four overfitting diagnostics: Probabilistic & Deflated Sharpe Ratio
  (Bailey & López de Prado 2012, 2014 — reproduces the DSR paper's published
  numerical example to all four printed decimals), Probability of Backtest
  Overfitting via CSCV (Bailey, Borwein, López de Prado & Zhu 2015),
  stationary block bootstrap with Politis–White block length (1994, 2004),
  and k-NN parameter-sensitivity maps.
- `skepsis.evaluate()` with progressive disclosure, rule-based overridable
  verdict, strict-JSON `to_dict()`, and a fully self-contained single-file
  HTML report.
- Vectorized bootstrap index generation (~830× on the iid path). Note:
  seeded resample streams differ from pre-release development builds; no
  released version depended on them.
- Demo: "Deflating the Golden Cross" notebook on 11 years of BTC-USD.
- Drawdowns are measured against initial capital (a first-period loss now
  registers); bootstrap index generation is chunked to bound peak memory
  (~10x lower on default workloads), input knobs are validated, and seeded
  resample streams differ from earlier development builds.
- Constant (zero-variance) inputs are detected exactly (immune to
  floating-point reduction noise) and rejected or scored deterministically;
  degenerate bootstrap resamples count toward the no-skill p-value as
  signed-infinite Sharpe and are excluded from confidence intervals, with
  their count reported. Exactly-zero-mean constant draws instead score a
  Sharpe of 0.0, correctly tying a zero observed Sharpe rather than
  vanishing as `sign(0) * inf == nan`; and a bootstrap whose resamples are
  all degenerate now raises instead of crashing on an empty quantile.
