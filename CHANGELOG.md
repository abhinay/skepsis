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
