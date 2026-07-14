# skepsis

> Is your backtest result real, or overfit?

skepsis takes the returns of a backtested strategy — and, ideally, the returns
of **every variant you tried along the way** — and produces statistical
evidence for or against the result being luck, plus a self-contained HTML
report you can hand to a PM.

**Status: pre-release.** Not yet on PyPI; install from git:

```bash
pip install "skepsis @ git+https://github.com/abhinay/skepsis"
```

## Quickstart

```python
import skepsis

result = skepsis.evaluate(
    returns,            # per-period returns of the chosen strategy
    trials=trials_df,   # T x N returns of all variants tried (optional, unlocks DSR + PBO)
    params=params_df,   # one row of parameter values per trial (optional, unlocks sensitivity)
    freq="daily",
)
print(result.summary())
result.save_html("skepsis_report.html")
```

More inputs unlock more diagnostics; the report states plainly which
diagnostics could not run and why.

## The diagnostics

| Diagnostic | Question it answers | Source |
|---|---|---|
| Probabilistic / Deflated Sharpe Ratio | Is the Sharpe distinguishable from zero once sample length, fat tails, and **how many things you tried** are priced in? | Bailey & López de Prado (2012, 2014) |
| Probability of Backtest Overfitting (CSCV) | How often does your in-sample winner land in the bottom half out-of-sample? | Bailey, Borwein, López de Prado & Zhu (2015) |
| Stationary block bootstrap | What does the Sharpe/drawdown distribution look like under resampling that preserves autocorrelation — and does a no-skill null explain it? | Politis & Romano (1994); Politis & White (2004) |
| Parameter sensitivity | Is the chosen configuration a plateau (robust) or an isolated spike (fitted to noise)? | — |

The implementations are verified against reference values in
[`tests/golden/`](tests/golden/) — including reproducing the published
numerical example of the Deflated Sharpe Ratio paper to all four printed
decimals ([`test_dsr_paper_example.py`](tests/golden/test_dsr_paper_example.py)).

## What skepsis is not

Not a backtester, not a data source, not a portfolio optimizer. It sits
downstream of vectorbt, backtesting.py, zipline, or your homegrown engine,
and it never silently repairs bad input: NaNs are rejected, strained
assumptions are warned about, skipped diagnostics say why.

## License

Apache-2.0.
