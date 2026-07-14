Title: Show HN: Skepsis – statistical tests for whether your backtest is overfit

I built an open-source Python library that answers one question: is this
backtest result real, or luck?

You give it your strategy's returns — and, crucially, the returns of every
variant you tried — and it runs the diagnostics from the backtest-overfitting
literature: Deflated Sharpe Ratio, Probability of Backtest Overfitting
(CSCV), a stationary block bootstrap, and a parameter-sensitivity map. It
produces a single-file HTML report with a verdict.

The demo notebook builds the most-published strategy on the internet — the
moving-average crossover, on 11 years of daily BTC — sweeps all 34
parameter combinations honestly (no look-ahead, no costs), and finds a
Sharpe 1.46, 651x backtest. Then it deflates it: measured against simply
holding, 33 of 34 variants lose, and the best survivor's Deflated Sharpe
Ratio is 0.35 — worse than a coin flip. The Sharpe was beta wearing a
costume.

The implementations are verified against the published papers, including
reproducing the DSR paper's numerical example to all four printed decimals.
Apache-2.0. Not a backtester — it sits downstream of whatever you use.

Repo: https://github.com/abhinay/skepsis
Docs: https://abhinay.github.io/skepsis
Demo: https://github.com/abhinay/skepsis/blob/main/notebooks/deflating-the-golden-cross.ipynb
