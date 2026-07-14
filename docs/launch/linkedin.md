I shipped an open-source project: skepsis — statistical diagnostics for
backtest overfitting.

The uncomfortable truth about backtests is that the headline Sharpe ratio
is meaningless without one more number: how many variants were tried before
picking the winner. skepsis makes that accounting one function call — the
Deflated Sharpe Ratio, Probability of Backtest Overfitting, block-bootstrap
confidence intervals, and a parameter-sensitivity map, with a shareable
one-file HTML report.

My favorite result from building it: the demo takes the most-published
strategy on the internet (the moving-average "golden cross"), runs it
honestly on 11 years of Bitcoin data, and gets a seductive Sharpe 1.46 /
651x backtest. Then it asks the fair question — does the timing beat just
holding? 33 of 34 parameter combinations don't, and the survivor's Deflated
Sharpe is 0.35. The edge was beta in a costume.

Implementations are verified against reference golden values (the DSR paper's
example reproduces to all four printed decimals). Apache-2.0.

Repo: https://github.com/abhinay/skepsis
Docs: https://abhinay.github.io/skepsis

If you work on research infrastructure and this is the kind of tooling you
care about — I'd love to compare notes.
