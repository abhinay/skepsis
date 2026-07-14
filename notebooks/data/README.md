# BTC-USD daily closes

`btc_usd_daily.csv` — 4012 daily closes, 2015-07-20 → 2026-07-13 (UTC),
fetched 2026-07-14 from the Coinbase Exchange public market-data API
(`api.exchange.coinbase.com/products/BTC-USD/candles`, granularity 86400) by
`fetch_btc.py`. Prices are factual market data; the fetch script, source,
and pinned date range are committed so the file is reproducible bit-for-bit.
The data is factual market price information retrieved from Coinbase's
public, unauthenticated market-data endpoint; the committed CSV is a
minimal derived extract (date and close only) redistributed for
reproducibility, with the fetch script provided so anyone can regenerate it
from the source.

The demo notebook reads this file and needs no network. The date range is
pinned because the notebook's narrative and the launch posts quote numbers
computed from exactly this dataset (fingerprint enforced by
`tests/unit/test_demo_data.py`). To extend the range: edit `END` in
`fetch_btc.py`, re-run it, re-execute the notebook, and update the quoted
numbers and the fingerprint test together.
