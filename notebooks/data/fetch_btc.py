"""Fetch daily BTC-USD closes from the Coinbase Exchange public candles API.

Provenance: https://api.exchange.coinbase.com/products/BTC-USD/candles
(granularity 86400, max 300 candles per request, paginated). The date range
is PINNED so the committed dataset — and every number in the demo notebook
and launch posts — is reproducible. Extend END only deliberately, together
with re-running the notebook and updating its narrative numbers.

Run: python notebooks/data/fetch_btc.py  (writes btc_usd_daily.csv next to itself)
"""

import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
START = dt.date(2015, 7, 20)  # first day Coinbase BTC-USD daily candles exist
END = dt.date(2026, 7, 13)    # PINNED: dataset fingerprint 4012 rows (see README)


def fetch_daily(start: dt.date, end: dt.date) -> pd.DataFrame:
    rows = []
    cur = start
    while cur < end:
        chunk_end = min(cur + dt.timedelta(days=299), end)
        url = (
            f"{BASE}?granularity=86400&start={cur.isoformat()}T00:00:00Z"
            f"&end={chunk_end.isoformat()}T00:00:00Z"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "skepsis-demo/0.1"})
        with urllib.request.urlopen(req) as resp:
            rows.extend(json.loads(resp.read()))  # [time, low, high, open, close, volume]
        cur = chunk_end + dt.timedelta(days=1)
        time.sleep(0.15)  # public rate-limit courtesy
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.date
    return (
        df.drop_duplicates("date").sort_values("date").reset_index(drop=True)[["date", "close"]]
    )


if __name__ == "__main__":
    out = Path(__file__).parent / "btc_usd_daily.csv"
    df = fetch_daily(START, END)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {df['date'].iloc[0]} .. {df['date'].iloc[-1]}")
