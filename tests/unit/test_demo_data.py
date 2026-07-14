"""Locks the committed demo dataset: the notebook's narrative numbers and the
launch posts quote results computed from exactly this data."""

from pathlib import Path

import pandas as pd
import pytest

CSV = Path(__file__).parents[2] / "notebooks" / "data" / "btc_usd_daily.csv"


def test_dataset_fingerprint() -> None:
    df = pd.read_csv(CSV)
    assert list(df.columns) == ["date", "close"]
    assert len(df) == 4012
    assert df["date"].iloc[0] == "2015-07-20" and df["close"].iloc[0] == 280.0
    assert df["date"].iloc[-1] == "2026-07-13" and df["close"].iloc[-1] == 62264.94
    assert df["close"].sum() == pytest.approx(124818690.85, abs=0.01)
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
