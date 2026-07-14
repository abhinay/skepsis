import numpy as np
import pandas as pd
import pytest

import skepsis
from skepsis.exceptions import SkepsisWarning


@pytest.fixture(scope="module")
def full_result() -> skepsis.Result:
    rng = np.random.default_rng(0)
    trials = rng.normal(0.0, 0.01, size=(400, 9))
    trials[:, 4] += 0.003
    params = pd.DataFrame(
        {"window": np.repeat([10, 20, 30], 3), "threshold": np.tile([0.1, 0.2, 0.3], 3)}
    )
    with pytest.warns(SkepsisWarning, match="non-positive"):
        return skepsis.evaluate(
            trials[:, 4].copy(), trials=trials, params=params, n_resamples=300, seed=0
        )


@pytest.fixture(scope="module")
def minimal_result() -> skepsis.Result:
    r = np.random.default_rng(1).normal(0.002, 0.01, 500)
    with pytest.warns(UserWarning):
        return skepsis.evaluate(r, n_resamples=300)


def test_full_report_renders_and_is_self_contained(full_result, tmp_path) -> None:
    out = full_result.save_html(tmp_path / "report.html")
    html = out.read_text(encoding="utf-8")
    assert "<script src=" not in html  # no external scripts
    assert '<link rel=' not in html  # no external stylesheets/fonts
    assert len(html) > 500_000  # plotly.js actually inlined
    assert full_result.verdict.level in html
    assert "Probability of Backtest Overfitting" in html
    assert "Deflated Sharpe" in html
    assert "Parameter sensitivity" in html
    assert "∞ (isolated spike)" in html  # fixture's stability is inf
    assert "9 trials" in html
    assert "trial(s)" not in html  # every trial count goes through count_trials


def test_minimal_report_shows_skipped_reasons(minimal_result, tmp_path) -> None:
    html = (minimal_result.save_html(tmp_path / "min.html")).read_text(encoding="utf-8")
    assert "trials not provided" in html
    assert "params not provided" in html
    assert "trial count 1" in html  # the single-trial warning is surfaced


def test_report_cites_sources(full_result, tmp_path) -> None:
    html = (full_result.save_html(tmp_path / "r.html")).read_text(encoding="utf-8")
    assert "Bailey" in html and "Politis" in html
