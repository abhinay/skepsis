import json
import warnings as warnings_mod

import numpy as np
import pandas as pd
import pytest

import skepsis
from skepsis.exceptions import MisalignedTrialsError, SkepsisWarning


def _sweep(seed: int = 0):
    """400 obs, 9 trials (3x3 param grid), col 4 has a real edge; returns = col 4."""
    rng = np.random.default_rng(seed)
    trials = rng.normal(0.0, 0.01, size=(400, 9))
    trials[:, 4] += 0.003
    params = pd.DataFrame(
        {"window": np.repeat([10, 20, 30], 3), "threshold": np.tile([0.1, 0.2, 0.3], 3)}
    )
    return trials[:, 4].copy(), trials, params


def test_returns_only_path() -> None:
    r = np.random.default_rng(0).normal(0.002, 0.01, 500)
    with pytest.warns(SkepsisWarning, match="trial count 1"):
        res = skepsis.evaluate(r, freq="daily")
    assert 0.0 <= res.psr.value <= 1.0
    assert res.deflated_sharpe.single_trial and res.deflated_sharpe.n_trials == 1
    assert res.deflated_sharpe.value == pytest.approx(res.psr.value)  # benchmark 0 either way
    assert res.bootstrap is not None and res.pbo is None and res.sensitivity is None
    assert "pbo" in res.skipped and "sensitivity" in res.skipped
    assert any("trial count 1" in w for w in res.warnings)
    assert res.verdict.level in ("STRONG", "MODERATE", "WEAK", "LIKELY_OVERFIT")


def test_full_path_detects_skilled_strategy() -> None:
    r, trials, params = _sweep()
    with pytest.warns(SkepsisWarning):
        res = skepsis.evaluate(r, trials=trials, params=params, freq="daily",
                               n_resamples=500, seed=0)
    assert res.pbo is not None and res.sensitivity is not None
    assert not res.deflated_sharpe.single_trial and res.deflated_sharpe.n_trials == 9
    assert res.meta["chosen_label"] == "trial_4"  # auto-detected via allclose
    assert res.pbo.value < 0.5  # real edge: IS winner holds up OOS
    assert not res.skipped


def test_chosen_by_label_with_pandas_trials() -> None:
    r, trials, params = _sweep()
    df = pd.DataFrame(trials, columns=[f"s{i}" for i in range(9)])
    with pytest.warns(SkepsisWarning):
        res = skepsis.evaluate(r, trials=df, params=params, chosen="s4", n_resamples=200)
    assert res.meta["chosen_label"] == "s4"
    with pytest.warns(SkepsisWarning):
        res2 = skepsis.evaluate(r, trials=df, params=params, chosen=4, n_resamples=200)
    assert res2.meta["chosen_label"] == "s4"


def test_small_sample_skips_pbo_and_bootstrap() -> None:
    rng = np.random.default_rng(1)
    trials = rng.normal(0, 0.01, size=(30, 3))
    r = trials[:, 0].copy()
    res = skepsis.evaluate(r, trials=trials, n_resamples=100)
    assert res.pbo is None and "pbo" in res.skipped and "32" in res.skipped["pbo"]
    assert res.bootstrap is None and "bootstrap" in res.skipped
    assert res.verdict.level in ("STRONG", "MODERATE", "WEAK", "LIKELY_OVERFIT")


def test_to_dict_is_json_serializable_and_deterministic() -> None:
    r, trials, params = _sweep()
    with pytest.warns(SkepsisWarning):
        a = skepsis.evaluate(r, trials=trials, params=params, n_resamples=200, seed=7)
    with pytest.warns(SkepsisWarning):
        b = skepsis.evaluate(r, trials=trials, params=params, n_resamples=200, seed=7)
    da, db = a.to_dict(), b.to_dict()
    assert json.dumps(da, sort_keys=True, allow_nan=False) == json.dumps(
        db, sort_keys=True, allow_nan=False
    )
    assert da["verdict"]["level"] == a.verdict.level
    assert "value" in da["psr"] and "p_value_no_skill" in da["bootstrap"]


def test_summary_mentions_verdict() -> None:
    r = np.random.default_rng(0).normal(0.002, 0.01, 500)
    with warnings_mod.catch_warnings():
        warnings_mod.simplefilter("ignore")
        res = skepsis.evaluate(r)
    assert res.verdict.level in res.summary()


def test_misalignment_raises() -> None:
    r = np.random.default_rng(0).normal(0, 0.01, 100)
    trials = np.random.default_rng(1).normal(0, 0.01, (99, 3))
    with pytest.raises(MisalignedTrialsError):
        skepsis.evaluate(r, trials=trials)
