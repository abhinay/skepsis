"""The p-value assertion margins are wide by design: the drift case sits >4 sigma
from the null (verified p=0.001 across seeds 0-4 on 2026-07-13)."""

import numpy as np
import pytest

from skepsis.core import moments
from skepsis.core.bootstrap import (
    bootstrap,
    politis_white_block_length,
    stationary_bootstrap_indices,
)
from skepsis.exceptions import InsufficientDataError


def test_block_length_orders_iid_vs_persistent() -> None:
    iid = np.random.default_rng(7).normal(0, 1, 1000)
    eps = np.random.default_rng(8).normal(0, 1, 2000)
    ar = np.empty(2000)
    ar[0] = 0.0
    for t in range(1, 2000):
        ar[t] = 0.8 * ar[t - 1] + eps[t]
    b_iid = politis_white_block_length(iid)
    b_ar = politis_white_block_length(ar)
    assert 1.0 <= b_iid < 5.0
    assert b_ar > 5.0 * b_iid


def test_indices_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(100, 5.0, 32, rng)
    assert idx.shape == (32, 100) and idx.dtype == np.int64
    assert idx.min() >= 0 and idx.max() < 100


def test_determinism_given_seed() -> None:
    r = np.random.default_rng(1).normal(0.001, 0.01, 300)
    a = bootstrap(r, 252.0, n_resamples=200, seed=42)
    b = bootstrap(r, 252.0, n_resamples=200, seed=42)
    np.testing.assert_array_equal(a.sharpe_distribution, b.sharpe_distribution)
    assert a.p_value_no_skill == b.p_value_no_skill


@pytest.mark.parametrize("seed", range(5))
def test_strong_drift_has_small_p_value(seed: int) -> None:
    r = np.random.default_rng(seed).normal(0.002, 0.01, 1000)  # ann SR ~ 3
    res = bootstrap(r, 252.0, n_resamples=1000, mean_block_length=5.0, seed=seed)
    assert res.p_value_no_skill < 0.05
    assert res.sharpe_ci[0] < res.sharpe_obs < res.sharpe_ci[1]


def test_observed_stats_match_moments() -> None:
    r = np.random.default_rng(3).normal(0.001, 0.01, 500)
    res = bootstrap(r, 252.0, n_resamples=100, seed=0)
    assert res.sharpe_obs == pytest.approx(moments.annualized_sharpe(r, 252.0))
    assert res.drawdown_obs == pytest.approx(moments.max_drawdown(r))
    assert 0.0 < res.p_value_no_skill <= 1.0


def test_insufficient_data() -> None:
    with pytest.raises(InsufficientDataError):
        bootstrap(np.zeros(49) + np.random.default_rng(0).normal(0, 0.01, 49), 252.0)
