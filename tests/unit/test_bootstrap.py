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
from skepsis.exceptions import InsufficientDataError, InvalidInputError, SkepsisWarning


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


def test_index_generation_is_vectorized_fast() -> None:
    # Spec benchmark: (n_obs=4000, L=1.0, n_resamples=5000). Verified 0.035s
    # vectorized vs 29s looped locally; the 10s CI bound is deliberately loose
    # so loaded runners never flake. A regression to the Python loop fails this.
    import time

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    idx = stationary_bootstrap_indices(4000, 1.0, 5000, rng)
    elapsed = time.perf_counter() - t0
    assert idx.shape == (5000, 4000)
    assert elapsed < 10.0


def test_bootstrap_drawdown_counts_initial_capital() -> None:
    # strictly losing but non-constant series: every resample draws down from 1.0
    r = np.tile([-0.01, -0.02], 30)
    res = bootstrap(r, 252.0, n_resamples=50, mean_block_length=2.0, seed=0)
    assert res.drawdown_obs == pytest.approx(1 - (0.99 * 0.98) ** 30)
    assert res.drawdown_ci[0] > 0.0


def test_index_generation_memory_bounded() -> None:
    # Full-matrix vectorization peaked ~917MB on this workload; chunked
    # generation must stay under a generous 500MB (output itself is 160MB).
    import tracemalloc

    rng = np.random.default_rng(0)
    tracemalloc.start()
    idx = stationary_bootstrap_indices(4000, 5.0, 5000, rng)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert idx.shape == (5000, 4000)
    assert peak < 500 * 1024 * 1024


def test_chunk_boundary_determinism() -> None:
    # 300 resamples crosses the 256-row chunk boundary
    a = stationary_bootstrap_indices(50, 3.0, 300, np.random.default_rng(9))
    b = stationary_bootstrap_indices(50, 3.0, 300, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (300, 50)


def test_index_knob_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(InvalidInputError):
        stationary_bootstrap_indices(100, 5.0, 0, rng)
    with pytest.raises(InvalidInputError):
        stationary_bootstrap_indices(0, 5.0, 10, rng)
    with pytest.raises(InvalidInputError):
        stationary_bootstrap_indices(100, float("nan"), 10, rng)
    with pytest.raises(InvalidInputError):
        stationary_bootstrap_indices(100, float("inf"), 10, rng)


def test_constant_input_raises() -> None:
    with pytest.raises(InvalidInputError, match="constant"):
        bootstrap(np.full(60, -0.01), 252.0, n_resamples=20, mean_block_length=2.0)


def test_degenerate_null_draws_count_as_exceedances() -> None:
    # 49 zeros + one loss: many null resamples are constant-positive after
    # demeaning -> +inf Sharpe -> exceedances. The buggy filter-and-keep-
    # denominator behavior reported ~0.55 here; correct handling is ~0.9.
    r = np.concatenate([np.zeros(49), [-0.1]])
    with pytest.warns(SkepsisWarning, match="constant"):
        res = bootstrap(r, 252.0, n_resamples=2000, mean_block_length=1.0, seed=0)
    assert res.p_value_no_skill > 0.8
    assert res.n_degenerate_resamples > 0
    assert np.isfinite(res.sharpe_distribution).all()
    # Pin the actual value (computed post-fix, seed=0): 0.9240. Wide margin
    # since this is a discrete count-based statistic sensitive to RNG stream.
    assert res.p_value_no_skill == pytest.approx(0.9240379810094953, abs=0.02)


def test_integral_knob_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(InvalidInputError):
        stationary_bootstrap_indices(100, 5.0, 1.5, rng)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        stationary_bootstrap_indices(True, 5.0, 10, rng)  # type: ignore[arg-type]


def test_bootstrap_periods_validation() -> None:
    r = np.random.default_rng(0).normal(0.001, 0.01, 100)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(InvalidInputError):
            bootstrap(r, bad, n_resamples=20)
