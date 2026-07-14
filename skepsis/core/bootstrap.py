"""Stationary block bootstrap for Sharpe and drawdown confidence intervals.

References:
- Politis & Romano (1994), "The Stationary Bootstrap", JASA 89(428).
- Politis & White (2004), "Automatic Block-Length Selection for the Dependent
  Bootstrap", Econometric Reviews 23(1), with the Patton-Politis-White (2009)
  correction.

The no-skill p-value resamples the DEMEANED series with the same index matrix
and reports (1 + #{SR_null >= SR_obs}) / (n_resamples + 1).
"""

import math
from dataclasses import dataclass

import numpy as np

from skepsis.core.moments import annualized_sharpe, max_drawdown
from skepsis.exceptions import InsufficientDataError, InvalidInputError

MIN_OBS = 50


def politis_white_block_length(x: np.ndarray) -> float:
    """Automatic mean block length for the stationary bootstrap (Politis-White 2004)."""
    n = len(x)
    if n < MIN_OBS:
        raise InsufficientDataError(f"block-length selection needs >= {MIN_OBS} obs, got {n}")
    kn = max(5, int(math.sqrt(math.log10(n))))
    m_max = int(math.ceil(math.sqrt(n))) + kn
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    xc = x - x.mean()
    acov = np.array([float(np.dot(xc[: n - k], xc[k:])) / n for k in range(m_max + 1)])
    if acov[0] == 0.0:
        raise InvalidInputError("returns have zero variance; block length is undefined")
    acorr = acov / acov[0]
    band = 2.0 * math.sqrt(math.log10(n) / n)
    m_hat = None
    for m in range(1, m_max + 1):
        window = acorr[m + 1 : m + kn + 1]
        if len(window) < kn:
            break
        if bool(np.all(np.abs(window) < band)):
            m_hat = m
            break
    if m_hat is None:
        big = np.where(np.abs(acorr[1:]) > band)[0]
        m_hat = int(big.max()) + 1 if len(big) else 1
    m_top = min(2 * m_hat, m_max)
    lags = np.arange(1, m_top + 1)
    lam = np.where(lags / m_top <= 0.5, 1.0, 2.0 * (1.0 - lags / m_top))
    g_hat = 2.0 * float(np.sum(lam * lags * acov[1 : m_top + 1]))
    d_hat = 2.0 * float(acov[0] + 2.0 * np.sum(lam * acov[1 : m_top + 1])) ** 2
    if d_hat <= 0.0 or g_hat == 0.0:
        return 1.0
    b = ((2.0 * g_hat * g_hat) / d_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b, 1.0, b_max))


def stationary_bootstrap_indices(
    n_obs: int, mean_block_length: float, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_resamples, n_obs) index matrix: geometric block lengths, circular wrap.

    Vectorized formulation: each position independently starts a new block with
    probability p = 1/mean_block_length (this IS the stationary bootstrap of
    Politis & Romano — geometric block lengths emerge from the per-position
    Bernoulli trials); block starts are uniform on [0, n_obs); within a block,
    indices continue circularly from the block's start.
    """
    if mean_block_length < 1.0:
        raise InvalidInputError(f"mean_block_length must be >= 1, got {mean_block_length}")
    if mean_block_length == 1.0:
        # every position is its own block: plain iid resampling
        return rng.integers(0, n_obs, size=(n_resamples, n_obs), dtype=np.int64)
    p = 1.0 / mean_block_length
    new_block = rng.random((n_resamples, n_obs)) < p
    new_block[:, 0] = True
    starts = rng.integers(0, n_obs, size=(n_resamples, n_obs), dtype=np.int64)
    pos = np.arange(n_obs, dtype=np.int64)
    start_pos = np.maximum.accumulate(np.where(new_block, pos, 0), axis=1)
    starts_at_block = np.take_along_axis(starts, start_pos, axis=1)
    out: np.ndarray = (starts_at_block + (pos - start_pos)) % n_obs
    return out


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap distributions and the no-skill p-value. Sharpe values are annualized."""

    sharpe_obs: float
    sharpe_ci: tuple[float, float]
    drawdown_obs: float
    drawdown_ci: tuple[float, float]
    p_value_no_skill: float
    mean_block_length: float
    n_resamples: int
    sharpe_distribution: np.ndarray
    drawdown_distribution: np.ndarray


def _sharpe_rows(resamples: np.ndarray, periods: float) -> np.ndarray:
    sd = resamples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: np.ndarray = np.where(
            sd > 0, resamples.mean(axis=1) / sd * math.sqrt(periods), np.nan
        )
    return out


def _drawdown_rows(resamples: np.ndarray) -> np.ndarray:
    equity = np.cumprod(1.0 + resamples, axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    out: np.ndarray = (1.0 - equity / peaks).max(axis=1)
    return out


def bootstrap(
    returns: np.ndarray,
    periods: float,
    n_resamples: int = 5000,
    mean_block_length: float | None = None,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapResult:
    """Stationary-bootstrap CIs for annualized Sharpe and max drawdown, plus
    a p-value against the no-skill (demeaned) null."""
    if len(returns) < MIN_OBS:
        raise InsufficientDataError(
            f"bootstrap needs >= {MIN_OBS} observations, got {len(returns)}"
        )
    if not 0.5 < ci < 1.0:
        raise InvalidInputError(f"ci must be in (0.5, 1), got {ci}")
    if mean_block_length is None:
        mean_block_length = politis_white_block_length(returns)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(len(returns), mean_block_length, n_resamples, rng)

    resamples = returns[idx]
    sharpe_dist = _sharpe_rows(resamples, periods)
    sharpe_dist = sharpe_dist[np.isfinite(sharpe_dist)]
    dd_dist = _drawdown_rows(resamples)

    sharpe_obs = annualized_sharpe(returns, periods)
    null_dist = _sharpe_rows((returns - returns.mean())[idx], periods)
    null_dist = null_dist[np.isfinite(null_dist)]
    p_value = float(1 + np.sum(null_dist >= sharpe_obs)) / (n_resamples + 1)

    lo, hi = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    return BootstrapResult(
        sharpe_obs=sharpe_obs,
        sharpe_ci=(float(np.quantile(sharpe_dist, lo)), float(np.quantile(sharpe_dist, hi))),
        drawdown_obs=max_drawdown(returns),
        drawdown_ci=(float(np.quantile(dd_dist, lo)), float(np.quantile(dd_dist, hi))),
        p_value_no_skill=p_value,
        mean_block_length=float(mean_block_length),
        n_resamples=n_resamples,
        sharpe_distribution=sharpe_dist,
        drawdown_distribution=dd_dist,
    )
