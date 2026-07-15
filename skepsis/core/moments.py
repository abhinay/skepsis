"""Shared moment helpers: Sharpe, skewness, kurtosis, drawdown, annualization.

Conventions (used across skepsis):
- "periodic Sharpe" = mean(r) / std(r, ddof=1) on per-period returns, NOT annualized.
- kurtosis is non-excess (normal = 3.0).
- Core functions assume finite 1-D float input (validated upstream in skepsis.inputs)
  but still guard degenerate statistics.

Reference: Sharpe, W. F. (1994), "The Sharpe Ratio", Journal of Portfolio
Management 21(1), for the Sharpe ratio convention.
"""

import math

import numpy as np
from scipy import stats

from skepsis.exceptions import InsufficientDataError, InvalidFrequencyError, InvalidInputError

_FREQ_MAP: dict[str, float] = {
    "hourly": 1638.0,  # 252 trading days x 6.5 US market hours
    "daily": 252.0,
    "weekly": 52.0,
    "monthly": 12.0,
}


def periods_per_year(freq: str | int | float) -> float:
    """Map a freq label to periods per year, or pass a positive number through."""
    if isinstance(freq, str):
        try:
            return _FREQ_MAP[freq]
        except KeyError:
            raise InvalidFrequencyError(
                f"freq must be one of {sorted(_FREQ_MAP)} or a positive number, got {freq!r}"
            ) from None
    if (
        isinstance(freq, (int, float))
        and not isinstance(freq, bool)
        and math.isfinite(freq)
        and freq > 0
    ):
        return float(freq)
    raise InvalidFrequencyError(f"freq must be a positive number, got {freq!r}")


def sharpe(returns: np.ndarray) -> float:
    """Periodic (non-annualized) Sharpe ratio: mean / std(ddof=1).

    Constancy is checked by exact equality (`np.all(x == x[0])`), never
    `std == 0`: float64 reductions over a constant array can leave a residual
    of ~1e-18 rather than an exact zero, which would let a genuinely
    degenerate series slip past a `std == 0.0` check.
    """
    if returns.size < 2:
        raise InsufficientDataError(f"sharpe needs >= 2 observations, got {returns.size}")
    if bool(np.all(returns == returns[0])):
        raise InvalidInputError("returns are constant (zero variance); Sharpe is undefined")
    sd = float(np.std(returns, ddof=1))
    return float(np.mean(returns)) / sd


def annualized_sharpe(returns: np.ndarray, periods: float) -> float:
    """Periodic Sharpe scaled by sqrt(periods per year)."""
    if not math.isfinite(periods) or periods <= 0:
        raise InvalidInputError(f"periods must be finite and > 0, got {periods}")
    return sharpe(returns) * float(np.sqrt(periods))


def skewness(returns: np.ndarray) -> float:
    """Sample skewness (biased, scipy default)."""
    return float(stats.skew(returns, bias=True))


def kurtosis(returns: np.ndarray) -> float:
    """Sample kurtosis, NON-excess (normal = 3.0)."""
    return float(stats.kurtosis(returns, fisher=False, bias=True))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough decline of the compounded equity curve.

    Returns a positive fraction (0.25 == a 25% drawdown). 0.0 if equity never falls.
    """
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum(np.maximum.accumulate(equity), 1.0)
    drawdowns = 1.0 - equity / peaks
    return float(drawdowns.max(initial=0.0))
