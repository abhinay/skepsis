"""Probabilistic and Deflated Sharpe Ratio.

References:
- Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier",
  Journal of Risk 15(2). [PSR]
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality",
  Journal of Portfolio Management 40(5). [DSR]

Conventions: `sr` and `sr_benchmark` are PERIODIC (non-annualized) Sharpe
ratios; `kurt` is NON-excess kurtosis (normal = 3.0).
"""

import math

from scipy import stats

from skepsis.exceptions import InsufficientDataError, InvalidInputError

_EULER_GAMMA = 0.5772156649015329
_MIN_OBS = 10


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n_obs: int, skew: float, kurt: float
) -> float:
    """P[true SR > sr_benchmark], correcting for sample length, skew and kurtosis.

    PSR = Phi( (sr - sr*) * sqrt(T - 1) / sqrt(1 - skew*sr + (kurt - 1)/4 * sr^2) )
    """
    if n_obs < _MIN_OBS:
        raise InsufficientDataError(f"PSR needs >= {_MIN_OBS} observations, got {n_obs}")
    den_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if den_sq <= 0.0:
        raise InvalidInputError(
            f"PSR denominator is non-positive ({den_sq:.6f}); the skew/kurtosis "
            "estimates are too extreme relative to the Sharpe ratio for the "
            "PSR approximation to hold"
        )
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1.0) / math.sqrt(den_sq)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(var_trial_sr: float, n_trials: int) -> float:
    """E[max periodic SR] across n_trials strategies under the no-skill null.

    E[max SR] ~= sqrt(V) * ((1-gamma) * z(1 - 1/N) + gamma * z(1 - 1/(N*e)))
    where gamma is the Euler-Mascheroni constant. Returns 0.0 for n_trials <= 1
    (a single trial has no selection bias; the formula diverges at N=1).
    """
    if var_trial_sr < 0.0:
        raise InvalidInputError(f"var_trial_sr must be >= 0, got {var_trial_sr}")
    if n_trials < 1:
        raise InvalidInputError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1:
        return 0.0
    sd = math.sqrt(var_trial_sr)
    return sd * (
        (1.0 - _EULER_GAMMA) * float(stats.norm.ppf(1.0 - 1.0 / n_trials))
        + _EULER_GAMMA * float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    )


def deflated_sharpe_ratio(
    sr: float, n_obs: int, skew: float, kurt: float, var_trial_sr: float, n_trials: int
) -> float:
    """PSR evaluated against the expected max Sharpe of n_trials null strategies."""
    sr_star = expected_max_sharpe(var_trial_sr, n_trials)
    return probabilistic_sharpe_ratio(sr, sr_star, n_obs, skew, kurt)
