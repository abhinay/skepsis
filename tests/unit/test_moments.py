import math

import numpy as np
import pytest

from skepsis.core import moments
from skepsis.exceptions import InsufficientDataError, InvalidFrequencyError, InvalidInputError


def test_periods_per_year_mapping() -> None:
    assert moments.periods_per_year("daily") == 252.0
    assert moments.periods_per_year("weekly") == 52.0
    assert moments.periods_per_year("monthly") == 12.0
    assert moments.periods_per_year("hourly") == 1638.0
    assert moments.periods_per_year(365) == 365.0


def test_periods_per_year_rejects_unknown() -> None:
    with pytest.raises(InvalidFrequencyError):
        moments.periods_per_year("fortnightly")
    with pytest.raises(InvalidFrequencyError):
        moments.periods_per_year(0)


def test_sharpe_hand_case() -> None:
    r = np.array([0.01, 0.02, 0.03])  # mean 0.02, std(ddof=1) 0.01
    assert moments.sharpe(r) == pytest.approx(2.0)
    assert moments.annualized_sharpe(r, 252.0) == pytest.approx(2.0 * math.sqrt(252.0))


def test_sharpe_guards() -> None:
    with pytest.raises(InsufficientDataError):
        moments.sharpe(np.array([0.01]))
    with pytest.raises(InvalidInputError):
        moments.sharpe(np.array([0.01, 0.01, 0.01]))  # zero variance


def test_skew_kurt_hand_case() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert moments.skewness(x) == pytest.approx(0.0, abs=1e-12)
    # population moments: m2=2, m4=6.8 -> kurtosis (non-excess) = 6.8/4 = 1.7
    assert moments.kurtosis(x) == pytest.approx(1.7)


def test_max_drawdown_hand_case() -> None:
    r = np.array([0.10, -0.05, 0.03, -0.20, 0.08])
    # peak 1.10 -> trough 1.10*0.95*1.03*0.80; dd = 1 - 0.95*1.03*0.80 = 0.2172
    assert moments.max_drawdown(r) == pytest.approx(0.2172, abs=1e-12)


def test_max_drawdown_monotone_gains() -> None:
    assert moments.max_drawdown(np.array([0.01, 0.02, 0.03])) == 0.0
