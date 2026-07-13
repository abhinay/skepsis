"""Reference values verified by independent scratch computation, 2026-07-13.

Do NOT edit an expected value to make a test pass — find the formula bug.
"""

import pytest

from skepsis.core.psr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from skepsis.exceptions import InsufficientDataError, InvalidInputError


def test_psr_normal_returns() -> None:
    # sr=0.1 daily periodic, T=1250 (~5y), skew 0, kurt 3, benchmark 0
    v = probabilistic_sharpe_ratio(0.1, 0.0, 1250, 0.0, 3.0)
    assert v == pytest.approx(0.9997885119312112, rel=1e-10)


def test_psr_fat_tails_reduce_confidence() -> None:
    # same Sharpe, negative skew + fat tails -> lower PSR than the normal case
    v = probabilistic_sharpe_ratio(0.1, 0.0, 1250, -2.448, 10.164)
    assert v == pytest.approx(0.999151953389939, rel=1e-10)
    assert v < probabilistic_sharpe_ratio(0.1, 0.0, 1250, 0.0, 3.0)


def test_psr_short_sample() -> None:
    v = probabilistic_sharpe_ratio(0.1, 0.05, 52, -0.5, 4.0)
    assert v == pytest.approx(0.6357900312255157, rel=1e-10)


def test_expected_max_sharpe() -> None:
    assert expected_max_sharpe(0.005, 100) == pytest.approx(0.17894064662732076, rel=1e-10)
    assert expected_max_sharpe(0.005, 1000) == pytest.approx(0.23017184958900594, rel=1e-10)
    assert expected_max_sharpe(0.005, 1) == 0.0  # single trial: no selection bias


def test_dsr_deflates_multiple_testing() -> None:
    # sr=0.1 looks great (PSR ~0.9996) but dies after 100 trials with var 0.005
    v = deflated_sharpe_ratio(0.1, 1250, -1.0, 5.0, 0.005, 100)
    assert v == pytest.approx(0.004048298937119694, rel=1e-9)


def test_dsr_single_trial_equals_psr_vs_zero() -> None:
    v = deflated_sharpe_ratio(0.1, 1250, -1.0, 5.0, 0.005, 1)
    assert v == pytest.approx(0.9996023676999426, rel=1e-10)
    assert v == probabilistic_sharpe_ratio(0.1, 0.0, 1250, -1.0, 5.0)


def test_guards() -> None:
    with pytest.raises(InsufficientDataError):
        probabilistic_sharpe_ratio(0.1, 0.0, 9, 0.0, 3.0)  # documented minimum T >= 10
    with pytest.raises(InvalidInputError, match="denominator"):
        probabilistic_sharpe_ratio(0.3, 0.0, 100, 5.0, 3.0)  # 1 - 5*0.3 + 0.045 < 0
    with pytest.raises(InvalidInputError):
        expected_max_sharpe(-0.001, 10)
    with pytest.raises(InvalidInputError):
        expected_max_sharpe(0.005, 0)
