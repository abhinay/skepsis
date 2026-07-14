"""Reproduces the numerical example of Bailey & Lopez de Prado (2014),
"The Deflated Sharpe Ratio", Journal of Portfolio Management 40(5),
section "A Numerical Example" (pp. 9-10 of the working-paper PDF at
davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

The paper's strategist discloses: N=100 independent trials, variance of the
trials' (non-annualized, daily) Sharpe ratios V = 1/(2*250) = 0.002, sample
length T=1250 (5 years at 250 obs/year), skewness -3, (non-excess) kurtosis
10, and a best-trial Sharpe of 2.5 annualized = 2.5/sqrt(250) daily.

The paper prints: SR0 ~= 0.1132 (non-annualized) and DSR = 0.9004 < 0.95,
plus two secondary results: DSR = 0.9505 had only N=46 trials been run, and
DSR = 0.9505 under Normal returns (skew 0, kurtosis 3) after N=88 trials.

skepsis reproduces all four printed values (verified 2026-07-14).
"""

import math

import pytest

from skepsis.core.psr import expected_max_sharpe, probabilistic_sharpe_ratio

V_TRIALS = 1.0 / (2.0 * 250.0)          # 0.002, daily (non-annualized)
SR_DAILY = 2.5 / math.sqrt(250.0)       # observed Sharpe, daily
T = 1250


def test_expected_max_sharpe_matches_paper_sr0() -> None:
    assert expected_max_sharpe(V_TRIALS, 100) == pytest.approx(0.1132, abs=5e-5)


def test_dsr_matches_paper() -> None:
    sr0 = expected_max_sharpe(V_TRIALS, 100)
    dsr = probabilistic_sharpe_ratio(SR_DAILY, sr0, T, -3.0, 10.0)
    assert dsr == pytest.approx(0.9004, abs=5e-5)
    assert dsr < 0.95  # the paper's conclusion: not significant at 95%


def test_paper_secondary_claims() -> None:
    dsr_46 = probabilistic_sharpe_ratio(
        SR_DAILY, expected_max_sharpe(V_TRIALS, 46), T, -3.0, 10.0
    )
    assert dsr_46 == pytest.approx(0.9505, abs=5e-5)
    dsr_88_normal = probabilistic_sharpe_ratio(
        SR_DAILY, expected_max_sharpe(V_TRIALS, 88), T, 0.0, 3.0
    )
    assert dsr_88_normal == pytest.approx(0.9505, abs=5e-5)
