"""Reference values verified by independent scratch computation, 2026-07-13.

The tiny case is fully deterministic and tie-free: every logit is exactly
ln(1/3) or 0.0. Do NOT edit expected values to make tests pass.
"""

import math

import numpy as np
import pytest

from skepsis.core.pbo import cscv
from skepsis.exceptions import InsufficientDataError, InvalidInputError

TINY = np.array(
    [
        [0.011, 0.020, -0.010],
        [0.020, -0.011, 0.012],
        [-0.010, 0.031, 0.021],
        [0.012, 0.010, -0.022],
        [0.030, -0.020, 0.013],
        [-0.021, 0.022, 0.032],
        [0.013, 0.011, -0.012],
        [0.022, -0.012, 0.023],
    ]
)


def test_tiny_deterministic_case() -> None:
    res = cscv(TINY, n_blocks=4)
    assert res.value == 1.0
    assert res.n_combinations == 6
    assert res.n_blocks == 4 and res.n_trials == 3 and res.n_obs_used == 8
    expected = np.sort(np.array([math.log(1 / 3)] * 5 + [0.0]))
    np.testing.assert_allclose(np.sort(res.logits), expected, atol=1e-12)


def test_random_case_and_path_agreement() -> None:
    rng = np.random.default_rng(123)
    m = rng.normal(0, 0.01, size=(200, 12))
    fast = cscv(m, n_blocks=8)
    assert fast.value == pytest.approx(0.8)
    assert fast.n_combinations == 70

    def sharpe_cols(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=0) / x.std(axis=0, ddof=1)

    slow = cscv(m, n_blocks=8, metric=sharpe_cols)
    assert slow.value == fast.value
    np.testing.assert_allclose(slow.logits, fast.logits, atol=1e-12)


def test_validation() -> None:
    with pytest.raises(InvalidInputError, match="even"):
        cscv(TINY, n_blocks=5)
    with pytest.raises(InvalidInputError, match="between 4 and 24"):
        cscv(TINY, n_blocks=2)
    with pytest.raises(InvalidInputError, match=">= 2"):
        cscv(TINY[:, :1], n_blocks=4)
    with pytest.raises(InsufficientDataError, match="2 \\* n_blocks"):
        cscv(TINY[:7], n_blocks=4)  # 7 rows < 2*4


def test_trim_warns() -> None:
    rng = np.random.default_rng(0)
    m = rng.normal(size=(18, 3))  # 18 % 4 = 2 rows trimmed
    with pytest.warns(UserWarning, match="trimmed"):
        res = cscv(m, n_blocks=4)
    assert res.n_obs_used == 16


def test_constant_column_ranks_last() -> None:
    rng = np.random.default_rng(4)
    m = rng.normal(0, 0.01, size=(64, 4))
    m[:, 2] = 0.01  # exactly constant column
    res = cscv(m, n_blocks=8)
    assert np.isfinite(res.logits).all()
    assert 0.0 <= res.value <= 1.0
