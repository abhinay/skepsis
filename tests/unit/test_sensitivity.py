"""Golden values verified by scratch computation on a 5x5 grid, 2026-07-13."""

import numpy as np
import pytest

from skepsis.core.sensitivity import sensitivity
from skepsis.exceptions import InsufficientDataError, InvalidInputError, SkepsisWarning


def _grid_5x5() -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(5), np.arange(5))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_flat_surface_scores_one() -> None:
    res = sensitivity(_grid_5x5(), np.full(25, 0.8), chosen_index=12)
    assert res.stability_score == pytest.approx(1.0)
    assert res.k == 4 and res.method == "knn"
    assert not res.flagged


def test_isolated_spike_is_flagged() -> None:
    m = np.full(25, 0.5)
    m[12] = 2.0
    res = sensitivity(_grid_5x5(), m, chosen_index=12)
    assert res.stability_score == pytest.approx(4.0)
    assert res.flagged


def test_smooth_plateau_peak_not_flagged() -> None:
    xs, ys = np.meshgrid(np.arange(5), np.arange(5))
    m = np.exp(-((xs.ravel() - 2.0) ** 2 + (ys.ravel() - 2.0) ** 2) / 8.0)
    res = sensitivity(_grid_5x5(), m, chosen_index=12)
    assert res.stability_score == pytest.approx(1.1331484530668263, rel=1e-9)
    assert not res.flagged


def test_one_dimensional_params() -> None:
    p = np.arange(10, dtype=float)
    m = np.full(10, 1.0)
    m[4] = 3.0
    res = sensitivity(p, m, chosen_index=4)
    assert res.stability_score == pytest.approx(3.0)
    assert res.k == 2 and res.flagged


def test_negative_chosen_metric_warns_nan() -> None:
    with pytest.warns(SkepsisWarning, match="non-positive"):
        res = sensitivity(np.arange(10, dtype=float), -np.ones(10), chosen_index=4)
    assert np.isnan(res.stability_score)
    assert not res.flagged


def test_positive_spike_over_nonpositive_neighbors_is_inf_and_flagged() -> None:
    m = np.full(10, -0.5)
    m[4] = 1.0
    with pytest.warns(SkepsisWarning, match="non-positive"):
        res = sensitivity(np.arange(10, dtype=float), m, chosen_index=4)
    assert np.isinf(res.stability_score)
    assert res.flagged


def test_guards() -> None:
    with pytest.raises(InsufficientDataError):
        sensitivity(np.arange(3, dtype=float), np.ones(3), chosen_index=0)
    with pytest.raises(InvalidInputError, match="chosen_index"):
        sensitivity(np.arange(10, dtype=float), np.ones(10), chosen_index=10)
    with pytest.raises(InvalidInputError, match="length"):
        sensitivity(np.arange(10, dtype=float), np.ones(9), chosen_index=0)
