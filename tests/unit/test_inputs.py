import numpy as np
import pandas as pd
import polars as pl
import pytest

from skepsis import inputs
from skepsis.exceptions import InvalidInputError, MisalignedTrialsError


def test_returns_from_numpy_list_and_column_vector() -> None:
    expected = np.array([0.01, -0.02, 0.03])
    np.testing.assert_array_equal(inputs.coerce_returns([0.01, -0.02, 0.03]), expected)
    np.testing.assert_array_equal(inputs.coerce_returns(expected.reshape(-1, 1)), expected)
    assert inputs.coerce_returns(expected).dtype == np.float64


def test_returns_from_pandas_and_polars() -> None:
    expected = np.array([0.01, -0.02, 0.03])
    np.testing.assert_allclose(inputs.coerce_returns(pd.Series(expected)), expected)
    np.testing.assert_allclose(
        inputs.coerce_returns(pd.DataFrame({"r": expected})), expected
    )
    np.testing.assert_allclose(inputs.coerce_returns(pl.Series("r", expected)), expected)
    np.testing.assert_allclose(
        inputs.coerce_returns(pl.DataFrame({"r": expected})), expected
    )


def test_returns_rejections() -> None:
    with pytest.raises(InvalidInputError, match="NaN"):
        inputs.coerce_returns([0.01, float("nan"), 0.02])
    with pytest.raises(InvalidInputError, match="finite"):
        inputs.coerce_returns([0.01, float("inf")])
    with pytest.raises(InvalidInputError, match="1-D"):
        inputs.coerce_returns(np.zeros((3, 2)))
    with pytest.raises(InvalidInputError, match="numeric"):
        inputs.coerce_returns(["a", "b"])
    with pytest.raises(InvalidInputError, match=">= 2"):
        inputs.coerce_returns([0.01])


def test_trials_labels_and_shape() -> None:
    arr = np.random.default_rng(0).normal(size=(10, 3))
    m, labels = inputs.coerce_trials(arr)
    assert m.shape == (10, 3) and labels == ["trial_0", "trial_1", "trial_2"]
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    m2, labels2 = inputs.coerce_trials(df)
    np.testing.assert_allclose(m2, arr)
    assert labels2 == ["a", "b", "c"]
    m3, labels3 = inputs.coerce_trials(pl.DataFrame(df))
    np.testing.assert_allclose(m3, arr)
    assert labels3 == ["a", "b", "c"]


def test_trials_rejections() -> None:
    with pytest.raises(InvalidInputError, match=">= 2 columns"):
        inputs.coerce_trials(np.zeros((10, 1)))
    with pytest.raises(InvalidInputError, match="2-D"):
        inputs.coerce_trials(np.zeros(10))
    bad = np.zeros((10, 2))
    bad[3, 1] = np.nan
    with pytest.raises(InvalidInputError, match="NaN"):
        inputs.coerce_trials(bad)


def test_params_coercion() -> None:
    df = pd.DataFrame({"window": [10, 20, 30], "threshold": [0.1, 0.2, 0.3]})
    p, names = inputs.coerce_params(df)
    assert p.shape == (3, 2) and names == ["window", "threshold"]
    p2, names2 = inputs.coerce_params(np.array([[1.0], [2.0]]))
    assert names2 == ["param_0"]


def test_alignment() -> None:
    r = np.zeros(10)
    trials = np.zeros((10, 3))
    params = np.zeros((3, 2))
    inputs.validate_alignment(r, trials, params)  # no raise
    with pytest.raises(MisalignedTrialsError, match="rows"):
        inputs.validate_alignment(np.zeros(9), trials, params)
    with pytest.raises(MisalignedTrialsError, match="params"):
        inputs.validate_alignment(r, trials, np.zeros((4, 2)))
