"""Coerce user input (numpy / pandas / polars / sequences) into validated numpy arrays.

pandas and polars are OPTIONAL dependencies: this module must import without them,
so detection is duck-typed (`.to_numpy()`, `.columns`) rather than isinstance-based.
NaN/inf are rejected, never dropped — align and clean your data upstream.
"""

from typing import Any

import numpy as np

from skepsis.exceptions import InvalidInputError, MisalignedTrialsError


def _to_numpy(obj: Any, name: str) -> np.ndarray:
    raw = obj.to_numpy() if hasattr(obj, "to_numpy") else obj
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        raise InvalidInputError(f"{name} must be numeric") from None
    return arr


def _check_finite(arr: np.ndarray, name: str) -> None:
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise InvalidInputError(
            f"{name} must be finite: found {n_nan} NaN and {n_inf} inf values. "
            "skepsis never drops or fills values — align and clean your data first."
        )


def coerce_returns(obj: Any, name: str = "returns") -> np.ndarray:
    """1-D float64 returns vector, finite, length >= 2. Accepts (T,) or (T,1)."""
    arr = _to_numpy(obj, name)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise InvalidInputError(f"{name} must be 1-D (or a single column), got shape {arr.shape}")
    if arr.size < 2:
        raise InvalidInputError(f"{name} needs >= 2 observations, got {arr.size}")
    _check_finite(arr, name)
    return arr


def _column_labels(obj: Any, n: int, prefix: str) -> list[str]:
    cols = getattr(obj, "columns", None)
    if cols is not None:
        return [str(c) for c in list(cols)]
    return [f"{prefix}_{i}" for i in range(n)]


def coerce_trials(obj: Any) -> tuple[np.ndarray, list[str]]:
    """(T, N) float64 trials matrix (columns = trials), N >= 2, plus column labels."""
    arr = _to_numpy(obj, "trials")
    if arr.ndim != 2:
        raise InvalidInputError(f"trials must be 2-D (T x N), got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise InvalidInputError(
            f"trials needs >= 2 columns (one per variant tried), got {arr.shape[1]}"
        )
    _check_finite(arr, "trials")
    return arr, _column_labels(obj, arr.shape[1], "trial")


def coerce_params(obj: Any) -> tuple[np.ndarray, list[str]]:
    """(n, d) float64 parameter matrix (one row per trial), plus parameter names."""
    arr = _to_numpy(obj, "params")
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise InvalidInputError(f"params must be 2-D (n x d), got shape {arr.shape}")
    _check_finite(arr, "params")
    return arr, _column_labels(obj, arr.shape[1], "param")


def validate_alignment(
    returns: np.ndarray, trials: np.ndarray | None, params: np.ndarray | None
) -> None:
    """Raise MisalignedTrialsError when returns/trials/params dimensions disagree."""
    if trials is not None and trials.shape[0] != returns.shape[0]:
        raise MisalignedTrialsError(
            f"trials has {trials.shape[0]} rows but returns has {returns.shape[0]} "
            "observations; they must cover the same periods"
        )
    if params is not None:
        if trials is None:
            raise MisalignedTrialsError("params were provided without trials")
        if params.shape[0] != trials.shape[1]:
            raise MisalignedTrialsError(
                f"params has {params.shape[0]} rows but trials has {trials.shape[1]} "
                "columns; params needs one row per trial"
            )
