"""Parameter-neighborhood stability: is the chosen configuration a plateau or a spike?

Method: k = min(2*d, n-1) nearest neighbors of the chosen configuration in
z-scored parameter space (for interior points of a regular grid these are the
orthogonal grid neighbors). stability_score = chosen_metric / median(neighbor
metrics): ~1.0 on a plateau, >> 1.0 on an isolated spike (fitted to noise).

Edge conventions:
- chosen metric <= 0        -> score = nan, SkepsisWarning (a non-positive chosen
                               metric is its own problem; the ratio is meaningless)
- neighbor median <= 0 < chosen -> score = inf, flagged, SkepsisWarning
"""

import warnings
from dataclasses import dataclass

import numpy as np

from skepsis.exceptions import InsufficientDataError, InvalidInputError, SkepsisWarning

_MIN_TRIALS = 4


@dataclass(frozen=True)
class SensitivityResult:
    """Neighborhood stability of the chosen parameter configuration."""

    stability_score: float
    chosen_index: int
    k: int
    neighbor_indices: np.ndarray
    neighbor_median: float
    flagged: bool
    method: str = "knn"


def sensitivity(
    params: np.ndarray,
    metrics: np.ndarray,
    chosen_index: int,
    spike_threshold: float = 1.5,
) -> SensitivityResult:
    """Score the chosen configuration against its k nearest parameter neighbors."""
    p = np.asarray(params, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    m = np.asarray(metrics, dtype=np.float64)
    n, d = p.shape
    if n < _MIN_TRIALS:
        raise InsufficientDataError(f"sensitivity needs >= {_MIN_TRIALS} trials, got {n}")
    if m.shape != (n,):
        raise InvalidInputError(f"metrics length {m.shape} must match params rows ({n},)")
    if not 0 <= chosen_index < n:
        raise InvalidInputError(f"chosen_index {chosen_index} out of range [0, {n})")

    sd = p.std(axis=0)
    sd[sd == 0.0] = 1.0  # constant parameter column: distance contribution 0
    z = (p - p.mean(axis=0)) / sd
    dists = np.linalg.norm(z - z[chosen_index], axis=1)
    order = np.argsort(dists)
    k = min(2 * d, n - 1)
    neighbors = np.array([i for i in order if i != chosen_index][:k])
    neighbor_median = float(np.median(m[neighbors]))
    chosen = float(m[chosen_index])

    if chosen <= 0.0:
        warnings.warn(
            f"chosen configuration has non-positive metric ({chosen:.4f}); "
            "stability score is undefined",
            SkepsisWarning,
            stacklevel=2,
        )
        score, flagged = float("nan"), False
    elif neighbor_median <= 0.0:
        warnings.warn(
            f"neighbors have non-positive median metric ({neighbor_median:.4f}) while the "
            "chosen configuration is positive — an extreme isolated spike",
            SkepsisWarning,
            stacklevel=2,
        )
        score, flagged = float("inf"), True
    else:
        score = chosen / neighbor_median
        flagged = score > spike_threshold

    return SensitivityResult(
        stability_score=score,
        chosen_index=int(chosen_index),
        k=int(k),
        neighbor_indices=neighbors,
        neighbor_median=neighbor_median,
        flagged=flagged,
    )
