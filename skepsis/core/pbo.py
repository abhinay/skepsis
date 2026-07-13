"""Probability of Backtest Overfitting via CSCV.

Reference: Bailey, Borwein, Lopez de Prado & Zhu (2015), "The Probability of
Backtest Overfitting", Journal of Computational Finance.

CSCV: split the (T, N) trial-returns matrix into S equal time blocks. For every
combination of S/2 blocks used as in-sample (IS): rank trials IS, take the IS
winner, find its relative rank omega among out-of-sample (OOS) scores, and
compute the logit lambda = ln(omega / (1 - omega)). PBO is the fraction of
combinations with lambda <= 0 (the IS winner lands in the bottom half OOS).

The default metric (periodic column Sharpe) runs on a fast path that combines
per-block sums and sums-of-squares instead of materializing submatrices.
Columns with zero variance score -inf so they rank last. A custom `metric`
callback forces the materializing path.
"""

import itertools
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from skepsis.exceptions import InsufficientDataError, InvalidInputError, SkepsisWarning

_MIN_BLOCKS = 4
_MAX_BLOCKS = 24  # C(24,12) ~ 2.7M combinations; documented ceiling


@dataclass(frozen=True)
class PboResult:
    """CSCV output. `value` is the PBO in [0, 1]; `logits` has one entry per combination."""

    value: float
    logits: np.ndarray
    n_combinations: int
    n_blocks: int
    n_trials: int
    n_obs_used: int


def _rank_logit(perf_oos: np.ndarray, n_star: int, n_trials: int) -> float:
    omega = float(stats.rankdata(perf_oos)[n_star]) / (n_trials + 1)
    return math.log(omega / (1.0 - omega))


def cscv(
    trials: np.ndarray,
    n_blocks: int = 16,
    metric: Callable[[np.ndarray], np.ndarray] | None = None,
) -> PboResult:
    """Run CSCV on a (T, N) matrix of per-period trial returns (columns = trials)."""
    if trials.ndim != 2 or trials.shape[1] < 2:
        raise InvalidInputError("trials must be a (T, N) matrix with N >= 2 columns")
    if n_blocks % 2 != 0:
        raise InvalidInputError(f"n_blocks must be even, got {n_blocks}")
    if not _MIN_BLOCKS <= n_blocks <= _MAX_BLOCKS:
        raise InvalidInputError(
            f"n_blocks must be between {_MIN_BLOCKS} and {_MAX_BLOCKS}, got {n_blocks}"
        )
    n_obs, n_trials = trials.shape
    if n_obs < 2 * n_blocks:
        raise InsufficientDataError(
            f"CSCV needs at least 2 * n_blocks = {2 * n_blocks} observations "
            f"(2 rows per block), got {n_obs}"
        )
    trimmed = n_obs % n_blocks
    if trimmed:
        warnings.warn(
            f"trimmed last {trimmed} observation(s) so {n_obs} rows split into "
            f"{n_blocks} equal blocks",
            SkepsisWarning,
            stacklevel=2,
        )
    n_used = n_obs - trimmed
    rows = n_used // n_blocks
    blocks = [trials[i * rows : (i + 1) * rows] for i in range(n_blocks)]
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    logits = np.empty(len(combos))

    if metric is None:
        s1 = np.array([b.sum(axis=0) for b in blocks])  # (S, N) block sums
        s2 = np.array([(b * b).sum(axis=0) for b in blocks])  # (S, N) block sums of squares
        all_blocks = frozenset(range(n_blocks))

        def perf(idx: frozenset[int]) -> np.ndarray:
            n = rows * len(idx)
            sel = list(idx)
            t1 = s1[sel].sum(axis=0)
            t2 = s2[sel].sum(axis=0)
            var = (t2 - t1 * t1 / n) / (n - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                out: np.ndarray = np.where(var > 0, (t1 / n) / np.sqrt(var), -np.inf)
            return out

        for i, combo in enumerate(combos):
            is_set = frozenset(combo)
            n_star = int(np.argmax(perf(is_set)))
            logits[i] = _rank_logit(perf(all_blocks - is_set), n_star, n_trials)
    else:
        for i, combo in enumerate(combos):
            is_set = frozenset(combo)
            m_is = np.vstack([blocks[j] for j in sorted(is_set)])
            m_oos = np.vstack([blocks[j] for j in range(n_blocks) if j not in is_set])
            n_star = int(np.argmax(metric(m_is)))
            logits[i] = _rank_logit(metric(m_oos), n_star, n_trials)

    return PboResult(
        value=float(np.mean(logits <= 0.0)),
        logits=logits,
        n_combinations=len(combos),
        n_blocks=n_blocks,
        n_trials=n_trials,
        n_obs_used=n_used,
    )
