"""Orchestration: run every diagnostic the inputs allow, assemble a Result."""

import math
import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from skepsis import __version__
from skepsis.core import moments
from skepsis.core.bootstrap import MIN_OBS as _MIN_BOOTSTRAP_OBS
from skepsis.core.bootstrap import BootstrapResult, bootstrap
from skepsis.core.pbo import PboResult, cscv, validate_n_blocks
from skepsis.core.psr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from skepsis.core.sensitivity import SensitivityResult, sensitivity
from skepsis.exceptions import InvalidInputError, SkepsisWarning
from skepsis.formatting import count_trials, format_stability
from skepsis.inputs import coerce_params, coerce_returns, coerce_trials, validate_alignment
from skepsis.verdict import Thresholds, Verdict, decide

_AUTOCORR_WARN_BLOCK_LENGTH = 10.0
"""Politis-White mean block length above which returns are considered heavily
autocorrelated; PSR/DSR assume IID-ish returns, so skepsis warns."""


def _finite_or_none(x: float) -> float | None:
    """Map non-finite floats to None so to_dict emits strict RFC-compliant JSON."""
    return x if math.isfinite(x) else None


@dataclass(frozen=True)
class PsrResult:
    """Probabilistic Sharpe Ratio vs a zero-skill benchmark."""

    value: float
    sharpe_periodic: float
    sharpe_annualized: float
    benchmark_sr: float
    n_obs: int
    skewness: float
    kurtosis: float


@dataclass(frozen=True)
class DsrResult:
    """Deflated Sharpe Ratio: PSR vs the expected max Sharpe of the trials."""

    value: float
    p_value: float
    benchmark_sr: float
    n_trials: int
    var_trial_sr: float
    single_trial: bool


@dataclass
class Result:
    """Everything skepsis concluded, plus the data the HTML report needs."""

    psr: PsrResult
    deflated_sharpe: DsrResult
    pbo: PboResult | None
    bootstrap: BootstrapResult | None
    sensitivity: SensitivityResult | None
    verdict: Verdict
    skipped: dict[str, str]
    warnings: list[str]
    meta: dict[str, Any]
    returns: np.ndarray
    params: np.ndarray | None = None
    param_names: list[str] | None = None
    trial_metrics: np.ndarray | None = None
    thresholds: Thresholds = field(default_factory=Thresholds)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary (scalars only, no arrays)."""
        d: dict[str, Any] = {
            "psr": {
                "value": self.psr.value,
                "sharpe_periodic": self.psr.sharpe_periodic,
                "sharpe_annualized": self.psr.sharpe_annualized,
                "n_obs": self.psr.n_obs,
                "skewness": self.psr.skewness,
                "kurtosis": self.psr.kurtosis,
            },
            "deflated_sharpe": {
                "value": self.deflated_sharpe.value,
                "p_value": self.deflated_sharpe.p_value,
                "benchmark_sr": self.deflated_sharpe.benchmark_sr,
                "n_trials": self.deflated_sharpe.n_trials,
                "single_trial": self.deflated_sharpe.single_trial,
            },
            "verdict": {"level": self.verdict.level, "reasons": list(self.verdict.reasons)},
            "skipped": dict(self.skipped),
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
        }
        if self.pbo is not None:
            d["pbo"] = {
                "value": self.pbo.value,
                "n_combinations": self.pbo.n_combinations,
                "n_blocks": self.pbo.n_blocks,
                "n_trials": self.pbo.n_trials,
            }
        if self.bootstrap is not None:
            d["bootstrap"] = {
                "sharpe_obs": self.bootstrap.sharpe_obs,
                "sharpe_ci": list(self.bootstrap.sharpe_ci),
                "drawdown_obs": self.bootstrap.drawdown_obs,
                "drawdown_ci": list(self.bootstrap.drawdown_ci),
                "p_value_no_skill": self.bootstrap.p_value_no_skill,
                "mean_block_length": self.bootstrap.mean_block_length,
            }
        if self.sensitivity is not None:
            d["sensitivity"] = {
                "stability_score": _finite_or_none(self.sensitivity.stability_score),
                "neighbor_median": _finite_or_none(self.sensitivity.neighbor_median),
                "k": self.sensitivity.k,
                "flagged": self.sensitivity.flagged,
            }
        return d

    def summary(self) -> str:
        lines = [
            f"skepsis {self.meta['skepsis_version']} — verdict: {self.verdict.level}",
            f"  annualized Sharpe: {self.psr.sharpe_annualized:.3f}  "
            f"(PSR {self.psr.value:.3f}, DSR {self.deflated_sharpe.value:.3f} "
            f"over {count_trials(self.deflated_sharpe.n_trials)})",
        ]
        if self.pbo is not None:
            lines.append(f"  PBO: {self.pbo.value:.3f} ({self.pbo.n_combinations} combinations)")
        if self.bootstrap is not None:
            lines.append(
                f"  bootstrap no-skill p: {self.bootstrap.p_value_no_skill:.4f}, "
                f"Sharpe 95% CI [{self.bootstrap.sharpe_ci[0]:.2f}, "
                f"{self.bootstrap.sharpe_ci[1]:.2f}]"
            )
        if self.sensitivity is not None:
            lines.append(
                f"  stability score: {format_stability(self.sensitivity.stability_score)}"
            )
        for reason in self.verdict.reasons:
            lines.append(f"  - {reason}")
        for name, why in self.skipped.items():
            lines.append(f"  skipped {name}: {why}")
        return "\n".join(lines)

    def save_html(self, path: str | Path) -> Path:
        """Render the self-contained HTML report (report module lands in Task 10)."""
        from skepsis.report.render import render_html

        out = Path(path)
        out.write_text(render_html(self), encoding="utf-8")
        return out


def _column_sharpes(trials: np.ndarray, warn: bool = True) -> np.ndarray:
    """Periodic Sharpe per column; zero-variance columns score -inf (warned)."""
    sd = trials.std(axis=0, ddof=1)
    mean = trials.mean(axis=0)
    zero = sd == 0.0
    if zero.any() and warn:
        _warnings.warn(
            f"{int(zero.sum())} trial column(s) have zero variance; they score -inf",
            SkepsisWarning,
            stacklevel=3,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        out: np.ndarray = np.where(zero, -np.inf, mean / np.where(zero, 1.0, sd))
    return out


def _find_chosen(
    returns: np.ndarray,
    trials: np.ndarray,
    labels: list[str],
    chosen: int | str | None,
    trial_metrics: np.ndarray,
) -> tuple[int, str]:
    if isinstance(chosen, str):
        if chosen not in labels:
            raise InvalidInputError(f"chosen label {chosen!r} not in trials columns {labels}")
        return labels.index(chosen), chosen
    if isinstance(chosen, int) and not isinstance(chosen, bool):
        if not 0 <= chosen < trials.shape[1]:
            raise InvalidInputError(f"chosen index {chosen} out of range")
        return chosen, labels[chosen]
    matches = [j for j in range(trials.shape[1]) if np.allclose(trials[:, j], returns)]
    if len(matches) == 1:
        return matches[0], labels[matches[0]]
    j = int(np.argmax(trial_metrics))
    _warnings.warn(
        "could not uniquely match `returns` to a trials column "
        f"({len(matches)} matches); using best-metric trial {labels[j]!r} for "
        "sensitivity — pass `chosen=` to be explicit",
        SkepsisWarning,
        stacklevel=3,
    )
    return j, labels[j]


def evaluate(
    returns: Any,
    trials: Any | None = None,
    params: Any | None = None,
    freq: str | int | float = "daily",
    chosen: int | str | None = None,
    pbo_blocks: int = 16,
    n_resamples: int = 5000,
    seed: int = 0,
    thresholds: Thresholds | None = None,
) -> Result:
    """Run every overfitting diagnostic the provided inputs allow. See README."""
    periods = moments.periods_per_year(freq)
    validate_n_blocks(pbo_blocks)
    r = coerce_returns(returns)
    trials_arr: np.ndarray | None = None
    labels: list[str] = []
    params_arr: np.ndarray | None = None
    param_names: list[str] | None = None
    if trials is not None:
        trials_arr, labels = coerce_trials(trials)
    if params is not None:
        params_arr, param_names = coerce_params(params)
    validate_alignment(r, trials_arr, params_arr)

    skipped: dict[str, str] = {}
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")

        if chosen is not None and params_arr is None:
            _warnings.warn(
                "`chosen=` was provided without `params=`; it only affects the "
                "sensitivity diagnostic and is ignored",
                SkepsisWarning,
                stacklevel=2,
            )

        n_obs = len(r)
        sr_p = moments.sharpe(r)
        skew = moments.skewness(r)
        kurt = moments.kurtosis(r)
        psr_value = probabilistic_sharpe_ratio(sr_p, 0.0, n_obs, skew, kurt)
        psr_res = PsrResult(
            value=psr_value,
            sharpe_periodic=sr_p,
            sharpe_annualized=moments.annualized_sharpe(r, periods),
            benchmark_sr=0.0,
            n_obs=n_obs,
            skewness=skew,
            kurtosis=kurt,
        )

        trial_metrics: np.ndarray | None = None
        chosen_label: str | None = None
        if trials_arr is not None:
            n_trials = trials_arr.shape[1]
            col_sharpes = _column_sharpes(trials_arr)
            finite = col_sharpes[np.isfinite(col_sharpes)]
            var_sr = float(np.var(finite, ddof=1)) if len(finite) >= 2 else 0.0
            if var_sr == 0.0:
                _warnings.warn(
                    "variance of trial Sharpe ratios is zero; DSR benchmark falls "
                    "back to 0 (no deflation)",
                    SkepsisWarning,
                    stacklevel=2,
                )
            single_trial = False
        else:
            n_trials, var_sr, single_trial = 1, 0.0, True
            _warnings.warn(
                "no trials provided — DSR uses trial count 1, which is almost "
                "certainly optimistic; pass `trials=` with every variant you tried",
                SkepsisWarning,
                stacklevel=2,
            )
        dsr_value = deflated_sharpe_ratio(sr_p, n_obs, skew, kurt, var_sr, n_trials)
        dsr_res = DsrResult(
            value=dsr_value,
            p_value=1.0 - dsr_value,
            benchmark_sr=expected_max_sharpe(var_sr, n_trials),
            n_trials=n_trials,
            var_trial_sr=var_sr,
            single_trial=single_trial,
        )

        pbo_res: PboResult | None = None
        if trials_arr is None:
            skipped["pbo"] = "trials not provided (pass the returns of every variant tried)"
        elif n_obs < 2 * pbo_blocks:
            skipped["pbo"] = (
                f"needs >= {2 * pbo_blocks} observations for n_blocks={pbo_blocks}, "
                f"got {n_obs}; pass a smaller pbo_blocks explicitly"
            )
        else:
            pbo_res = cscv(trials_arr, n_blocks=pbo_blocks)

        boot_res: BootstrapResult | None = None
        if n_obs < _MIN_BOOTSTRAP_OBS:
            skipped["bootstrap"] = f"needs >= {_MIN_BOOTSTRAP_OBS} observations, got {n_obs}"
        else:
            boot_res = bootstrap(r, periods, n_resamples=n_resamples, seed=seed)
            if boot_res.mean_block_length > _AUTOCORR_WARN_BLOCK_LENGTH:
                _warnings.warn(
                    f"estimated mean block length {boot_res.mean_block_length:.1f} "
                    f"exceeds {_AUTOCORR_WARN_BLOCK_LENGTH:.0f}: returns are heavily "
                    "autocorrelated, which strains the IID-ish assumptions behind "
                    "PSR/DSR — read those diagnostics with extra skepticism",
                    SkepsisWarning,
                    stacklevel=2,
                )

        sens_res: SensitivityResult | None = None
        if params_arr is None:
            skipped["sensitivity"] = "params not provided (one row per trial)"
        else:
            assert trials_arr is not None  # validate_alignment guarantees this
            metrics = _column_sharpes(trials_arr, warn=False) * float(np.sqrt(periods))
            chosen_idx, chosen_label = _find_chosen(r, trials_arr, labels, chosen, metrics)
            trial_metrics = metrics
            sens_res = sensitivity(params_arr, metrics, chosen_idx)

    warning_msgs = [str(w.message) for w in caught if issubclass(w.category, SkepsisWarning)]
    for w in caught:
        _warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    verdict = decide(
        dsr=dsr_res.value,
        dsr_single_trial=dsr_res.single_trial,
        pbo=pbo_res.value if pbo_res else None,
        bootstrap_p=boot_res.p_value_no_skill if boot_res else None,
        stability_score=sens_res.stability_score if sens_res else None,
        thresholds=thresholds,
    )
    return Result(
        psr=psr_res,
        deflated_sharpe=dsr_res,
        pbo=pbo_res,
        bootstrap=boot_res,
        sensitivity=sens_res,
        verdict=verdict,
        skipped=skipped,
        warnings=warning_msgs,
        meta={
            "freq": freq,
            "periods_per_year": periods,
            "n_obs": n_obs,
            "n_trials": n_trials,
            "chosen_label": chosen_label,
            "skepsis_version": __version__,
        },
        returns=r,
        params=params_arr,
        param_names=param_names,
        trial_metrics=trial_metrics,
        thresholds=thresholds or Thresholds(),
    )
