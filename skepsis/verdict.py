"""Rule-based verdict aggregation. The verdict is a labeled HEURISTIC:
every threshold is documented here, overridable, and every fired rule is
reported. It never replaces reading the diagnostics themselves."""

import math
from dataclasses import dataclass

from skepsis.formatting import format_stability

_LEVELS = ("STRONG", "MODERATE", "WEAK", "LIKELY_OVERFIT")


@dataclass(frozen=True)
class Thresholds:
    """Default rule thresholds. fail => LIKELY_OVERFIT; warn counts toward WEAK/MODERATE."""

    dsr_fail: float = 0.5    # DSR below this: worse than a coin flip after deflation
    dsr_warn: float = 0.95
    pbo_fail: float = 0.5    # in-sample winner lands bottom-half OOS more often than not
    pbo_warn: float = 0.2
    bootstrap_fail: float = 0.5   # no-skill p-value above this
    bootstrap_warn: float = 0.05
    sensitivity_fail: float = 2.0  # chosen config > 2x its neighbors' median
    sensitivity_warn: float = 1.5


@dataclass(frozen=True)
class Verdict:
    level: str
    reasons: tuple[str, ...]


def decide(
    dsr: float | None,
    dsr_single_trial: bool,
    pbo: float | None,
    bootstrap_p: float | None,
    stability_score: float | None,
    thresholds: Thresholds | None = None,
) -> Verdict:
    """Aggregate available diagnostics into a verdict. None = diagnostic skipped."""
    t = thresholds or Thresholds()
    fails: list[str] = []
    warns: list[str] = []

    if dsr is not None:
        if dsr < t.dsr_fail:
            fails.append(f"DSR {dsr:.3f} < {t.dsr_fail} (fail)")
        elif dsr < t.dsr_warn:
            warns.append(f"DSR {dsr:.3f} < {t.dsr_warn} (warn)")
        if dsr_single_trial:
            warns.append(
                "DSR computed with trial count 1 (no trials provided) — almost "
                "certainly optimistic"
            )
    if pbo is not None:
        if pbo > t.pbo_fail:
            fails.append(f"PBO {pbo:.3f} > {t.pbo_fail} (fail)")
        elif pbo > t.pbo_warn:
            warns.append(f"PBO {pbo:.3f} > {t.pbo_warn} (warn)")
    if bootstrap_p is not None:
        if bootstrap_p > t.bootstrap_fail:
            fails.append(f"bootstrap no-skill p {bootstrap_p:.3f} > {t.bootstrap_fail} (fail)")
        elif bootstrap_p > t.bootstrap_warn:
            warns.append(f"bootstrap no-skill p {bootstrap_p:.3f} > {t.bootstrap_warn} (warn)")
    if stability_score is not None:
        if math.isnan(stability_score):
            warns.append("sensitivity undefined: chosen configuration has non-positive metric")
        elif stability_score > t.sensitivity_fail:  # inf lands here
            fails.append(
                f"stability score {format_stability(stability_score)} > "
                f"{t.sensitivity_fail} (fail)"
            )
        elif stability_score > t.sensitivity_warn:
            warns.append(
                f"stability score {format_stability(stability_score)} > "
                f"{t.sensitivity_warn} (warn)"
            )

    if fails:
        level = "LIKELY_OVERFIT"
    elif len(warns) >= 2:
        level = "WEAK"
    elif len(warns) == 1:
        level = "MODERATE"
    else:
        level = "STRONG"
    reasons = tuple(fails + warns) or ("all diagnostics within thresholds",)
    assert level in _LEVELS
    return Verdict(level=level, reasons=reasons)
