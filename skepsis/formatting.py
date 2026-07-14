"""Shared human-facing formatting for diagnostic values.

One helper per convention so the report, summary(), and verdict reasons can
never drift apart (spec 2026-07-14 §3.2.5: all user-visible stability text)."""

import math


def format_stability(score: float) -> str:
    """Finite → 2dp; inf → labeled spike; nan → labeled undefined."""
    if math.isnan(score):
        return "undefined (chosen metric non-positive)"
    if math.isinf(score):
        return "∞ (isolated spike)"
    return f"{score:.2f}"


def count_trials(n: int) -> str:
    """Grammatical trial count: '1 trial', '34 trials'."""
    return "1 trial" if n == 1 else f"{n} trials"
