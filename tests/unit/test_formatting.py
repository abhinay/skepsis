import math

from skepsis.formatting import count_trials, format_stability


def test_format_stability_finite() -> None:
    assert format_stability(1.7) == "1.70"


def test_format_stability_inf_and_nan() -> None:
    assert format_stability(math.inf) == "∞ (isolated spike)"
    assert format_stability(math.nan) == "undefined (chosen metric non-positive)"


def test_count_trials_grammar() -> None:
    assert count_trials(1) == "1 trial"
    assert count_trials(34) == "34 trials"
