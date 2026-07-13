import math

from skepsis.verdict import Thresholds, Verdict, decide


def test_all_clear_is_strong() -> None:
    v = decide(dsr=0.99, dsr_single_trial=False, pbo=0.10, bootstrap_p=0.01, stability_score=1.1)
    assert v.level == "STRONG"
    assert v.reasons == ("all diagnostics within thresholds",)


def test_one_warn_is_moderate() -> None:
    v = decide(dsr=0.99, dsr_single_trial=False, pbo=0.30, bootstrap_p=0.01, stability_score=1.1)
    assert v.level == "MODERATE"
    assert any("PBO" in r for r in v.reasons)


def test_two_warns_is_weak() -> None:
    v = decide(dsr=0.90, dsr_single_trial=False, pbo=0.30, bootstrap_p=0.01, stability_score=1.1)
    assert v.level == "WEAK"
    assert len(v.reasons) == 2


def test_any_fail_is_likely_overfit() -> None:
    v = decide(dsr=0.99, dsr_single_trial=False, pbo=0.55, bootstrap_p=0.01, stability_score=1.1)
    assert v.level == "LIKELY_OVERFIT"


def test_single_trial_caps_at_moderate() -> None:
    v = decide(dsr=0.99, dsr_single_trial=True, pbo=None, bootstrap_p=0.01, stability_score=None)
    assert v.level == "MODERATE"
    assert any("trial count 1" in r for r in v.reasons)


def test_missing_diagnostics_are_ignored() -> None:
    v = decide(dsr=0.99, dsr_single_trial=False, pbo=None, bootstrap_p=None, stability_score=None)
    assert v.level == "STRONG"


def test_infinite_spike_fails_and_nan_warns() -> None:
    inf = decide(dsr=0.99, dsr_single_trial=False, pbo=0.1, bootstrap_p=0.01,
                 stability_score=math.inf)
    assert inf.level == "LIKELY_OVERFIT"
    nan = decide(dsr=0.99, dsr_single_trial=False, pbo=0.1, bootstrap_p=0.01,
                 stability_score=math.nan)
    assert nan.level == "MODERATE"


def test_custom_thresholds() -> None:
    lax = Thresholds(pbo_warn=0.6, pbo_fail=0.9)
    v = decide(dsr=0.99, dsr_single_trial=False, pbo=0.55, bootstrap_p=0.01,
               stability_score=1.1, thresholds=lax)
    assert v.level == "STRONG"
    assert isinstance(v, Verdict)
