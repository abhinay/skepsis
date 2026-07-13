from hypothesis import given
from hypothesis import strategies as st

from skepsis.core.psr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)

# bounded so the PSR denominator stays strictly positive
srs = st.floats(-0.3, 0.3)
skews = st.floats(-2.0, 2.0)
kurts = st.floats(2.0, 12.0)
ns = st.integers(10, 100_000)
trial_counts = st.integers(2, 10_000)
variances = st.floats(1e-6, 1.0)


@given(sr=srs, bench=srs, n=ns, skew=skews, kurt=kurts)
def test_psr_is_probability(sr: float, bench: float, n: int, skew: float, kurt: float) -> None:
    assert 0.0 <= probabilistic_sharpe_ratio(sr, bench, n, skew, kurt) <= 1.0


@given(var=variances, n=trial_counts)
def test_expected_max_sharpe_monotone_in_trials(var: float, n: int) -> None:
    assert expected_max_sharpe(var, n + 1) >= expected_max_sharpe(var, n) - 1e-12


@given(sr=srs, n=ns, skew=skews, kurt=kurts, var=variances, k=trial_counts)
def test_more_trials_never_increase_dsr(
    sr: float, n: int, skew: float, kurt: float, var: float, k: int
) -> None:
    assert (
        deflated_sharpe_ratio(sr, n, skew, kurt, var, k + 1)
        <= deflated_sharpe_ratio(sr, n, skew, kurt, var, k) + 1e-12
    )
