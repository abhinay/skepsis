"""Statistical behavior of CSCV. Per-dataset PBO on noise is HIGH-VARIANCE
(observed range 0.16-0.86 across seeds), so the noise test averages 10 seeds;
the verified mean is 0.5429."""

import numpy as np

from skepsis.core.pbo import cscv


def test_pbo_of_pure_noise_averages_near_half() -> None:
    vals = [
        cscv(np.random.default_rng(seed).normal(0, 0.01, size=(512, 20)), n_blocks=8).value
        for seed in range(10)
    ]
    assert 0.35 <= float(np.mean(vals)) <= 0.65


def test_pbo_with_one_truly_skilled_trial_is_low() -> None:
    for seed in range(3):
        m = np.random.default_rng(seed).normal(0, 0.01, size=(512, 20))
        m[:, 7] += 0.004  # unmissable daily edge
        assert cscv(m, n_blocks=8).value < 0.1


def test_logit_count_matches_combinations() -> None:
    m = np.random.default_rng(1).normal(size=(64, 4))
    res = cscv(m, n_blocks=8)
    assert res.logits.shape == (70,)  # C(8,4)
    assert res.n_combinations == 70
