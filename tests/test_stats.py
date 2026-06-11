"""Honest-statistics helpers — pure, runs in CI."""
from mh import stats


def test_wilson_all_success_is_not_one() -> None:
    lo, hi = stats.wilson_ci(16, 16)
    assert hi == 1.0
    assert lo < 0.9  # 16/16 still admits ~0.81 at 95% — not certainty


def test_wilson_widens_as_n_shrinks() -> None:
    wide = stats.wilson_ci(4, 8)
    narrow = stats.wilson_ci(400, 800)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_bootstrap_degenerates_at_all_success_boundary() -> None:
    # documents the pitfall: percentile bootstrap on an all-1 vector lies [1,1]
    assert stats.bootstrap_mean_ci([1.0] * 12) == (1.0, 1.0)


def test_bootstrap_brackets_the_mean_when_mixed() -> None:
    vals = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]  # mean 0.7
    lo, hi = stats.bootstrap_mean_ci(vals, seed=0)
    assert lo < 0.7 < hi


def test_bootstrap_is_deterministic_for_seed() -> None:
    v = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert stats.bootstrap_mean_ci(v, seed=1) == stats.bootstrap_mean_ci(v, seed=1)
