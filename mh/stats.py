"""Small honest-statistics helpers: confidence intervals + power flags.

Reporting Brier/ECE/accuracy to 3-4 decimals on n=16 is false precision. These
helpers attach a 95% Wilson interval to a proportion and flag estimates whose
sample is too small for the metric (ECE needs many points per bin).
"""

from __future__ import annotations

import math


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def accuracy_with_ci(correct: int, n: int) -> dict:
    lo, hi = wilson_ci(correct, n)
    return {"accuracy": round(correct / n, 4) if n else 0.0, "n": n, "ci95": [lo, hi]}


def ece_is_underpowered(n: int, bins: int, min_per_bin: int = 5) -> bool:
    """ECE on n points across `bins` bins is unreliable when n < bins*min_per_bin."""
    return n < bins * min_per_bin


def bootstrap_mean_ci(
    values: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values` (e.g. a 0/1 correctness
    vector). Distribution-free; works for routed/ensemble estimates where a
    closed-form interval does not exist. Wide intervals at small n are the honest
    signal — a benchmark point estimate must not be mistaken for generalisation."""
    import numpy as np

    if not values:
        return (0.0, 1.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (round(float(lo), 4), round(float(hi), 4))
