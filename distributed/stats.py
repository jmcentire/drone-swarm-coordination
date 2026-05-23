# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Statistical helpers for benches: Wilson CIs, bootstrap CIs."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion. Stable at extremes."""
    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import norm
    p = successes / n
    z = norm.ppf(1 - alpha / 2)
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - half) / denom
    hi = (centre + half) / denom
    return (max(0.0, lo), min(1.0, hi))


def bootstrap_ci(
    values: list[float] | np.ndarray, alpha: float = 0.05, n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI for the mean. Returns (mean, lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    means = np.zeros(n_boot)
    n = arr.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = float(arr[idx].mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (float(arr.mean()), lo, hi)


def format_pct_ci(successes: int, n: int) -> str:
    p = successes / n if n else 0.0
    lo, hi = wilson_ci(successes, n)
    return f"{p*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"


def format_continuous_ci(values, units: str = "") -> str:
    mean, lo, hi = bootstrap_ci(values)
    suffix = (" " + units) if units else ""
    return f"{mean:.3f}{suffix} [{lo:.3f}, {hi:.3f}]"


if __name__ == "__main__":
    # Verify Wilson behaves sensibly.
    print("Wilson 20/20:", wilson_ci(20, 20))
    print("Wilson 0/20:", wilson_ci(0, 20))
    print("Wilson 10/20:", wilson_ci(10, 20))
    print("Bootstrap of N(0,1) N=100:", bootstrap_ci(np.random.default_rng(0).normal(size=100)))
