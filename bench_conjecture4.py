# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Empirical validation of Conjecture 4: gap shrinkage form.

Fit the gap-vs-N data to several candidate functional forms and compare
quality of fit using AIC/BIC. The forms tested:

  M1: gap = a / log(N)            (the writeup's primary conjecture)
  M2: gap = a / log(N) + b        (offset variant)
  M3: gap = a / N^p              (power law)
  M4: gap = a * log(N) / N        (rounding-error scaling, theoretical)
  M5: gap = a + b / sqrt(N)       (sqrt scaling)

For honest reporting: which functional form actually best describes
the empirical data, and is M1 (the conjectured form) significantly
better than alternatives?

Output: fit table + AIC/BIC for each model, plus residuals.
"""

import numpy as np
from scipy.optimize import curve_fit

# Empirical data from bench_assignment.py scaling sweep on sphere
N = np.array([10, 30, 100, 300, 1000, 3000, 10000])
gap = np.array([5.54, 5.05, 3.02, 2.20, 1.70, 1.52, 1.43])
gap_sd = np.array([4.30, 2.04, 0.73, 0.29, 0.04, 0.04, 0.01])
n_seeds = np.array([30, 30, 20, 10, 5, 3, 2])
# weights for least squares: 1/sigma^2 (more seeds → tighter)
weights = 1.0 / np.maximum(gap_sd, 0.05)**2  # floor sigma at 0.05 to avoid divbyzero

# Candidate model functions
def M1(N, a):                      # 1/log
    return a / np.log(N)
def M2(N, a, b):                   # 1/log + offset
    return a / np.log(N) + b
def M3(N, a, p):                   # power law
    return a / np.power(N, p)
def M4(N, a):                      # rounding-error scaling
    return a * np.log(N) / N
def M5(N, a, b):                   # sqrt scaling
    return a + b / np.sqrt(N)


def fit_model(name, fn, p0, N, gap, weights):
    sigma = 1.0 / np.sqrt(weights)
    popt, pcov = curve_fit(fn, N, gap, p0=p0, sigma=sigma, absolute_sigma=False)
    residuals = gap - fn(N, *popt)
    rss = float(np.sum(weights * residuals**2))
    n_obs = len(gap)
    n_params = len(popt)
    nll = 0.5 * rss
    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(n_obs) + 2 * nll
    rmse = float(np.sqrt(np.mean(residuals**2)))
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)
    return {
        'name': name,
        'params': popt,
        'param_se': perr,
        'rss': rss,
        'rmse': rmse,
        'aic': aic,
        'bic': bic,
        'residuals': residuals,
    }


def main():
    print(f"{'model':<28} {'params':<28} {'RSS':>10} {'RMSE':>10} "
          f"{'AIC':>10} {'BIC':>10}")
    results = []
    results.append(fit_model("M1: a/ln(N)",       M1, [13.0],         N, gap, weights))
    results.append(fit_model("M2: a/ln(N) + b",   M2, [13.0, 0.0],    N, gap, weights))
    results.append(fit_model("M3: a/N^p",         M3, [10.0, 0.3],    N, gap, weights))
    results.append(fit_model("M4: a*ln(N)/N",     M4, [10.0],         N, gap, weights))
    results.append(fit_model("M5: a + b/sqrt(N)", M5, [1.0, 15.0],    N, gap, weights))

    for r in results:
        params_str = ", ".join(f"{p:.3f}" for p in r['params'])
        print(f"{r['name']:<28} ({params_str:<26}) {r['rss']:>9.2f}  "
              f"{r['rmse']:>9.4f} {r['aic']:>10.2f} {r['bic']:>10.2f}")

    print()
    print("Residuals (predicted minus observed) per N:")
    print(f"{'N':>8}", end='')
    for r in results:
        print(f"  {r['name'][:8]:>10}", end='')
    print()
    for i, n_val in enumerate(N):
        print(f"{n_val:>8}", end='')
        for r in results:
            print(f"  {r['residuals'][i]:>+10.3f}", end='')
        print()

    # Best model by AIC
    print()
    best_aic = min(results, key=lambda r: r['aic'])
    best_bic = min(results, key=lambda r: r['bic'])
    print(f"Best by AIC: {best_aic['name']}")
    print(f"Best by BIC: {best_bic['name']}")

    # Parametric 95% CI on the best-AIC model's coefficients (Wald: ±1.96·SE).
    print()
    print(f"Best model parameter 95% CI (Wald, parametric from curve_fit covariance):")
    for p, se in zip(best_aic['params'], best_aic['param_se']):
        print(f"  {p:.3f}  [{p - 1.96 * se:.3f}, {p + 1.96 * se:.3f}]  (SE = {se:.3f})")

    # Also: compute predicted vs observed for M1 (the writeup's model)
    print()
    print("M1 predictions (writeup's primary conjecture):")
    a_M1 = results[0]['params'][0]
    print(f"  Fitted constant: a = {a_M1:.3f}")
    print(f"  {'N':>8} {'predicted':>12} {'observed':>12} {'error_pp':>10}")
    for i, n_val in enumerate(N):
        pred = a_M1 / np.log(n_val)
        err = pred - gap[i]
        print(f"  {n_val:>8} {pred:>12.3f} {gap[i]:>12.3f} {err:>+10.3f}")


if __name__ == '__main__':
    main()
