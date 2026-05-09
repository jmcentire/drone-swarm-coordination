# /// script
# dependencies = ["numpy<3"]
# ///
"""Drift × directive × reset bench (Sweep F).

Models per-drone INS bias as a 3D random walk; drones broadcast
est_pos = true_pos + bias and run idealized control to drive
est_pos -> target_pos. Two coherence metrics tell the story:

  internal_coherence  = mean ||est - target||
                        what the swarm thinks of itself
  external_pos_error  = mean ||true - target||
                        what an external observer sees
  external_shape_err  = mean ||(true - true_centroid)
                                - (target - target_centroid)||
                        relative-configuration distortion

Architectural prediction (the v1.2 supplemental claim):

  - drift hurts external, not internal: every drone runs the same
    deterministic rules over the same drifted broadcast; the swarm
    coordinates correctly in its own (drifted) frame regardless of
    drift magnitude
  - reset (reference matching) bounds external_pos_error to roughly
    sigma_drift * sqrt(T_reset)
  - directives are orthogonal: changing the manifold mid-mission
    does not bound drift, it just retargets the swarm
  - inter-drone ranging bounds shape error even without absolute
    reset (the submersible / GPS-denied case)

Sweep F   drift x reset x directive (full grid)
Sweep G   no-reset baseline vs inter-drone ranging
"""

import json
import os
import sys
import time

import numpy as np

NUM_DRONES = int(os.environ.get("NUM_DRONES", "100"))
N_SEEDS = int(os.environ.get("N_SEEDS", "30"))
TICK_DT = 0.04                       # 25 Hz
HORIZON = float(os.environ.get("HORIZON", "600.0"))   # 10-minute mission
NUM_TICKS = int(HORIZON / TICK_DT)

SPHERE_R = 50.0                       # manifold radius (m)

# INS-class drift rates (m / sqrt(s)).
#   ~0.01: fiber-optic INS  (~1 m/hr drift)
#   ~0.1 : tactical IMU     (~10-100 m/hr)
#   ~1.0 : MEMS / dead-reckon (~hundreds of m/hr)
SIGMA_DRIFTS = [0.01, 0.1, 1.0]

# Reference-matching reset interval (s); inf = never.
T_RESETS = [30.0, 300.0, float("inf")]

# Operator directive interval (s); inf = static mission.
T_DIRECTIVES = [30.0, 300.0, float("inf")]

# Reset residual noise (m) — bathymetric / terrain-match accuracy.
SIGMA_RESET = 0.1

# Inter-drone ranging residual (m) — UWB / acoustic ToF accuracy.
SIGMA_RANGE = 0.05

SAMPLE_EVERY_S = 30.0


# ---------------------------------------------------------------- manifolds

def fibonacci_sphere(n: int, r: float) -> np.ndarray:
    pts = np.zeros((n, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(max(n - 1, 1))) * 2
        radius = np.sqrt(max(1 - y * y, 0.0))
        theta = phi * i
        pts[i] = [np.cos(theta) * radius * r, y * r, np.sin(theta) * radius * r]
    return pts


def ring_z0(n: int, r: float) -> np.ndarray:
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(thetas), r * np.sin(thetas), np.zeros(n)], axis=1)


def ellipsoid(n: int, r: float) -> np.ndarray:
    base = fibonacci_sphere(n, 1.0)
    return base * np.array([1.5 * r, 0.7 * r, r])


def montecarlo_density(n: int, r: float, rng) -> np.ndarray:
    """Sample n points from a non-uniform 'feature density' on the sphere.

    Generalisation note: the four layers don't care whether the manifold
    came from geometry or from sampling. This generates a tilted gaussian
    blob on the sphere -- an example of statistical-feature sampling
    (mineral concentration, species presence, etc).
    """
    pts = rng.normal(size=(n, 3))
    # Tilt the density toward +z by adding a bias before normalising.
    pts[:, 2] += 1.5
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts * r


MANIFOLDS = [
    ("sphere", lambda n, r, rng: fibonacci_sphere(n, r)),
    ("ellipsoid", lambda n, r, rng: ellipsoid(n, r)),
    ("ring", lambda n, r, rng: ring_z0(n, r)),
    ("mc_density", lambda n, r, rng: montecarlo_density(n, r, rng)),
]


def generate_manifold(idx: int, n: int, r: float, rng) -> np.ndarray:
    name, fn = MANIFOLDS[idx % len(MANIFOLDS)]
    return fn(n, r, rng)


# --------------------------------------------------------------- assignment

def greedy_assign(est_pos: np.ndarray, manifold: np.ndarray) -> np.ndarray:
    """Greedy nearest-unused assignment.

    Proxy for the paper's hierarchical PCA-tree assignment: order doesn't
    matter for the metrics here (mean external error, mean shape error).
    """
    n = len(est_pos)
    targets = np.zeros((n, 3))
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        d = np.linalg.norm(manifold - est_pos[i], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        targets[i] = manifold[j]
        used[j] = True
    return targets


# ----------------------------------------------------------------- simulate

def simulate(sigma_drift: float,
             T_reset: float,
             T_directive: float,
             seed: int,
             ranging: bool = False) -> dict:
    rng = np.random.default_rng(seed)

    manifold = fibonacci_sphere(NUM_DRONES, SPHERE_R)
    targets = manifold.copy()
    true_pos = manifold.copy()
    bias = np.zeros((NUM_DRONES, 3))

    last_reset_t = 0.0
    last_directive_t = 0.0
    manifold_idx = 0

    samples = []
    next_sample_t = 0.0

    drift_step = sigma_drift * np.sqrt(TICK_DT)

    for tick in range(NUM_TICKS + 1):
        t = tick * TICK_DT

        # ---- bias evolution (random-walk INS error)
        if drift_step > 0.0:
            bias += rng.normal(0.0, drift_step, size=(NUM_DRONES, 3))

        # ---- idealised tight control: every tick, true snaps so est = target.
        # est = true + bias  =>  true = target - bias  forces est == target.
        true_pos = targets - bias
        est_pos = targets.copy()           # by construction est == target

        # ---- inter-drone ranging (Sweep G): collapse bias variance to a
        # common centroid bias plus per-drone residual sigma_range.
        if ranging:
            mean_bias = bias.mean(axis=0)
            bias = mean_bias + rng.normal(0.0, SIGMA_RANGE, size=(NUM_DRONES, 3))
            true_pos = targets - bias

        # ---- reference-matching reset
        if T_reset != float("inf") and t - last_reset_t >= T_reset:
            bias = rng.normal(0.0, SIGMA_RESET, size=(NUM_DRONES, 3))
            true_pos = targets - bias
            last_reset_t = t

        # ---- directive change: new manifold, re-assign from current broadcast
        if T_directive != float("inf") and t - last_directive_t >= T_directive:
            manifold_idx += 1
            new_manifold = generate_manifold(manifold_idx, NUM_DRONES, SPHERE_R, rng)
            broadcast = true_pos + bias       # what the swarm sees
            targets = greedy_assign(broadcast, new_manifold)
            # Snap to settled state at the new manifold (idealised control)
            true_pos = targets - bias
            est_pos = targets.copy()
            last_directive_t = t

        # ---- sample metrics
        if t >= next_sample_t - 1e-9:
            ext_pos = float(np.linalg.norm(true_pos - targets, axis=1).mean())
            true_c = true_pos - true_pos.mean(axis=0)
            tgt_c = targets - targets.mean(axis=0)
            ext_shape = float(np.linalg.norm(true_c - tgt_c, axis=1).mean())
            int_coh = float(np.linalg.norm(est_pos - targets, axis=1).mean())
            samples.append({
                "t": t,
                "ext_pos": ext_pos,
                "ext_shape": ext_shape,
                "int_coh": int_coh,
            })
            next_sample_t += SAMPLE_EVERY_S

    return {
        "samples": samples,
        "ext_pos_final": samples[-1]["ext_pos"],
        "ext_shape_final": samples[-1]["ext_shape"],
        "int_coh_final": samples[-1]["int_coh"],
    }


# ------------------------------------------------------------------ sweeps

def fmt_T(t: float) -> str:
    return "inf" if t == float("inf") else f"{t:>4.0f}s"


def bootstrap_ci(arr, n=1000, seed=0):
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(a), size=(n, len(a)))
    means = a[idxs].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run_sweep_f():
    print(f"\n# Sweep F: drift x reset x directive")
    print(f"# N={NUM_DRONES}, horizon={HORIZON}s, seeds={N_SEEDS}")
    print(f"# {'sd':>5}  {'T_r':>6}  {'T_d':>6}  "
          f"{'ext_pos_m':>22}  {'shape_m':>22}  {'int_coh_m':>10}")
    rows = []
    for sd in SIGMA_DRIFTS:
        for tr in T_RESETS:
            for td in T_DIRECTIVES:
                pos, shape, intc = [], [], []
                for seed in range(N_SEEDS):
                    r = simulate(sd, tr, td, seed)
                    pos.append(r["ext_pos_final"])
                    shape.append(r["ext_shape_final"])
                    intc.append(r["int_coh_final"])
                pos_a = np.array(pos)
                shape_a = np.array(shape)
                intc_a = np.array(intc)
                pos_lo, pos_hi = bootstrap_ci(pos_a, seed=seed)
                shape_lo, shape_hi = bootstrap_ci(shape_a, seed=seed + 1)
                rows.append({
                    "sigma_drift": sd, "T_reset": tr, "T_directive": td,
                    "ext_pos_mean": float(pos_a.mean()),
                    "ext_pos_ci": [pos_lo, pos_hi],
                    "ext_shape_mean": float(shape_a.mean()),
                    "ext_shape_ci": [shape_lo, shape_hi],
                    "int_coh_mean": float(intc_a.mean()),
                })
                print(f"  {sd:>4.2f}  {fmt_T(tr):>6}  {fmt_T(td):>6}  "
                      f"{pos_a.mean():>8.3f} [{pos_lo:>5.2f},{pos_hi:>6.2f}]  "
                      f"{shape_a.mean():>8.3f} [{shape_lo:>5.2f},{shape_hi:>6.2f}]  "
                      f"{intc_a.mean():>10.2e}")
    return rows


def run_sweep_g():
    print(f"\n# Sweep G: ranging vs no ranging (T_reset=inf, T_directive=inf)")
    print(f"# N={NUM_DRONES}, horizon={HORIZON}s, seeds={N_SEEDS}")
    print(f"# {'sd':>5}  {'mode':>10}  {'ext_pos_m':>22}  {'shape_m':>22}")
    rows = []
    for sd in SIGMA_DRIFTS:
        for ranging in (False, True):
            pos, shape = [], []
            for seed in range(N_SEEDS):
                r = simulate(sd, float("inf"), float("inf"), seed,
                             ranging=ranging)
                pos.append(r["ext_pos_final"])
                shape.append(r["ext_shape_final"])
            pos_a = np.array(pos)
            shape_a = np.array(shape)
            pos_lo, pos_hi = bootstrap_ci(pos_a, seed=seed)
            shape_lo, shape_hi = bootstrap_ci(shape_a, seed=seed + 1)
            mode = "ranging" if ranging else "no-ranging"
            rows.append({
                "sigma_drift": sd, "ranging": bool(ranging),
                "ext_pos_mean": float(pos_a.mean()),
                "ext_pos_ci": [pos_lo, pos_hi],
                "ext_shape_mean": float(shape_a.mean()),
                "ext_shape_ci": [shape_lo, shape_hi],
            })
            print(f"  {sd:>4.2f}  {mode:>10}  "
                  f"{pos_a.mean():>8.3f} [{pos_lo:>5.2f},{pos_hi:>6.2f}]  "
                  f"{shape_a.mean():>8.3f} [{shape_lo:>5.2f},{shape_hi:>6.2f}]")
    return rows


def main():
    t0 = time.time()
    f_rows = run_sweep_f()
    g_rows = run_sweep_g()
    print(f"\nWall time: {time.time() - t0:.1f}s")
    out = {
        "config": {
            "num_drones": NUM_DRONES, "n_seeds": N_SEEDS,
            "horizon_s": HORIZON, "tick_dt": TICK_DT,
            "sigma_drifts": SIGMA_DRIFTS,
            "T_resets": [str(t) for t in T_RESETS],
            "T_directives": [str(t) for t in T_DIRECTIVES],
            "sigma_reset": SIGMA_RESET, "sigma_range": SIGMA_RANGE,
            "sphere_r": SPHERE_R,
        },
        "sweep_f": f_rows,
        "sweep_g": g_rows,
    }
    out_path = os.environ.get("OUT_PATH", "bench_drift_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
