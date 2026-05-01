# /// script
# dependencies = ["numpy<3", "scipy"]
# ###
"""Layer 4: full implementation of fiducial selection and cooperative
localization on the broadcast-as-shared-state substrate.

Setting: each drone has noisy GPS fixes at random Poisson times, and
between fixes runs INS dead-reckoning. Drones broadcast (est_pos,
confidence). Confidence decays since last fix. The substrate's job:

  1. **Fiducial selection** (deterministic, decentralized): at each
     refinement tick, every drone independently selects the n_fid
     drones to use as fiducials, picked by the highest-confidence
     drone in each of n_fid spatially-distinct PCA-tree subtrees.
     Same algorithm, same broadcast input → same selection.

  2. **Cooperative localization** (Layer 4): non-fiducial drones
     observe their relative position to each fiducial (with bearing/
     range noise σ_obs), combine with fiducial broadcast est_pos,
     and refine their own est_pos via least-squares.

  3. **Same substrate primitive**: PCA tree is the same one used for
     ASSIGN; selection is at depth ⌈log₂(n_fid)⌉ of the tree.

We compare three regimes under sparse GPS (~10% of drones have a
fresh fix at any moment):

  - INS only: each drone runs INS, no inter-drone help. Drift
    accumulates uniformly.
  - INS + GPS: each drone uses its own GPS when available.
  - Layer 4: drones with stale fixes refine using fresh-fix
    fiducials' broadcasts.

Metric: formation_err in true coordinates (mean ‖true_pos −
target_leaf‖) over time, across the three regimes.
"""

import os
import numpy as np

NUM_DRONES = 100
DRONE_TICK = 0.04
APPROACH_RADIUS = 4.0
REPULSION_RADIUS = 3.5
MAX_TICKS = int(os.environ.get("MAX_TICKS", "2000"))
WORLD = 40.0

# Noise / GPS params
ACCEL_RW = float(os.environ.get("ACCEL_RW", "0.04"))
SHARED_FRACTION = float(os.environ.get("SHARED", "0.5"))
GPS_NOISE = float(os.environ.get("GPS_NOISE", "0.1"))
GPS_RATE_PER_S = float(os.environ.get("GPS_RATE", "0.1"))  # per-drone fix rate
OBS_NOISE = float(os.environ.get("OBS_NOISE", "0.05"))    # σ on relative-position measurements

# Heavy-tailed noise model: mixture of standard Gaussian + multipath spikes.
# With probability HEAVY_TAIL_PROB, the noise magnitude is HEAVY_TAIL_SCALE×
# the nominal σ. Models GPS multipath in obstructed sky views, UWB NLOS
# (non-line-of-sight) errors, and visual-fiducial detection failures. Set
# HEAVY_TAIL_PROB=0 for pure Gaussian (idealized).
HEAVY_TAIL_PROB = float(os.environ.get("HEAVY_TAIL_PROB", "0.1"))
HEAVY_TAIL_SCALE = float(os.environ.get("HEAVY_TAIL_SCALE", "5.0"))


def heavy_tailed_normal(rng, shape, sigma):
    """Sample from a mixture: (1-p) × N(0, σ) + p × N(0, k·σ).
    Models real-sensor heavy tails from multipath, NLOS, and outliers."""
    base = rng.standard_normal(shape) * sigma
    is_outlier = rng.random(shape if isinstance(shape, int) else shape[:1]) < HEAVY_TAIL_PROB
    if isinstance(shape, tuple) and len(shape) > 1:
        is_outlier = is_outlier.reshape(-1, *([1] * (len(shape) - 1)))
        is_outlier = np.broadcast_to(is_outlier, shape)
    spike = rng.standard_normal(shape) * sigma * HEAVY_TAIL_SCALE
    return np.where(is_outlier, spike, base)

# Layer 4 params
N_FIDUCIALS = int(os.environ.get("N_FIDUCIALS", "8"))
REFINE_EVERY_TICKS = int(os.environ.get("REFINE_EVERY", "10"))
CONFIDENCE_DECAY_S = float(os.environ.get("CONF_DECAY", "5.0"))

N_SEEDS = int(os.environ.get("N_SEEDS", "30"))

# Modes: "ins" (INS+GPS only), "fiducial" (INS+GPS + fiducial refinement)
MODE = os.environ.get("MODE", "compare")  # "compare" runs all three

ACCEL_RW_PER_SQRT_S = ACCEL_RW / 60.0
VEL_SIGMA_PER_TICK = ACCEL_RW_PER_SQRT_S * np.sqrt(DRONE_TICK)


def make_sphere(n, radius=15, center=(0, 0, 20)):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    cx, cy, cz = center
    return np.column_stack((
        radius * np.cos(theta) * np.sin(phi) + cx,
        radius * np.sin(theta) * np.sin(phi) + cy,
        radius * np.cos(phi) + cz,
    ))


class ManifoldNode:
    def __init__(self, positions, depth=0):
        self.positions = np.array(positions)
        self.center = np.mean(positions, axis=0)
        self.depth = depth
        self.split_axis = None
        self.left = self.right = None
        if len(positions) > 1:
            self._split()

    def _split(self):
        c = self.positions - self.center
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        self.split_axis = Vt[0]
        proj = c @ self.split_axis
        order = np.argsort(proj, kind='stable')
        mid = len(order) // 2
        self.left = ManifoldNode(self.positions[order[:mid]], self.depth + 1)
        self.right = ManifoldNode(self.positions[order[mid:]], self.depth + 1)


def compute_target_position(my_id, drones, root):
    """Strict-mode hierarchical assignment, returns leaf position."""
    node = root
    parent = root
    cur = list(drones)
    my_pos = next(np.array(d['pos']) for d in drones if d['id'] == my_id)
    while node.left is not None and len(cur) > 1:
        n = len(cur)
        nl = len(node.left.positions)
        nt = len(node.positions)
        dl = max(0, min(n, int(round(n * nl / nt))))
        if dl == 0:
            parent = node; node = node.right; continue
        if dl == n:
            parent = node; node = node.left; continue
        positions = np.array([d['pos'] for d in cur])
        proj = positions @ node.split_axis
        order = np.argsort(proj, kind='stable')
        groups = [
            [cur[order[i]] for i in range(dl)],
            [cur[order[i]] for i in range(dl, n)],
        ]
        subs = [node.left, node.right]
        for i, group in enumerate(groups):
            if any(d['id'] == my_id for d in group):
                parent = node; node = subs[i]; cur = group; break
    while node.left is not None:
        parent = node
        dl_ = np.linalg.norm(my_pos - node.left.center)
        dr_ = np.linalg.norm(my_pos - node.right.center)
        node = node.left if dl_ <= dr_ else node.right
    leaf_pos = node.positions[0] if len(node.positions) == 1 else node.center
    return leaf_pos.copy()


# ==== Layer 4: Fiducial selection ====

def collect_subtrees_at_depth(tree, target_depth):
    """BFS-walk the tree, collect 2^depth subtrees rooted at level=target_depth.
    If the tree is shallower than target_depth, return the leaf-level nodes."""
    nodes = [tree]
    for _ in range(target_depth):
        new_nodes = []
        for n in nodes:
            if n.left is not None:
                new_nodes.append(n.left)
                new_nodes.append(n.right)
            else:
                new_nodes.append(n)  # leaf
        nodes = new_nodes
    return nodes


def select_fiducials(positions, confidences, dead, tree, n_fid):
    """Decentralized fiducial selection: pick the highest-confidence live
    drone whose current position falls within each of n_fid spatially-
    diverse subtrees of the PCA tree.

    The selection is a deterministic function of (positions, confidences,
    dead, tree) — every drone running this against the same broadcast
    arrives at the same n_fid drone IDs. No coordination needed.
    """
    target_depth = int(np.ceil(np.log2(max(n_fid, 1))))
    subtrees = collect_subtrees_at_depth(tree, target_depth)[:n_fid]

    fiducials = []
    used = set()
    for st in subtrees:
        # Find live drones whose current position is closest to this subtree's centroid
        dists = []
        for i in range(len(positions)):
            if dead[i] or i in used:
                continue
            d = float(np.linalg.norm(positions[i] - st.center))
            dists.append((d, i))
        if not dists:
            continue
        dists.sort()
        # Among the closest 5 to this subtree, pick highest confidence
        top = dists[:5]
        best = max(top, key=lambda x: confidences[x[1]])
        fiducials.append(best[1])
        used.add(best[1])
    return fiducials


def refine_via_fiducials(true_pos, est_pos, fiducial_ids, true_pos_arr, est_pos_arr, rng):
    """For each non-fiducial drone, compute a fiducial-derived position
    estimate and return it.

    Drone d observes its TRUE relative position to each fiducial (with
    noise σ_obs). Combined with each fiducial's broadcast est_pos:

        d_estimate_via_f = fiducial_est_pos[f] + (true_pos[d] - true_pos[f] + noise)

    If fiducials' est_pos are accurate, this estimate is accurate to O(noise).
    Average across fiducials reduces noise by 1/√n_fid.
    """
    n = len(true_pos_arr)
    new_est = est_pos_arr.copy()
    if not fiducial_ids:
        return new_est
    for d in range(n):
        if d in fiducial_ids:
            continue
        estimates = []
        weights = []
        for f in fiducial_ids:
            obs_relative = (true_pos_arr[d] - true_pos_arr[f]) + heavy_tailed_normal(rng, 3, OBS_NOISE)
            est_via_f = est_pos_arr[f] + obs_relative
            estimates.append(est_via_f)
            weights.append(1.0)
        weights = np.array(weights) / sum(weights)
        # Weighted average. RANSAC-style residual rejection would further
        # bound the influence of outlier observations; we report the
        # simple weighted-average baseline here.
        new_est[d] = np.sum(np.array(estimates) * weights[:, None], axis=0)
    return new_est


# ==== Simulation ====

def simulate(starts, targets, mode="ins", seed=0):
    """mode ∈ {"ins" (no GPS, INS only), "ins_gps" (INS + per-drone Poisson GPS),
    "fiducial" (INS + GPS + Layer 4 fiducial refinement)}."""
    rng = np.random.default_rng(seed)
    n = len(starts)
    tree = ManifoldNode(targets)

    true_pos = starts.copy().astype(float)
    est_pos = true_pos.copy()
    vel_err = np.zeros((n, 3))
    shared_vel_err = np.zeros(3)
    last_fix_tick = np.zeros(n, dtype=int)  # tick of last GPS fix

    actual_vel = np.zeros((n, 3))
    locked = np.zeros(n, dtype=bool)
    dead = np.zeros(n, dtype=bool)
    stall = np.zeros(n, dtype=int)
    last_dist = np.full(n, np.inf)

    drones = [{'id': i, 'pos': est_pos[i].copy()} for i in range(n)]
    target_pos = np.array([
        compute_target_position(i, drones, tree) for i in range(n)
    ])

    measurements = []

    p_gps_per_tick = GPS_RATE_PER_S * DRONE_TICK if mode != "ins" else 0.0
    decay_ticks = CONFIDENCE_DECAY_S / DRONE_TICK

    for tick in range(MAX_TICKS):
        sigma_i = VEL_SIGMA_PER_TICK * np.sqrt(1 - SHARED_FRACTION)
        sigma_s = VEL_SIGMA_PER_TICK * np.sqrt(SHARED_FRACTION)
        vel_err += rng.standard_normal((n, 3)) * sigma_i
        shared_vel_err += rng.standard_normal(3) * sigma_s
        est_pos += (vel_err + shared_vel_err) * DRONE_TICK

        # Per-drone Poisson GPS fixes
        if p_gps_per_tick > 0:
            fix_mask = rng.random(n) < p_gps_per_tick
            for i in np.where(fix_mask)[0]:
                if dead[i]:
                    continue
                est_pos[i] = true_pos[i] + heavy_tailed_normal(rng, 3, GPS_NOISE)
                vel_err[i] = 0
                last_fix_tick[i] = tick

        # Layer 4: fiducial refinement
        if mode == "fiducial" and tick > 0 and tick % REFINE_EVERY_TICKS == 0:
            confidences = np.exp(-(tick - last_fix_tick) / decay_ticks)
            confidences[dead] = 0
            fiducial_ids = select_fiducials(est_pos, confidences, dead, tree, N_FIDUCIALS)
            est_pos = refine_via_fiducials(
                true_pos, est_pos, fiducial_ids, true_pos, est_pos, rng)
            # Refined drones gain confidence equivalent to a GPS-fix
            # (They have observed relative positions to high-confidence fiducials.)
            for i in range(n):
                if i in fiducial_ids or dead[i]:
                    continue
                last_fix_tick[i] = tick

        # Steering
        new_vel = np.zeros((n, 3))
        for i in range(n):
            if dead[i]:
                continue
            tgt = target_pos[i]
            diff = tgt - est_pos[i]
            dist = float(np.linalg.norm(diff))
            is_final = dist < APPROACH_RADIUS
            attr = (diff / dist) * min(0.6, dist * 0.1) if dist > 0 else np.zeros(3)
            er = REPULSION_RADIUS * 0.4 if is_final else REPULSION_RADIUS
            rep = np.zeros(3)
            for j in range(n):
                if j == i or dead[j]:
                    continue
                d = est_pos[i] - est_pos[j]
                r = float(np.linalg.norm(d))
                if 0 < r < er:
                    unit = d / r
                    closing = max(0.0, -np.dot(actual_vel[i] - actual_vel[j], unit))
                    force = ((er - r) / r) * (1.0 + closing * 2.0)
                    rep += unit * force * 0.15
            v = attr + rep
            if is_final:
                progress = last_dist[i] - dist
                if abs(progress) < 0.01 and dist > 0.3:
                    stall[i] += 1
                else:
                    stall[i] = max(0, stall[i] - 5)
                last_dist[i] = dist
                if stall[i] > 10:
                    fade = max(0.0, 1.0 - (stall[i] - 10) / 40.0)
                    rep *= fade
                    v = attr + rep
            s_norm = float(np.linalg.norm(v))
            max_speed = 0.8
            if is_final and dist < 3.0:
                max_speed = max(0.05, dist * 0.2)
            if s_norm > max_speed:
                v = (v / s_norm) * max_speed
            new_vel[i] = v

        actual_vel = new_vel
        true_pos += actual_vel
        est_pos += actual_vel

        # Sample
        if tick % 25 == 0:
            t = tick * DRONE_TICK
            live = ~dead
            abs_drift = float(np.mean(np.linalg.norm((est_pos - true_pos)[live], axis=1)))
            valid = live & ~np.isnan(target_pos[:, 0])
            form_err = float(np.mean(np.linalg.norm((true_pos - target_pos)[valid], axis=1))) if valid.any() else 0.0
            measurements.append((t, abs_drift, form_err))

    return measurements


def bootstrap_ci(samples, ci=0.95, n_resamples=2000, rng=None):
    """Bootstrap 95% CI on the mean of `samples`. Returns (lo, hi)."""
    samples = np.asarray(samples)
    if len(samples) == 0:
        return (float('nan'), float('nan'))
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(samples)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        means[i] = samples[idx].mean()
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return (lo, hi)


def run_mode(label, mode):
    targets = make_sphere(NUM_DRONES)
    runs = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        starts = rng.uniform(-WORLD, WORLD, (NUM_DRONES, 3))
        m = simulate(starts, targets, mode=mode, seed=seed)
        runs.append(m)
    n_steps = min(len(r) for r in runs)
    times = np.array([runs[0][i][0] for i in range(n_steps)])
    abs_d = np.array([[runs[s][i][1] for s in range(N_SEEDS)] for i in range(n_steps)])
    form = np.array([[runs[s][i][2] for s in range(N_SEEDS)] for i in range(n_steps)])
    # Bootstrap 95% CIs at each timestep
    rng = np.random.default_rng(0)
    form_ci_lo = np.array([bootstrap_ci(form[i], rng=rng)[0] for i in range(n_steps)])
    form_ci_hi = np.array([bootstrap_ci(form[i], rng=rng)[1] for i in range(n_steps)])
    return {
        'label': label,
        'times': times,
        'abs_drift_mean': abs_d.mean(axis=1),
        'abs_drift_std': abs_d.std(axis=1),
        'form_err_mean': form.mean(axis=1),
        'form_err_std': form.std(axis=1),
        'form_err_ci_lo': form_ci_lo,
        'form_err_ci_hi': form_ci_hi,
    }


def main():
    print(f"Layer 4 benchmark: N={NUM_DRONES} drones on sphere, "
          f"sparse Poisson GPS at {GPS_RATE_PER_S}/s/drone")
    print(f"Fiducial selection: {N_FIDUCIALS} fiducials at depth "
          f"{int(np.ceil(np.log2(N_FIDUCIALS)))} of the PCA tree, "
          f"refresh every {REFINE_EVERY_TICKS} ticks ({REFINE_EVERY_TICKS * DRONE_TICK:.1f}s)")
    print(f"Relative-measurement noise σ = {OBS_NOISE} m, GPS noise σ = {GPS_NOISE} m, "
          f"{N_SEEDS} seeds\n")

    results = []
    print("Running INS only (no GPS) ...")
    results.append(run_mode("INS only", "ins"))
    print("Running INS + sparse GPS ...")
    results.append(run_mode("INS + sparse GPS", "ins_gps"))
    print("Running INS + sparse GPS + Layer 4 fiducial refinement ...")
    results.append(run_mode("INS + GPS + Layer 4", "fiducial"))

    print()
    print(f"{'mode':<28} {'t=10s':>17} {'t=30s':>17} {'t=60s':>17} "
          f"{'t=final':>17}  (mean [95% CI])")
    for r in results:
        def at(t):
            i = int(np.argmin(np.abs(r['times'] - t)))
            return (r['form_err_mean'][i],
                    r['form_err_ci_lo'][i],
                    r['form_err_ci_hi'][i])
        m10, l10, h10 = at(10)
        m30, l30, h30 = at(30)
        m60, l60, h60 = at(60)
        mf, lf, hf = (r['form_err_mean'][-1], r['form_err_ci_lo'][-1],
                      r['form_err_ci_hi'][-1])
        print(f"{r['label']:<28} "
              f"{m10:>4.3f}[{l10:.3f},{h10:.3f}] "
              f"{m30:>4.3f}[{l30:.3f},{h30:.3f}] "
              f"{m60:>4.3f}[{l60:.3f},{h60:.3f}] "
              f"{mf:>4.3f}[{lf:.3f},{hf:.3f}]")

    print()
    print("Trajectory (mean ± std across seeds):")
    print(f"{'t_s':>6}", end="")
    for r in results:
        print(f"  {r['label'][:18]:>20}", end="")
    print()
    n_show = min(len(results[0]['times']), 12)
    step = max(1, len(results[0]['times']) // n_show)
    for i in range(0, len(results[0]['times']), step):
        print(f"{results[0]['times'][i]:>6.1f}", end="")
        for r in results:
            print(f"  {r['form_err_mean'][i]:>10.3f}±{r['form_err_std'][i]:<5.2f}", end="")
        print()


if __name__ == '__main__':
    main()
