# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Localization-aware swarm simulation: drones use INS-derived position
estimates broadcast to each other, instead of perfect ground-truth.

State per drone:
  true_pos  — omniscient narrator's ground-truth position
  est_pos   — drone's belief, derived from (last GPS fix) + (integrated IMU)
  vel_err   — drone's velocity-error accumulator (independent random walk)

Shared (correlated environmental bias):
  shared_vel_err — translates the whole swarm uniformly (gravity model,
                   temperature, manufacturing offsets)

The broadcast carries est_pos. The hierarchical assignment and patch
protocol both use est_pos — under noise, drones may compute slightly
different assignments than they would with perfect information, but
the architecture is robust to this (the bisection is stable to small
perturbations of input positions).

Metrics measured in TRUE coordinates (the audience-visible formation):
  formation_err — mean ‖true_pos[i] - assigned_target[i]‖ across live drones
  abs_drift    — mean ‖est_pos[i] - true_pos[i]‖
  rel_drift    — pairwise true distance distortion

GPS regimes:
  on:      GPS correction every GPS_INTERVAL_S seconds
  off:     pure dead-reckoning from launch
  outage:  on for the first OUTAGE_AFTER_S seconds, off after

Optional kill event injects a primary death at KILL_TICK to test recovery
under uncertainty (patch protocol picks closest surplus by est_pos).
"""

import os
import numpy as np

NUM_DRONES = 100
DRONE_TICK = 0.04
APPROACH_RADIUS = 4.0
REPULSION_RADIUS = 3.5
MAX_TICKS = int(os.environ.get("MAX_TICKS", "750"))
WORLD = 40.0

# IMU and GPS parameters
ACCEL_RW = float(os.environ.get("ACCEL_RW", "0.04"))  # m/s/√hr
SHARED_FRACTION = float(os.environ.get("SHARED", "0.5"))
GPS_MODE = os.environ.get("GPS", "on")
GPS_INTERVAL_S = float(os.environ.get("GPS_INTERVAL", "1.0"))
OUTAGE_AFTER_S = float(os.environ.get("OUTAGE_AFTER", "60"))
GPS_NOISE = float(os.environ.get("GPS_NOISE", "0.5"))

# Optional death event
KILL_TICK = int(os.environ.get("KILL_TICK", "0"))
KILL_ID = int(os.environ.get("KILL_ID", "42"))
SURPLUS = int(os.environ.get("SURPLUS", "0"))
TOTAL_DRONES = NUM_DRONES + SURPLUS

# Number of seeds for averaging
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))

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


def compute_target(my_id, drones, root):
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
    if len(cur) == 1:
        return leaf_pos.copy()
    distances = sorted(
        (float(np.linalg.norm(np.array(d['pos']) - leaf_pos)), d['id'])
        for d in cur
    )
    primary_id = distances[0][1]
    if my_id == primary_id:
        return leaf_pos.copy()
    return parent.center.copy()


def assign_all(broadcast_pos, dead, tree):
    n = len(broadcast_pos)
    survivors = [{'id': i, 'pos': broadcast_pos[i].copy()}
                 for i in range(n) if not dead[i]]
    targets_pos = np.full((n, 3), np.nan)
    for i in range(n):
        if dead[i]:
            continue
        targets_pos[i] = compute_target(i, survivors, tree)
    return targets_pos


def classify_primary(target_pos, manifold_targets, dead):
    n = len(target_pos)
    is_primary = np.zeros(n, dtype=bool)
    for i in range(n):
        if dead[i] or np.any(np.isnan(target_pos[i])):
            continue
        dmin = float(np.linalg.norm(manifold_targets - target_pos[i], axis=1).min())
        is_primary[i] = dmin < 1e-6
    return is_primary


def patch_after_death(target_pos, is_primary, broadcast_pos, dead, dead_id):
    """Patch using broadcast positions (which are noisy est_pos)."""
    new_targets = target_pos.copy()
    new_primary = is_primary.copy()
    dead_target = target_pos[dead_id].copy()
    new_targets[dead_id] = np.nan
    new_primary[dead_id] = False
    surplus_ids = [i for i in range(len(broadcast_pos))
                   if not dead[i] and not is_primary[i]]
    if not surplus_ids:
        return new_targets, new_primary, None
    distances = sorted(
        (float(np.linalg.norm(broadcast_pos[i] - dead_target)), i)
        for i in surplus_ids
    )
    promoted = distances[0][1]
    new_targets[promoted] = dead_target
    new_primary[promoted] = True
    return new_targets, new_primary, promoted


def simulate(starts, targets, seed=0):
    rng = np.random.default_rng(seed)
    n = len(starts)
    tree = ManifoldNode(targets)

    true_pos = starts.copy().astype(float)
    est_pos = true_pos.copy()  # initial GPS fix is perfect
    vel_err = np.zeros((n, 3))
    shared_vel_err = np.zeros(3)
    actual_vel = np.zeros((n, 3))

    locked = np.zeros(n, dtype=bool)
    dead = np.zeros(n, dtype=bool)
    stall = np.zeros(n, dtype=int)
    last_dist = np.full(n, np.inf)

    # Initial assignment uses est_pos (perfect at t=0)
    target_pos = assign_all(est_pos, dead, tree)
    is_primary = classify_primary(target_pos, targets, dead)
    initial_assignment = target_pos.copy()

    # Compute the perfect-info assignment for comparison
    perfect_target_pos = assign_all(true_pos, dead, tree)

    measurements = []  # (t, abs_drift, rel_drift, formation_err, locked_count)
    gps_period = max(1, int(GPS_INTERVAL_S / DRONE_TICK))
    outage_tick = int(OUTAGE_AFTER_S / DRONE_TICK)

    for tick in range(MAX_TICKS):
        # Optional kill
        if KILL_TICK > 0 and tick == KILL_TICK and not dead[KILL_ID]:
            dead[KILL_ID] = True
            if SURPLUS > 0:
                new_targets, new_primary, promoted = patch_after_death(
                    target_pos, is_primary, est_pos, dead, KILL_ID)
                target_pos = new_targets
                is_primary = new_primary
                if promoted is not None:
                    locked[promoted] = False
                    stall[promoted] = 0
                    last_dist[promoted] = np.inf

        # IMU integration noise: add per-tick velocity perturbation
        sigma_i = VEL_SIGMA_PER_TICK * np.sqrt(1 - SHARED_FRACTION)
        sigma_s = VEL_SIGMA_PER_TICK * np.sqrt(SHARED_FRACTION)
        vel_err += rng.standard_normal((n, 3)) * sigma_i
        shared_vel_err += rng.standard_normal(3) * sigma_s

        # Drift contribution to est_pos this tick (over all drones)
        est_pos += (vel_err + shared_vel_err) * DRONE_TICK

        # GPS correction (resets est_pos to truth + small noise)
        gps_active = (
            GPS_MODE == "on" or
            (GPS_MODE == "outage" and tick < outage_tick)
        )
        if gps_active and tick > 0 and tick % gps_period == 0:
            est_pos = true_pos + rng.standard_normal((n, 3)) * GPS_NOISE
            vel_err = np.zeros((n, 3))
            shared_vel_err = np.zeros(3)

        # Compute steering for each drone using est_pos. We do NOT latch a
        # "locked" state that freezes motion: a hovering drone whose IMU
        # drifts will keep issuing small correction commands based on its
        # belief, and those commands actually move its true position. This
        # is the dynamic that produces drift-induced formation error in
        # real systems even when drones are nominally stationary.
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
                if j == i or dead[j] or locked[j]:
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
        # True motion: perfect actuator follows commanded velocity
        true_pos += actual_vel
        # Drone's belief: command also updates est_pos (drift was already added above)
        est_pos += actual_vel

        # No lock latching — drones continuously track. A drone that has
        # arrived will hover with small corrections; if its IMU drifts,
        # the corrections will gradually push true_pos away from target
        # while keeping est_pos near target (drone thinks it's still there).

        # Sample once per second
        if tick % 25 == 0:
            t = tick * DRONE_TICK
            live_mask = ~dead
            abs_drift = float(np.mean(np.linalg.norm(
                (est_pos - true_pos)[live_mask], axis=1)))
            # Pairwise relative drift on a fixed sample of drones
            sample = np.arange(0, n, 2)
            sample = sample[~dead[sample]]
            if len(sample) >= 2:
                a = sample[: len(sample) // 2]
                b = sample[len(sample) // 2: 2 * (len(sample) // 2)]
                true_pairs = np.linalg.norm(true_pos[a] - true_pos[b], axis=1)
                est_pairs = np.linalg.norm(est_pos[a] - est_pos[b], axis=1)
                rel_drift = float(np.mean(np.abs(true_pairs - est_pairs)))
            else:
                rel_drift = 0.0

            # Formation error: how far is each drone's TRUE pos from its assigned target?
            valid = live_mask & ~np.isnan(target_pos[:, 0])
            if valid.any():
                form_err = float(np.mean(np.linalg.norm(
                    (true_pos - target_pos)[valid], axis=1)))
            else:
                form_err = 0.0
            measurements.append((t, abs_drift, rel_drift, form_err, int(locked[live_mask].sum())))

    return measurements, target_pos, perfect_target_pos, dead


def run_regime(label, mode, gps_interval=None, outage_after=None,
               accel_rw=None, shared_fraction=None, surplus=0, kill_tick=0):
    global GPS_MODE, GPS_INTERVAL_S, OUTAGE_AFTER_S, ACCEL_RW
    global SHARED_FRACTION, VEL_SIGMA_PER_TICK, ACCEL_RW_PER_SQRT_S
    global SURPLUS, TOTAL_DRONES, KILL_TICK
    GPS_MODE = mode
    if gps_interval is not None:
        GPS_INTERVAL_S = gps_interval
    if outage_after is not None:
        OUTAGE_AFTER_S = outage_after
    if accel_rw is not None:
        ACCEL_RW = accel_rw
        ACCEL_RW_PER_SQRT_S = ACCEL_RW / 60.0
        VEL_SIGMA_PER_TICK = ACCEL_RW_PER_SQRT_S * np.sqrt(DRONE_TICK)
    if shared_fraction is not None:
        SHARED_FRACTION = shared_fraction
    SURPLUS = surplus
    TOTAL_DRONES = NUM_DRONES + SURPLUS
    KILL_TICK = kill_tick

    targets = make_sphere(NUM_DRONES)
    all_runs = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        starts = rng.uniform(-WORLD, WORLD, (TOTAL_DRONES, 3))
        m, target_pos, perfect_target_pos, dead = simulate(starts, targets, seed)
        all_runs.append((m, target_pos, perfect_target_pos, dead))

    # Aggregate across seeds at each time step
    times = [m[0][0] for m in all_runs]  # all should match
    # Align by time
    T_count = min(len(r[0]) for r in all_runs)
    abs_arr = np.zeros((N_SEEDS, T_count))
    rel_arr = np.zeros((N_SEEDS, T_count))
    form_arr = np.zeros((N_SEEDS, T_count))
    locked_arr = np.zeros((N_SEEDS, T_count))
    times_arr = np.zeros(T_count)
    for i, (m, _, _, _) in enumerate(all_runs):
        for j in range(T_count):
            t, ad, rd, fe, lc = m[j]
            times_arr[j] = t
            abs_arr[i, j] = ad
            rel_arr[i, j] = rd
            form_arr[i, j] = fe
            locked_arr[i, j] = lc

    # Assignment consistency: in the perfect-info case, what fraction of drones got the same leaf?
    consist_frac = []
    for _, target_pos, perfect_target_pos, dead in all_runs:
        live = ~dead
        same = np.all(np.isclose(target_pos[live], perfect_target_pos[live], atol=1e-4), axis=1)
        consist_frac.append(float(same.mean()))
    return {
        'label': label,
        'times': times_arr,
        'abs_drift': np.mean(abs_arr, axis=0),
        'rel_drift': np.mean(rel_arr, axis=0),
        'form_err': np.mean(form_arr, axis=0),
        'locked': np.mean(locked_arr, axis=0),
        'consist_frac': float(np.mean(consist_frac)),
    }


def main():
    print(f"Localization benchmark: N={NUM_DRONES}, sphere, {N_SEEDS} seeds")
    print(f"Accel RW = {ACCEL_RW} m/s/√hr  (consumer MEMS), shared bias = {SHARED_FRACTION}")
    print()

    regimes = [
        ('GPS on (1Hz)',                  'on',  1.0,  None,  None,  None),
        ('GPS off',                       'off', None, None,  None,  None),
        ('GPS for 30s, then outage',      'outage', 1.0, 30.0, None, None),
        ('Tactical IMU (10× better), off', 'off', None, None, 0.004, None),
        ('All-shared bias, GPS off',       'off', None, None, None,  1.0),
        ('All-independent, GPS off',       'off', None, None, None,  0.0),
    ]

    results = []
    for label, mode, gi, oa, rw, sh in regimes:
        print(f"Running {label}...")
        r = run_regime(label, mode, gps_interval=gi, outage_after=oa,
                       accel_rw=rw, shared_fraction=sh)
        results.append(r)

    final_t = float(results[0]['times'][-1])

    print()
    print(f"Formation error (mean ‖true_pos − target‖) over time:")
    print(f"{'regime':<40} {'t=10s':>8} {'t=30s':>8} {'t=60s':>8} "
          f"{f't={int(final_t)}s':>9}")
    for r in results:
        def at(t):
            i = int(np.argmin(np.abs(r['times'] - t)))
            return r['form_err'][i]
        print(f"{r['label']:<40} {at(10):>7.3f}m {at(30):>7.3f}m "
              f"{at(60):>7.3f}m {r['form_err'][-1]:>8.3f}m")

    # Show abs vs rel drift comparison
    print()
    print(f"Absolute vs relative drift (GPS off, consumer IMU):")
    print(f"{'t_s':>6} {'abs_drift_m':>14} {'rel_drift_m':>14} "
          f"{'form_err_m':>14}")
    r = results[1]
    n_show = 8
    step = max(1, len(r['times']) // n_show)
    for i in range(0, len(r['times']), step):
        print(f"{r['times'][i]:>6.1f} {r['abs_drift'][i]:>14.3f} "
              f"{r['rel_drift'][i]:>14.3f} {r['form_err'][i]:>14.3f}")


if __name__ == '__main__':
    main()
