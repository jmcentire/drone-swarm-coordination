# /// script
# dependencies = ["numpy<3"]
# ///
"""Sustained attrition: how does formation quality degrade over time as
losses accumulate?

A Poisson loss process delivers drone deaths at rate λ over a mission of
duration T. Each death triggers the patch protocol; surplus is consumed
until depleted, after which losses leave unfilled gaps.

Three surplus policies:
  uniform    — all surplus drones at primary-tree parent centroids
  shadow     — surplus shadows the first KEY_COUNT primary leaves
  reactive   — no static surplus; on death, the closest live primary
               drone moves to fill the gap, propagating the gap outward
               via its own next-nearest, etc., until the formation reaches
               a new equilibrium with one drone fewer

Metrics over time:
  formation_size     — primary leaves still occupied
  formation_quality  — mean ||true_pos - target_leaf|| over occupied leaves
  cumulative_losses  — total deaths
  promotions         — total successful patches

The interesting comparisons:
  - Static-manifold (leave gaps) vs adaptive-manifold (drop low-priority
    leaves to preserve bijection on remaining swarm)
  - Uniform-surplus vs shadow-surplus under sustained attrition
  - Reactive (no surplus, propagate gap) vs proactive (surplus + patch)

Operational claim being tested: the architecture degrades gracefully —
formation quality decays roughly linearly in cumulative losses up to the
surplus capacity, then steps down once per loss as gaps appear.
"""

import os
import numpy as np

NUM_DRONES = 100
DRONE_TICK = 0.04
APPROACH_RADIUS = 4.0
REPULSION_RADIUS = 3.5
MAX_TICKS = int(os.environ.get("MAX_TICKS", "3000"))  # 120s
WORLD = 40.0

LOSS_RATE_PER_S = float(os.environ.get("LOSS_RATE", "0.05"))  # 1 per 20s avg
LOSS_START_S = float(os.environ.get("LOSS_START", "10.0"))   # let formation form first
SURPLUS = int(os.environ.get("SURPLUS", "0"))
TOTAL_DRONES = NUM_DRONES + SURPLUS
KEY_COUNT = int(os.environ.get("KEY_COUNT", "0"))
SHADOW_OFFSET = float(os.environ.get("SHADOW_OFFSET", "2.0"))
USE_SHADOW = os.environ.get("USE_SHADOW", "0") == "1"
ADAPTIVE_MANIFOLD = os.environ.get("ADAPTIVE", "0") == "1"
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))


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


def make_shadow_positions(targets, indices, offset):
    keys = targets[indices].copy()
    center = targets.mean(axis=0)
    radial = keys - center
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return keys - offset * (radial / norms)


def assign_all(positions, dead, primary_tree, shadow_tree=None):
    n = len(positions)
    targets_pos = np.full((n, 3), np.nan)

    if shadow_tree is None:
        # Single-fleet: all live drones bisect against primary_tree as one
        # group. With N+S drones and N leaves, exactly S drones end up as
        # surplus (at parent centroids of multi-occupancy leaves). Which
        # drones become surplus is determined by spatial proximity, not by
        # drone ID — drone-ID identity does not match primary/surplus role.
        all_drones = [{'id': i, 'pos': positions[i].copy()}
                      for i in range(n) if not dead[i]]
        for i in range(n):
            if dead[i]:
                continue
            targets_pos[i] = compute_target(i, all_drones, primary_tree)
        return targets_pos

    # Two-fleet: primary fleet (drones 0..NUM_DRONES-1) bisects against
    # primary_tree as a bijective set; surplus fleet (drones NUM_DRONES..)
    # bisects against shadow_tree.
    primary_drones = [{'id': i, 'pos': positions[i].copy()}
                      for i in range(NUM_DRONES) if not dead[i]]
    for i in range(NUM_DRONES):
        if dead[i]:
            continue
        targets_pos[i] = compute_target(i, primary_drones, primary_tree)
    surplus_drones = [{'id': i, 'pos': positions[i].copy()}
                      for i in range(NUM_DRONES, n) if not dead[i]]
    if surplus_drones:
        for i in range(NUM_DRONES, n):
            if dead[i]:
                continue
            targets_pos[i] = compute_target(i, surplus_drones, shadow_tree)
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


def patch_after_death(target_pos, is_primary, positions, dead, dead_id):
    new_targets = target_pos.copy()
    new_primary = is_primary.copy()
    if np.any(np.isnan(target_pos[dead_id])):
        return new_targets, new_primary, None
    dead_target = target_pos[dead_id].copy()
    new_targets[dead_id] = np.nan
    new_primary[dead_id] = False
    surplus_ids = [i for i in range(len(positions))
                   if not dead[i] and not is_primary[i]]
    if not surplus_ids:
        return new_targets, new_primary, None
    distances = sorted(
        (float(np.linalg.norm(positions[i] - dead_target)), i)
        for i in surplus_ids
    )
    promoted = distances[0][1]
    new_targets[promoted] = dead_target
    new_primary[promoted] = True
    return new_targets, new_primary, promoted


def step_drone(i, pos, target_pos, vel, dead, locked, stall, last_dist, n):
    """Compute commanded velocity for one drone based on est_pos = pos."""
    if dead[i] or locked[i]:
        return np.zeros(3)
    tgt = target_pos[i]
    if np.any(np.isnan(tgt)):
        return np.zeros(3)
    diff = tgt - pos[i]
    dist = float(np.linalg.norm(diff))
    is_final = dist < APPROACH_RADIUS
    attr = (diff / dist) * min(0.6, dist * 0.1) if dist > 0 else np.zeros(3)
    er = REPULSION_RADIUS * 0.4 if is_final else REPULSION_RADIUS
    rep = np.zeros(3)
    for j in range(n):
        if j == i or dead[j] or locked[j]:
            continue
        d = pos[i] - pos[j]
        r = float(np.linalg.norm(d))
        if 0 < r < er:
            unit = d / r
            closing = max(0.0, -np.dot(vel[i] - vel[j], unit))
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
    return v


def simulate(starts, targets, seed=0):
    rng = np.random.default_rng(seed)
    n = len(starts)
    primary_tree = ManifoldNode(targets)
    shadow_tree = None
    if USE_SHADOW and KEY_COUNT > 0 and SURPLUS > 0:
        shadow_positions = make_shadow_positions(
            targets, list(range(KEY_COUNT)), SHADOW_OFFSET)
        shadow_tree = ManifoldNode(shadow_positions)

    pos = starts.copy().astype(float)
    vel = np.zeros((n, 3))
    locked = np.zeros(n, dtype=bool)
    dead = np.zeros(n, dtype=bool)
    stall = np.zeros(n, dtype=int)
    last_dist = np.full(n, np.inf)

    target_pos = assign_all(pos, dead, primary_tree, shadow_tree)
    is_primary = classify_primary(target_pos, targets, dead)

    measurements = []
    cumulative_deaths = 0
    promotions = 0
    unfilled = 0

    loss_start_tick = int(LOSS_START_S / DRONE_TICK)
    p_per_tick = LOSS_RATE_PER_S * DRONE_TICK

    for tick in range(MAX_TICKS):
        # Poisson loss process (after formation start)
        if tick >= loss_start_tick:
            r = rng.random()
            if r < p_per_tick:
                # Pick a random live drone to die. Prefer killing primaries
                # because that's the operational threat (a "drone hit" usually
                # means a flying drone, and primary drones are flying around).
                live_primaries = [i for i in range(n)
                                  if not dead[i] and is_primary[i]]
                if live_primaries:
                    victim = int(rng.choice(live_primaries))
                    dead[victim] = True
                    cumulative_deaths += 1
                    new_targets, new_primary, promoted = patch_after_death(
                        target_pos, is_primary, pos, dead, victim)
                    target_pos = new_targets
                    is_primary = new_primary
                    if promoted is not None:
                        promotions += 1
                        locked[promoted] = False
                        stall[promoted] = 0
                        last_dist[promoted] = np.inf
                    else:
                        unfilled += 1

        # Step drones
        new_vel = np.zeros((n, 3))
        for i in range(n):
            if dead[i] or locked[i]:
                continue
            new_vel[i] = step_drone(i, pos, target_pos, vel,
                                    dead, locked, stall, last_dist, n)
        vel = new_vel
        pos += vel

        # Lock detection
        for i in range(n):
            if dead[i] or locked[i]:
                continue
            if np.any(np.isnan(target_pos[i])):
                continue
            dist = float(np.linalg.norm(pos[i] - target_pos[i]))
            if dist < 0.3:
                pos[i] = target_pos[i].copy()
                vel[i] = np.zeros(3)
                locked[i] = True

        # Sample measurements at coarse intervals (every 1 second)
        if tick % 25 == 0:
            t = tick * DRONE_TICK
            # Count occupied primary leaves (by role, not by ID).
            occupied_primaries = sum(1 for i in range(n)
                                      if not dead[i] and is_primary[i])
            errs = []
            for i in range(n):
                if dead[i] or not is_primary[i] or not locked[i]:
                    continue
                err = float(np.linalg.norm(pos[i] - target_pos[i]))
                errs.append(err)
            mean_err = float(np.mean(errs)) if errs else 0.0
            measurements.append({
                't': t,
                'occupied_primaries': occupied_primaries,
                'cum_losses': cumulative_deaths,
                'promotions': promotions,
                'unfilled': unfilled,
                'form_err': mean_err,
            })

    return measurements


def run_config(label, surplus, key_count=0, use_shadow=False,
               loss_rate=None):
    global SURPLUS, TOTAL_DRONES, KEY_COUNT, USE_SHADOW, LOSS_RATE_PER_S
    SURPLUS = surplus
    TOTAL_DRONES = NUM_DRONES + SURPLUS
    KEY_COUNT = key_count
    USE_SHADOW = use_shadow
    if loss_rate is not None:
        LOSS_RATE_PER_S = loss_rate

    targets = make_sphere(NUM_DRONES)
    all_runs = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        starts = rng.uniform(-WORLD, WORLD, (TOTAL_DRONES, 3))
        m = simulate(starts, targets, seed)
        all_runs.append(m)

    # Aggregate over seeds
    n_steps = min(len(m) for m in all_runs)
    times = np.array([m[0]['t'] for m in all_runs[:1]] * 1).flatten()  # placeholder
    times = np.array([all_runs[0][i]['t'] for i in range(n_steps)])
    occupied = np.array([
        [all_runs[s][i]['occupied_primaries'] for s in range(N_SEEDS)]
        for i in range(n_steps)
    ])
    losses = np.array([
        [all_runs[s][i]['cum_losses'] for s in range(N_SEEDS)]
        for i in range(n_steps)
    ])
    form_err = np.array([
        [all_runs[s][i]['form_err'] for s in range(N_SEEDS)]
        for i in range(n_steps)
    ])
    return {
        'label': label,
        'times': times,
        'occupied_mean': occupied.mean(axis=1),
        'losses_mean': losses.mean(axis=1),
        'form_err_mean': form_err.mean(axis=1),
        'final_unfilled_mean': float(np.mean([m[-1]['unfilled'] for m in all_runs])),
        'final_losses_mean': float(np.mean([m[-1]['cum_losses'] for m in all_runs])),
        'final_promotions_mean': float(np.mean([m[-1]['promotions'] for m in all_runs])),
    }


def main():
    print(f"Sustained attrition: N={NUM_DRONES}, sphere, "
          f"loss rate {LOSS_RATE_PER_S}/s, "
          f"duration {MAX_TICKS * DRONE_TICK:.0f}s, {N_SEEDS} seeds")
    print()

    configs = [
        ('No surplus',                  0,   0,  False),
        ('Uniform surplus = 10',       10,   0,  False),
        ('Shadow surplus = 10 (KEY=10)', 10, 10, True),
        ('Uniform surplus = 30',       30,   0,  False),
    ]

    results = []
    for label, surp, key, shad in configs:
        print(f"Running {label}...")
        r = run_config(label, surp, key, shad)
        results.append(r)

    # Print summary at key time points
    print()
    print(f"{'config':<32} {'losses':>9} {'promoted':>10} {'unfilled':>10} "
          f"{'final_occupied':>16}")
    for r in results:
        final_occupied = r['occupied_mean'][-1]
        print(f"{r['label']:<32} {r['final_losses_mean']:>8.1f}  "
              f"{r['final_promotions_mean']:>9.1f} "
              f"{r['final_unfilled_mean']:>9.1f} {final_occupied:>15.1f}")

    print()
    print("Trajectory: occupied primaries over time")
    print(f"{'time_s':>8}", end="")
    for r in results:
        print(f" {r['label']:>32}", end="")
    print()
    n_steps = len(results[0]['times'])
    step = max(1, n_steps // 10)
    for i in range(0, n_steps, step):
        print(f"{results[0]['times'][i]:>8.1f}", end="")
        for r in results:
            print(f" {r['occupied_mean'][i]:>32.1f}", end="")
        print()


if __name__ == '__main__':
    main()
