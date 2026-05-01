# /// script
# dependencies = ["numpy<3"]
# ///
"""Drone-loss recovery benchmark for the hierarchical PCA-tree assignment.

Stepped (non-threaded) simulation so baseline and loss runs are perfectly
reproducible. At a fixed tick we set a `dead` flag for one drone in the
simulated broadcast; surviving drones detect the change on their next
read and rerun compute_leaf_target against the surviving set. They steer
to their (possibly new) target with the same flight controller.

Metrics:
  - reassignment_count: how many surviving drones got a NEW leaf after
    the death event (excluding the dead drone)
  - time_to_lock: ticks/seconds until all survivors lock at their leaves
  - overhead_s: extra time vs the no-death baseline on identical seeds
  - max_extra_distance: longest single drone's flight-from-death-onward
    minus what it would have flown in the baseline scenario

Because baseline has N drones and loss has N-1 survivors, total flight
distance isn't directly comparable; the discriminator is reassignment
count and time overhead.
"""

import time
import numpy as np

import os

# Physics (mirror simulator.py)
NUM_DRONES = 100
SURPLUS = int(os.environ.get("SURPLUS", "0"))
TOTAL_DRONES = NUM_DRONES + SURPLUS
DRONE_TICK = 0.04
APPROACH_RADIUS = 4.0
REPULSION_RADIUS = 3.5
MAX_TICKS = 3000
KILL_TICK = 75   # ~3 s in — typically mid-formation
KILL_ID = int(os.environ.get("KILL_ID", "42"))  # anchor of the kill cluster
CLUSTER_SIZE = int(os.environ.get("CLUSTER_SIZE", "1"))  # 1 = single death
# When CLUSTER_AT_KEYS=1, the cluster victims are drones whose current target
# is one of the first KEY_COUNT primary leaves (the keys), instead of spatial
# neighbors of an anchor drone. Models a threat correlated with key positions.
CLUSTER_AT_KEYS = os.environ.get("CLUSTER_AT_KEYS", "0") == "1"
REASSIGN_THRESHOLD = 1.0  # target moved by more than this counts as reassigned

# Shadow-manifold parameters. KEY_COUNT designates which primary leaves are
# "keys" (the first N of them). USE_SHADOW=1 makes the surplus fleet form a
# shadow sub-manifold of those keys (offset radially inward by SHADOW_OFFSET);
# USE_SHADOW=0 keeps surplus uniform (parent centroids of the primary tree).
# CLUSTER_AT_KEYS uses the same KEY_COUNT to identify cluster victims.
KEY_COUNT = int(os.environ.get("KEY_COUNT", "0"))
SHADOW_OFFSET = float(os.environ.get("SHADOW_OFFSET", "2.0"))
USE_SHADOW = os.environ.get("USE_SHADOW", "0") == "1"

# Two-fleet tiered redundancy. KEY_SURPLUS drones shadow the first KEY_COUNT
# primary leaves (high redundancy at keys). FILLER_SURPLUS drones shadow the
# remaining N-KEY_COUNT non-key leaves (lower redundancy at filler). When
# either is > 0, two-fleet mode is active and overrides USE_SHADOW. Total
# surplus = KEY_SURPLUS + FILLER_SURPLUS.
KEY_SURPLUS = int(os.environ.get("KEY_SURPLUS", "0"))
FILLER_SURPLUS = int(os.environ.get("FILLER_SURPLUS", "0"))
USE_TIERED = (KEY_SURPLUS > 0 or FILLER_SURPLUS > 0) and KEY_COUNT > 0
if USE_TIERED:
    SURPLUS = KEY_SURPLUS + FILLER_SURPLUS
    TOTAL_DRONES = NUM_DRONES + SURPLUS

WORLD = 40.0


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
    """Hierarchical assignment that handles N+S drones for N targets.

    Returns a target *position* (3-vector). When more drones reach a leaf
    than the leaf has positions (which happens whenever surplus > 0), the
    drone closest to the leaf gets the leaf as its target ("primary"); the
    rest get the leaf's parent centroid as their target ("surplus" — they
    park at an interior point ready to fill if a primary is lost).

    With surplus = 0 this reduces to the strict bijective version.
    """
    node = root
    parent = root  # leaf's parent at end of traversal
    cur = list(drones)
    my_pos = next(np.array(d['pos']) for d in drones if d['id'] == my_id)
    while node.left is not None and len(cur) > 1:
        n = len(cur)
        nl = len(node.left.positions)
        nt = len(node.positions)
        dl = max(0, min(n, int(round(n * nl / nt))))
        if dl == 0:
            parent = node
            node = node.right
            continue
        if dl == n:
            parent = node
            node = node.left
            continue
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
                parent = node
                node = subs[i]
                cur = group
                break
    while node.left is not None:
        parent = node
        dl_ = np.linalg.norm(my_pos - node.left.center)
        dr_ = np.linalg.norm(my_pos - node.right.center)
        node = node.left if dl_ <= dr_ else node.right

    leaf_pos = node.positions[0] if len(node.positions) == 1 else node.center

    # Single drone reached this leaf — it's the primary
    if len(cur) == 1:
        return leaf_pos.copy()

    # Multi-drone leaf — closest is primary; the rest park at parent centroid
    distances = sorted(
        (float(np.linalg.norm(np.array(d['pos']) - leaf_pos)), d['id'])
        for d in cur
    )
    primary_id = distances[0][1]
    if my_id == primary_id:
        return leaf_pos.copy()
    return parent.center.copy()


def make_shadow_positions(targets, indices, offset):
    """Shadow positions = primary leaves at the given indices, offset
    radially inward toward the manifold center by `offset`."""
    keys = targets[indices].copy() if not isinstance(indices, slice) else targets[indices].copy()
    center = targets.mean(axis=0)
    radial = keys - center
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    radial_unit = radial / norms
    return keys - offset * radial_unit


def mad_outlier_mask(positions, k=3.0):
    """Robust outlier rejection via MAD on spatial distance from median.
    Returns a boolean mask of inliers. Used as an optional broadcast-
    sanitization layer when adversarial-byzantine drones may be present.
    Empirically catches ~9.8/10 random byzantines at k=3 (vs ~4.5/10 for
    σ-based mean+std rejection); fails on coordinated adversaries that
    shift the spatial median itself (PKI is the necessary primary defense
    in that regime).
    """
    if len(positions) < 4:
        return np.ones(len(positions), dtype=bool)
    centroid = np.median(positions, axis=0)
    dists = np.linalg.norm(positions - centroid, axis=1)
    med = np.median(dists)
    mad = np.median(np.abs(dists - med))
    if mad < 1e-9:
        mad = dists.std() + 1e-9
    threshold = med + k * 1.4826 * mad
    return dists < threshold


def assign_all(positions, dead, primary_tree,
               shadow_tree=None,
               key_shadow_tree=None, filler_shadow_tree=None,
               key_surplus_count=0):
    """Compute target position for every live drone.

    Single-fleet (shadow_tree=None and no tiered): all drones via primary_tree.
    Single-fleet shadow: primary via primary_tree, surplus via shadow_tree.
    Tiered (key_shadow_tree and filler_shadow_tree set): primary via
    primary_tree; the first key_surplus_count surplus drones via
    key_shadow_tree; the remaining surplus via filler_shadow_tree.
    """
    n = len(positions)
    targets_pos = np.full((n, 3), np.nan)

    # Primary fleet: assign via primary tree.
    primary_drones = [{'id': i, 'pos': positions[i].copy()}
                      for i in range(NUM_DRONES) if not dead[i]]
    for i in range(NUM_DRONES):
        if dead[i]:
            continue
        targets_pos[i] = compute_target(i, primary_drones, primary_tree)

    if key_shadow_tree is not None or filler_shadow_tree is not None:
        # Tiered surplus: split at boundary key_surplus_count.
        boundary = NUM_DRONES + key_surplus_count
        key_shadow_drones = [{'id': i, 'pos': positions[i].copy()}
                             for i in range(NUM_DRONES, boundary) if not dead[i]]
        filler_shadow_drones = [{'id': i, 'pos': positions[i].copy()}
                                for i in range(boundary, n) if not dead[i]]
        if key_shadow_drones and key_shadow_tree is not None:
            for i in range(NUM_DRONES, boundary):
                if dead[i]:
                    continue
                targets_pos[i] = compute_target(i, key_shadow_drones, key_shadow_tree)
        if filler_shadow_drones and filler_shadow_tree is not None:
            for i in range(boundary, n):
                if dead[i]:
                    continue
                targets_pos[i] = compute_target(
                    i, filler_shadow_drones, filler_shadow_tree)
        return targets_pos

    if shadow_tree is None:
        # Single-fleet (no surplus or uniform): everyone via primary_tree.
        all_survivors = [{'id': i, 'pos': positions[i].copy()}
                         for i in range(n) if not dead[i]]
        for i in range(NUM_DRONES, n):
            if dead[i]:
                continue
            targets_pos[i] = compute_target(i, all_survivors, primary_tree)
        # Re-assign primary too with the full set so partition matches.
        for i in range(NUM_DRONES):
            if dead[i]:
                continue
            targets_pos[i] = compute_target(i, all_survivors, primary_tree)
        return targets_pos

    # Single-fleet shadow.
    surplus_drones = [{'id': i, 'pos': positions[i].copy()}
                      for i in range(NUM_DRONES, n) if not dead[i]]
    if surplus_drones:
        for i in range(NUM_DRONES, n):
            if dead[i]:
                continue
            targets_pos[i] = compute_target(i, surplus_drones, shadow_tree)
    return targets_pos


def classify_primary(target_pos, targets, dead):
    """A drone is 'primary' iff its target is one of the manifold's leaves.
    Surplus drones target an internal-node centroid instead.
    """
    n = len(target_pos)
    is_primary = np.zeros(n, dtype=bool)
    for i in range(n):
        if dead[i] or np.any(np.isnan(target_pos[i])):
            continue
        dmin = float(np.linalg.norm(targets - target_pos[i], axis=1).min())
        is_primary[i] = dmin < 1e-6
    return is_primary


def patch_after_death(target_pos, is_primary, positions, dead, dead_id):
    """Promote the closest live surplus drone to the dead drone's leaf —
    but only if the dead drone was a primary. Surplus deaths just
    decrement the pool with no patch needed (there is no leaf to fill).

    This local O(N) patch changes exactly one drone's target when a
    primary dies, vs the full bisection rerun that cascades partition
    shifts down the tree. Returns (new_targets, new_is_primary,
    promoted_id_or_None). `promoted_id` is None when (a) the dead drone
    was a surplus, (b) it had a NaN target (already removed), or (c) no
    live surplus is available.
    """
    new_targets = target_pos.copy()
    new_primary = is_primary.copy()
    if np.any(np.isnan(target_pos[dead_id])):
        return new_targets, new_primary, None
    # Surplus death: pool shrinks, no patch needed (no primary leaf empty).
    if not is_primary[dead_id]:
        new_targets[dead_id] = np.nan
        new_primary[dead_id] = False
        return new_targets, new_primary, None
    # Primary death: promote closest live surplus.
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


def simulate(starts, targets, kill=None):
    """Step the simulation. Returns metrics dict."""
    n = len(starts)
    primary_tree = ManifoldNode(targets)
    shadow_tree = None
    key_shadow_tree = None
    filler_shadow_tree = None

    if USE_TIERED:
        if KEY_SURPLUS > 0:
            key_idx = list(range(KEY_COUNT))
            key_shadow_positions = make_shadow_positions(targets, key_idx, SHADOW_OFFSET)
            key_shadow_tree = ManifoldNode(key_shadow_positions)
        if FILLER_SURPLUS > 0:
            filler_idx = list(range(KEY_COUNT, NUM_DRONES))
            filler_shadow_positions = make_shadow_positions(
                targets, filler_idx, SHADOW_OFFSET)
            filler_shadow_tree = ManifoldNode(filler_shadow_positions)
    elif USE_SHADOW and KEY_COUNT > 0 and SURPLUS > 0:
        shadow_positions = make_shadow_positions(
            targets, list(range(KEY_COUNT)), SHADOW_OFFSET)
        shadow_tree = ManifoldNode(shadow_positions)

    pos = starts.copy().astype(float)
    vel = np.zeros((n, 3))
    locked = np.zeros(n, dtype=bool)
    dead = np.zeros(n, dtype=bool)
    stall = np.zeros(n, dtype=int)
    last_dist = np.full(n, np.inf)
    flight_dist = np.zeros(n)
    pos_at_death = None
    flight_at_death = None

    target_pos = assign_all(
        pos, dead, primary_tree, shadow_tree,
        key_shadow_tree, filler_shadow_tree, KEY_SURPLUS)
    is_primary = classify_primary(target_pos, targets, dead)
    initial_targets = target_pos.copy()
    targets_after_death = None
    cluster_dead_ids = []
    n_promoted = 0
    n_unfilled = 0

    for tick in range(MAX_TICKS):
        # Inject cluster kill at the requested tick.
        if kill is not None and tick == kill[0] and not dead[kill[1]]:
            pos_at_death = pos.copy()
            flight_at_death = flight_dist.copy()
            anchor_id = kill[1]
            if CLUSTER_AT_KEYS and KEY_COUNT > 0:
                # Threat-correlated-with-keys: kill drones whose current target
                # is one of the first KEY_COUNT primary leaves.
                key_positions = targets[:KEY_COUNT]
                cluster_dead_ids = []
                for i in range(NUM_DRONES):
                    if dead[i]:
                        continue
                    if np.any(np.isnan(target_pos[i])):
                        continue
                    diffs = np.linalg.norm(target_pos[i] - key_positions, axis=1)
                    if diffs.min() < 1e-6:
                        cluster_dead_ids.append(i)
                    if len(cluster_dead_ids) >= CLUSTER_SIZE:
                        break
            else:
                # Spatial cluster: anchor + nearest-in-position live drones.
                distances = np.linalg.norm(pos - pos[anchor_id], axis=1)
                distances[dead] = np.inf
                distances[anchor_id] = -1.0
                cluster_dead_ids = list(np.argsort(distances)[:CLUSTER_SIZE])
            for did in cluster_dead_ids:
                dead[int(did)] = True

            if SURPLUS > 0:
                # Sequential patch: each PRIMARY death promotes the nearest
                # live surplus. Surplus deaths just decrement the pool with
                # no patch needed (there is no primary leaf to fill).
                # `n_unfilled` counts only primary-leaf failures (true gaps
                # in the formation), matching Lemma 7's framing of K *primary*
                # deaths against S surplus.
                for did in cluster_dead_ids:
                    was_primary = bool(is_primary[int(did)])
                    new_targets, new_primary, promoted = patch_after_death(
                        target_pos, is_primary, pos, dead, int(did))
                    target_pos = new_targets
                    is_primary = new_primary
                    if promoted is not None:
                        locked[promoted] = False
                        stall[promoted] = 0
                        last_dist[promoted] = np.inf
                        n_promoted += 1
                    elif was_primary:
                        # Primary leaf left empty — true unfilled gap.
                        n_unfilled += 1
            else:
                # Full bisection rerun on the surviving fleet (cascades).
                new_targets = assign_all(pos, dead, primary_tree, shadow_tree)
                for i in range(n):
                    if dead[i]:
                        continue
                    if not np.allclose(new_targets[i], target_pos[i]):
                        locked[i] = False
                        stall[i] = 0
                        last_dist[i] = np.inf
                target_pos = new_targets
            targets_after_death = target_pos.copy()

        survivors = ~dead
        if survivors.any() and locked[survivors].all():
            break

        # Advance each live, non-locked drone one step
        new_pos = pos.copy()
        new_vel = vel.copy()
        new_locked = locked.copy()
        for i in range(n):
            if dead[i] or locked[i]:
                continue
            tgt = target_pos[i]
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

            s = float(np.linalg.norm(v))
            max_speed = 0.8
            if is_final and dist < 3.0:
                max_speed = max(0.05, dist * 0.2)
            if s > max_speed:
                v = (v / s) * max_speed
                s = max_speed

            new_vel[i] = v
            new_pos[i] = pos[i] + v
            flight_dist[i] += s

            if is_final and dist < 0.3:
                new_pos[i] = tgt.copy()
                new_vel[i] = np.zeros(3)
                new_locked[i] = True

        pos, vel, locked = new_pos, new_vel, new_locked

    return {
        'tick': tick,
        'time_s': tick * DRONE_TICK,
        'locked': int(locked[~dead].sum()),
        'survivors': int((~dead).sum()),
        'flight': flight_dist.copy(),
        'flight_at_death': flight_at_death,
        'pos_at_death': pos_at_death,
        'initial_targets': initial_targets,
        'final_targets': target_pos.copy(),
        'targets_after_death': targets_after_death,
        'cluster_dead_ids': cluster_dead_ids,
        'n_promoted': n_promoted,
        'n_unfilled': n_unfilled,
    }


def reassigned_count(initial_targets, after_death_targets, dead_ids):
    """Count drones whose target moved by > REASSIGN_THRESHOLD (excl. dead)."""
    if after_death_targets is None:
        return 0
    diff = np.linalg.norm(initial_targets - after_death_targets, axis=1)
    for did in dead_ids:
        diff[int(did)] = 0
    return int((diff > REASSIGN_THRESHOLD).sum())


def main():
    if USE_TIERED:
        surplus_kind = (f"tiered (key={KEY_SURPLUS}, filler={FILLER_SURPLUS}, "
                        f"keys={KEY_COUNT}, offset={SHADOW_OFFSET})")
    elif USE_SHADOW and KEY_COUNT > 0 and SURPLUS > 0:
        surplus_kind = f"shadow (keys={KEY_COUNT}, offset={SHADOW_OFFSET})"
    elif SURPLUS > 0:
        surplus_kind = "uniform"
    else:
        surplus_kind = "none"
    print(f"Loss benchmark: N={NUM_DRONES} primary + {SURPLUS} surplus "
          f"({surplus_kind}) = {TOTAL_DRONES} drones, sphere")
    print(f"Cluster size = {CLUSTER_SIZE} (anchor = drone {KILL_ID} + "
          f"{CLUSTER_SIZE - 1} nearest live neighbors)")
    print(f"Kill at tick {KILL_TICK} (t={KILL_TICK * DRONE_TICK:.2f}s)\n")
    targets = make_sphere(NUM_DRONES)

    seeds = [0, 1, 2, 3, 4]
    print(f"{'seed':>5} {'base_t':>8} {'loss_t':>8} "
          f"{'overhd':>7} {'reassgn':>8} {'reassgn_pct':>13} "
          f"{'promoted':>9} {'unfilled':>9} {'max_extra':>10}")

    overall = {'reassign': [], 'overhead': [], 'max_extra': [],
               'promoted': [], 'unfilled': []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        starts = rng.uniform(-WORLD, WORLD, (TOTAL_DRONES, 3))

        base = simulate(starts, targets)
        loss = simulate(starts, targets, kill=(KILL_TICK, KILL_ID))

        dead_ids = loss['cluster_dead_ids']
        n_reassigned = reassigned_count(
            loss['initial_targets'], loss['targets_after_death'], dead_ids)
        n_alive = TOTAL_DRONES - len(dead_ids)
        overhead = loss['time_s'] - base['time_s']

        if loss['flight_at_death'] is not None and n_reassigned > 0:
            extra = loss['flight'] - loss['flight_at_death']
            diffs = np.linalg.norm(
                loss['initial_targets'] - loss['targets_after_death'], axis=1)
            mask = diffs > REASSIGN_THRESHOLD
            for did in dead_ids:
                mask[int(did)] = False
            max_extra = float(extra[mask].max()) if mask.any() else 0.0
        else:
            max_extra = 0.0

        print(f"{seed:>5d} {base['time_s']:>7.2f}s {loss['time_s']:>7.2f}s "
              f"{overhead:>+6.2f}s {n_reassigned:>8d} "
              f"{100*n_reassigned/n_alive:>12.1f}% "
              f"{loss['n_promoted']:>9d} {loss['n_unfilled']:>9d} "
              f"{max_extra:>10.2f}")

        overall['reassign'].append(n_reassigned / n_alive)
        overall['overhead'].append(overhead)
        overall['max_extra'].append(max_extra)
        overall['promoted'].append(loss['n_promoted'])
        overall['unfilled'].append(loss['n_unfilled'])

    def _ci(samples, ci=0.95, n_resamples=2000):
        samples = np.asarray(samples)
        if len(samples) == 0:
            return (float('nan'), float('nan'))
        rng = np.random.default_rng(0)
        n = len(samples)
        means = np.empty(n_resamples)
        for i in range(n_resamples):
            means[i] = samples[rng.integers(0, n, n)].mean()
        return (float(np.percentile(means, 100 * (1 - ci) / 2)),
                float(np.percentile(means, 100 * (1 + ci) / 2)))

    reassign_pct = 100 * np.array(overall['reassign'])
    overhead = np.array(overall['overhead'])
    max_extra = np.array(overall['max_extra'])
    rl, rh = _ci(reassign_pct)
    ol, oh = _ci(overhead)
    ml, mh = _ci(max_extra)
    print()
    print(f"Mean reassignment: {reassign_pct.mean():.1f}% [{rl:.1f}, {rh:.1f}] (95% CI)")
    print(f"Mean overhead: {overhead.mean():+.2f}s [{ol:+.2f}, {oh:+.2f}]")
    print(f"Mean max-extra-distance: {max_extra.mean():.2f} [{ml:.2f}, {mh:.2f}]")
    print(f"Mean promoted: {np.mean(overall['promoted']):.1f}, "
          f"unfilled: {np.mean(overall['unfilled']):.1f}")


if __name__ == '__main__':
    main()
