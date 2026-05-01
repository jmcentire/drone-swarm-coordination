# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Empirically compare assignment quality across three methods.

Metric: total path length = sum over drones of straight-line distance from
start to assigned leaf. Hungarian is the optimum. We compare:
  1. Hungarian (scipy.optimize.linear_sum_assignment) — O(n^3) optimal
  2. Hierarchical 2x2 PCA tree (this method) — O(n log n) per drone,
     fully parallel across drones
  3. Greedy nearest-neighbor — each drone takes closest unassigned target,
     processed in order of cheapest available match

Reports mean total cost, mean max single cost, and wall-clock time, plus
the percentage gap vs Hungarian. Uses identical starts and targets across
methods and seeds so any difference is attributable to the algorithm.
"""

import time
import numpy as np
from scipy.optimize import linear_sum_assignment

NUM_DRONES = 100
SEEDS = list(range(10))
WORLD = 40.0  # starts uniform in [-W, W]^3


# --- Manifold builders (mirrors simulator.py) ---
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


_PLASTIC = 1.32471795724474602596


def _r2(i):
    a1, a2 = 1.0 / _PLASTIC, 1.0 / (_PLASTIC ** 2)
    return (0.5 + a1 * i) % 1.0, (0.5 + a2 * i) % 1.0


def make_torus(n, R=12, r=5, center=(0, 0, 20)):
    cx, cy, cz = center
    pts = np.empty((n, 3))
    for i in range(n):
        u, v = _r2(i)
        theta, phi = 2 * np.pi * u, 2 * np.pi * v
        pts[i] = [
            (R + r * np.cos(phi)) * np.cos(theta) + cx,
            (R + r * np.cos(phi)) * np.sin(theta) + cy,
            r * np.sin(phi) + cz,
        ]
    return pts


def make_cube_shell(n, size=16, center=(0, 0, 20)):
    cx, cy, cz = center
    half = size / 2
    side = 4
    cell = size / side

    def m(face, u, v):
        return [(half, u, v), (-half, u, v), (u, half, v),
                (u, -half, v), (u, v, half), (u, v, -half)][face]

    base, extra = n // 6, n - (n // 6) * 6
    positions = []
    for face in range(6):
        count = base + (1 if face < extra else 0)
        grid = []
        for row in range(side):
            cols = range(side) if row % 2 == 0 else range(side - 1, -1, -1)
            for col in cols:
                u = -half + (row + 0.5) * cell
                v = -half + (col + 0.5) * cell
                grid.append((u, v))
        grid.append((0.0, 0.0))
        for u, v in grid[:count]:
            p = m(face, u, v)
            positions.append([p[0] + cx, p[1] + cy, p[2] + cz])
    return np.array(positions[:n])


def make_star_3d(n, outer_r=20, inner_r=9, depth=8, center=(0, 0, 20)):
    cx, cy, cz = center
    MIN_SEP = 2.0

    def star_point(t):
        seg = t * 10
        seg_idx = int(seg) % 10
        seg_frac = seg - int(seg)
        ao = lambda k: np.pi / 2 + (k % 5) * 2 * np.pi / 5
        ai = lambda k: np.pi / 2 + (k + 0.5) * 2 * np.pi / 5
        if seg_idx % 2 == 0:
            a1, r1 = ao(seg_idx // 2), outer_r
            a2, r2 = ai(seg_idx // 2), inner_r
        else:
            a1, r1 = ai(seg_idx // 2), inner_r
            a2, r2 = ao(seg_idx // 2 + 1), outer_r
        x = r1 * np.cos(a1) * (1 - seg_frac) + r2 * np.cos(a2) * seg_frac
        y = r1 * np.sin(a1) * (1 - seg_frac) + r2 * np.sin(a2) * seg_frac
        return x, y

    n_layers = max(2, int(np.ceil(depth / MIN_SEP)))
    pts_per_layer = int(np.ceil(n / n_layers))
    positions = []
    for layer in range(n_layers):
        z_off = -depth / 2 + depth * layer / max(1, n_layers - 1)
        for i in range(pts_per_layer):
            if len(positions) >= n:
                break
            t = (i + 0.5 * (layer % 2)) / pts_per_layer
            x, y = star_point(t)
            positions.append([x + cx, y + cy, cz + z_off])
    return np.array(positions[:n])


# --- Hierarchical PCA tree assignment (this method) ---
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


def compute_leaf(my_id, drones, root, allow_swap=True):
    """Hierarchical 2x2 assignment.

    With allow_swap=True this matches simulator.py exactly, including the
    cost-crossed swap that can break the count invariant in lopsided
    subtrees (and so does not guarantee a bijective assignment).
    With allow_swap=False the swap is disabled — drones always go to the
    same-side target subtree as their projection-half, preserving the
    count invariant and producing a strictly bijective assignment.
    """
    node = root
    cur = list(drones)
    my_pos = next(np.array(d['pos']) for d in drones if d['id'] == my_id)
    while node.left is not None and len(cur) > 1:
        n = len(cur)
        nl = len(node.left.positions)
        nt = len(node.positions)
        dl = max(0, min(n, int(round(n * nl / nt))))
        if dl == 0:
            node = node.right
            continue
        if dl == n:
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
        if allow_swap:
            gc = [np.mean([d['pos'] for d in g], axis=0) for g in groups]
            tc = [s.center for s in subs]
            if (np.linalg.norm(gc[0] - tc[1]) + np.linalg.norm(gc[1] - tc[0]) <
                    np.linalg.norm(gc[0] - tc[0]) + np.linalg.norm(gc[1] - tc[1])):
                subs = [subs[1], subs[0]]
        for i, group in enumerate(groups):
            if any(d['id'] == my_id for d in group):
                node = subs[i]
                cur = group
                break
    while node.left is not None:
        dl_ = np.linalg.norm(my_pos - node.left.center)
        dr_ = np.linalg.norm(my_pos - node.right.center)
        node = node.left if dl_ <= dr_ else node.right
    if len(node.positions) == 1:
        return node.positions[0]
    return node.center


def _assign_hier(starts, targets, allow_swap):
    tree = ManifoldNode(targets)
    drones = [{'id': i, 'pos': p.copy()} for i, p in enumerate(starts)]
    n = len(starts)
    assignment = np.zeros(n, dtype=int)
    for i in range(n):
        leaf = compute_leaf(i, drones, tree, allow_swap=allow_swap)
        dists = np.linalg.norm(targets - leaf, axis=1)
        assignment[i] = int(np.argmin(dists))
    return assignment


def assign_hierarchical_swap(starts, targets):
    return _assign_hier(starts, targets, allow_swap=True)


def assign_hierarchical_strict(starts, targets):
    return _assign_hier(starts, targets, allow_swap=False)


def assign_hungarian(starts, targets):
    cost = np.linalg.norm(starts[:, None] - targets[None, :], axis=-1)
    _, col = linear_sum_assignment(cost)
    return col


def assign_greedy_nn(starts, targets):
    n = len(starts)
    cost = np.linalg.norm(starts[:, None] - targets[None, :], axis=-1)
    assigned = -np.ones(n, dtype=int)
    target_used = np.zeros(n, dtype=bool)
    drone_order = np.argsort(cost.min(axis=1))
    for i in drone_order:
        for t in np.argsort(cost[i]):
            if not target_used[t]:
                assigned[i] = int(t)
                target_used[t] = True
                break
    return assigned


def cost_sum(starts, targets, a):
    return float(np.linalg.norm(starts - targets[a], axis=1).sum())


def cost_max(starts, targets, a):
    return float(np.linalg.norm(starts - targets[a], axis=1).max())


def main():
    methods = [
        ('Hungarian',                  assign_hungarian),
        ('Hierarchical 2x2 (swap)',    assign_hierarchical_swap),
        ('Hierarchical 2x2 (strict)',  assign_hierarchical_strict),
        ('Greedy NN',                  assign_greedy_nn),
    ]
    manifolds = [
        ('sphere', make_sphere),
        ('torus', make_torus),
        ('cube', make_cube_shell),
        ('star', make_star_3d),
    ]
    print(f"Benchmark: N={NUM_DRONES}, seeds={len(SEEDS)}, starts ~ U[-{WORLD},{WORLD}]^3\n")
    for mname, mfn in manifolds:
        targets = mfn(NUM_DRONES)
        rows = {name: {'sum': [], 'max': [], 'time': [], 'dups': []}
                for name, _ in methods}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            starts = rng.uniform(-WORLD, WORLD, (NUM_DRONES, 3))
            for name, fn in methods:
                t0 = time.perf_counter()
                a = fn(starts, targets)
                dt = time.perf_counter() - t0
                dups = len(a) - len(np.unique(a))
                rows[name]['dups'].append(dups)
                # Cost is computed as straight-line start->assigned; if the
                # algorithm produces duplicates, cost is still well-defined
                # but represents a non-bijective assignment.
                rows[name]['sum'].append(cost_sum(starts, targets, a))
                rows[name]['max'].append(cost_max(starts, targets, a))
                rows[name]['time'].append(dt * 1000)
        print(f"--- {mname} ---")
        baseline = float(np.mean(rows['Hungarian']['sum']))
        print(f"{'method':<28} {'total':>10} {'max':>8} {'time_ms':>10} "
              f"{'dups':>6} {'vs Hung':>10}")
        for name, _ in methods:
            r = rows[name]
            tot = float(np.mean(r['sum']))
            mx = float(np.mean(r['max']))
            tm = float(np.mean(r['time']))
            dp = float(np.mean(r['dups']))
            delta = '---' if name == 'Hungarian' else f"{(tot / baseline - 1.0) * 100:+5.2f}%"
            print(f"{name:<28} {tot:>10.1f} {mx:>8.2f} {tm:>10.2f} "
                  f"{dp:>6.1f} {delta:>10}")
        print()


def bootstrap_ci(samples, ci=0.95, n_resamples=2000, rng=None):
    """Bootstrap CI on the mean. Returns (lo, hi)."""
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


def main_scaling(manifolds=None, Ns=None, seeds_per_n=None):
    """Sweep N to see how the hierarchical-vs-Hungarian gap scales."""
    if manifolds is None:
        manifolds = [('sphere', make_sphere)]
    if Ns is None:
        Ns = [10, 30, 100, 300, 1000, 3000]
    if seeds_per_n is None:
        seeds_per_n = {10: 30, 30: 30, 100: 20, 300: 10, 1000: 5, 3000: 3, 10000: 2}
    print(f"\n=== Scaling sweep, starts ~ U[-{WORLD},{WORLD}]^3 ===\n")
    for mname, mfn in manifolds:
        if mname != 'sphere':
            print(f"\n--- {mname} ---")
        else:
            print(f"--- {mname} ---")
        print(f"{'N':>6} {'seeds':>5} {'Hung tot':>11} {'Hier tot':>11} "
              f"{'gap_tot':>14} {'gap_max':>9} "
              f"{'t_hung_ms':>11} {'t_hier_ms':>11}")
        for N in Ns:
            try:
                targets = mfn(N)
            except Exception:
                continue
            if len(targets) != N:
                continue
            seeds = list(range(seeds_per_n.get(N, 3)))
            gaps_tot, gaps_max = [], []
            hung_tot, hier_tot = [], []
            hung_times, hier_times = [], []
            any_dups = 0
            for seed in seeds:
                rng = np.random.default_rng(seed)
                starts = rng.uniform(-WORLD, WORLD, (N, 3))
                t0 = time.perf_counter()
                a_h = assign_hungarian(starts, targets)
                t_h = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                a_hier = assign_hierarchical_strict(starts, targets)
                t_hier = (time.perf_counter() - t0) * 1000
                if len(np.unique(a_hier)) != N:
                    any_dups += 1
                c_h = cost_sum(starts, targets, a_h)
                c_hier = cost_sum(starts, targets, a_hier)
                m_h = cost_max(starts, targets, a_h)
                m_hier = cost_max(starts, targets, a_hier)
                gaps_tot.append(c_hier / c_h - 1.0)
                gaps_max.append(m_hier / m_h - 1.0)
                hung_tot.append(c_h)
                hier_tot.append(c_hier)
                hung_times.append(t_h)
                hier_times.append(t_hier)
            gt = np.mean(gaps_tot) * 100
            gt_lo, gt_hi = bootstrap_ci(np.array(gaps_tot) * 100)
            gm = np.mean(gaps_max) * 100
            flag = "" if any_dups == 0 else f"  (dup-runs: {any_dups})"
            print(f"{N:>6} {len(seeds):>5} {np.mean(hung_tot):>11.1f} "
                  f"{np.mean(hier_tot):>11.1f} "
                  f"{gt:>+6.2f}%[{gt_lo:>+5.2f},{gt_hi:>+5.2f}]   "
                  f"{gm:>+6.2f}%   "
                  f"{np.mean(hung_times):>9.2f} "
                  f"{np.mean(hier_times):>9.2f}{flag}")


if __name__ == '__main__':
    main()
    main_scaling()
    main_scaling(
        manifolds=[
            ('sphere', make_sphere),
            ('torus', make_torus),
            ('cube', make_cube_shell),
            ('star', make_star_3d),
        ],
        Ns=[100, 1000],
    )
