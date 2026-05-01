# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""How much does greedy sequential patch lose vs Hungarian-optimal cluster
recovery?

Setup: K dead leaves (positions in ℝ³) and S ≥ K live surplus drones.
Each surplus must be assigned to one dead leaf. Total flight cost is
the sum of ‖surplus_i - assigned_leaf_i‖.

Greedy: process deaths in some order; for each, pick the closest live
surplus and consume it. The result depends on processing order; we
test both forward (anchor first) and reverse, and report the worse.

Hungarian: bipartite-matching optimum via scipy.optimize.linear_sum_assignment.
This is the minimum total cost achievable.

The test sweeps cluster sizes and surplus configurations. We compare:
  - "Random" surplus: surplus drones at uniform random positions
  - "Shadow" surplus: surplus drones near a designated key region
  - "Uniform-on-manifold": surplus at parent-centroid-like interior points

For each, K dead leaves are sampled either at random or near a chosen
anchor (cluster). The optimality gap is reported as
(greedy_cost - hungarian_cost) / hungarian_cost.

Lemma 8 in PROOFS.md establishes that greedy is locally optimal but not
globally; this script measures how big the gap actually is.
"""

import os
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


N_TARGETS = 100
WORLD_SCALE = 15.0  # manifold radius
SURPLUS_SCALE = 10.0  # surplus drones in a smaller region near origin
SEEDS = list(range(20))


def make_sphere(n, radius=15, center=(0, 0, 0)):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    cx, cy, cz = center
    return np.column_stack((
        radius * np.cos(theta) * np.sin(phi) + cx,
        radius * np.sin(theta) * np.sin(phi) + cy,
        radius * np.cos(phi) + cz,
    ))


def greedy_assign(dead_leaves, surplus_pos, order='forward'):
    """Greedy sequential: each death picks closest live surplus."""
    K = len(dead_leaves)
    S = len(surplus_pos)
    used = np.zeros(S, dtype=bool)
    total = 0.0
    if order == 'forward':
        idx = list(range(K))
    elif order == 'reverse':
        idx = list(range(K - 1, -1, -1))
    else:
        idx = order  # custom order
    for k in idx:
        leaf = dead_leaves[k]
        dists = np.linalg.norm(surplus_pos - leaf, axis=1)
        dists[used] = np.inf
        s = int(np.argmin(dists))
        total += float(dists[s])
        used[s] = True
    return total


def hungarian_assign(dead_leaves, surplus_pos):
    """Optimal bipartite matching of K dead leaves to S surplus."""
    K = len(dead_leaves)
    S = len(surplus_pos)
    # Cost matrix: rows = deaths, cols = surplus
    cost = np.linalg.norm(
        dead_leaves[:, None] - surplus_pos[None, :], axis=-1)
    # If S > K, scipy handles rectangular by default
    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum())
    return total


def random_cluster(targets, K, anchor_idx=None, rng=None):
    """Pick K leaf indices forming a cluster anchored at anchor_idx.
    If anchor_idx is None, pick random K leaves."""
    rng = rng or np.random.default_rng(0)
    if anchor_idx is None:
        return rng.choice(len(targets), size=K, replace=False)
    dists = np.linalg.norm(targets - targets[anchor_idx], axis=1)
    order = np.argsort(dists)
    return order[:K]


def make_random_surplus(rng, S, scale):
    return rng.uniform(-scale, scale, (S, 3))


def make_shadow_surplus(targets, key_count, S, offset=2.0):
    """Surplus drones at offset-inward positions of the first key_count
    leaves. If S > key_count, some shadow positions get multi-occupancy
    (perturb slightly)."""
    keys = targets[:key_count]
    center = targets.mean(axis=0)
    radial = keys - center
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    shadow = keys - offset * (radial / norms)
    if S <= key_count:
        return shadow[:S]
    # Replicate and perturb
    rng = np.random.default_rng(0)
    extras = S - key_count
    base = rng.choice(key_count, size=extras)
    extra_positions = shadow[base] + rng.normal(0, 0.5, (extras, 3))
    return np.vstack([shadow, extra_positions])


def run_experiment():
    targets = make_sphere(N_TARGETS, radius=WORLD_SCALE)

    cluster_sizes = [1, 3, 5, 10, 20]
    cluster_modes = ['random', 'spatial']  # random leaves vs nearest-N to anchor
    surplus_modes = [
        ('random_uniform', lambda rng, S: make_random_surplus(rng, S, WORLD_SCALE)),
        ('shadow_keys',    lambda rng, S: make_shadow_surplus(targets, 10, S)),
    ]

    print(f"Patch optimality: {N_TARGETS} target leaves on a sphere "
          f"(radius {WORLD_SCALE}), {len(SEEDS)} seeds\n")

    print(f"{'cluster_mode':<10} {'K':>3} {'surplus':<18} {'S':>4} "
          f"{'greedy_cost':>12} {'hung_cost':>12} {'gap':>8}")

    for cmode in cluster_modes:
        for K in cluster_sizes:
            for sname, sfn in surplus_modes:
                S = max(K, 15)  # ensure S >= K
                gaps_fwd = []
                gaps_rev = []
                gaps_max = []
                greedy_costs_fwd = []
                hung_costs = []
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    if cmode == 'random':
                        idx = random_cluster(targets, K, rng=rng)
                    else:
                        anchor = rng.integers(0, N_TARGETS)
                        idx = random_cluster(targets, K, anchor_idx=int(anchor), rng=rng)
                    dead_leaves = targets[idx]
                    surplus = sfn(rng, S)
                    g_fwd = greedy_assign(dead_leaves, surplus, 'forward')
                    g_rev = greedy_assign(dead_leaves, surplus, 'reverse')
                    h = hungarian_assign(dead_leaves, surplus)
                    if h > 0:
                        gaps_fwd.append(g_fwd / h - 1.0)
                        gaps_rev.append(g_rev / h - 1.0)
                        gaps_max.append(max(g_fwd, g_rev) / h - 1.0)
                        greedy_costs_fwd.append(g_fwd)
                        hung_costs.append(h)
                if greedy_costs_fwd:
                    g_avg = float(np.mean(greedy_costs_fwd))
                    h_avg = float(np.mean(hung_costs))
                    gap_avg = 100 * float(np.mean(gaps_fwd))
                    print(f"{cmode:<10} {K:>3} {sname:<18} {S:>4} "
                          f"{g_avg:>11.2f}  {h_avg:>11.2f}  {gap_avg:>+6.2f}%")
        print()


if __name__ == '__main__':
    run_experiment()
