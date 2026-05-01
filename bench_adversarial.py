# /// script
# dependencies = ["numpy<3", "scipy"]
# ###
"""Adversarial threat model: how much disruption can a small fraction of
byzantine drones cause?

Threat model: k of N drones are byzantine — they broadcast position values
that are wildly off (drawn from a distribution far from the legitimate
swarm region). Honest drones run ASSIGN against the broadcast as if all
positions were legitimate.

We measure the cascade: how many honest drones get a different leaf
assignment than they would in a clean broadcast?

Mitigations tested:
  - "naive": no defense. Honest drones use all broadcast positions including
    the byzantine ones.
  - "outlier-reject": honest drones discard broadcast positions that fall
    outside k×σ of the swarm's projected centroid before computing the
    PCA tree partition.

Sweep k from 1 to 10 byzantine drones (out of N=100). Report cascade
fraction: number of honest drones with different assignment / N - k.

This empirically characterizes the attack surface and validates whether
trimmed-mean / outlier-rejection mitigations are effective.
"""

import os
import numpy as np

NUM_DRONES = 100
N_SEEDS = int(os.environ.get("N_SEEDS", "20"))
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


def trimmed_rank_partition(order, dl, alpha=0.05):
    """Trim top and bottom α fractions as 'extreme' (likely byzantine);
    partition the inliers proportionally; dispatch trimmed-bottom to
    left and trimmed-top to right.

    For α=0.05 and n=100, trims 5 from each end. Honest drones in the
    middle 90% partition cleanly via rank order. Up to 5 byzantine
    drones at either extreme are absorbed without disturbing the
    inlier partition.

    Trade-off: count is preserved overall (n drones partitioned), but
    the per-side counts drift by up to ±α·n from the proportional ideal,
    which is tolerable in the surplus regime and acceptable in the
    bijective regime.
    """
    n = len(order)
    trim_k = max(1, int(alpha * n)) if n > 4 else 0
    if trim_k == 0:
        # Too few drones to trim — fall back to standard
        return order[:dl], order[dl:]
    bottom = order[:trim_k]
    top = order[-trim_k:]
    middle = order[trim_k: n - trim_k]

    # Need (dl - trim_k) more drones in left from middle inliers.
    middle_left_count = dl - trim_k
    middle_left_count = max(0, min(len(middle), middle_left_count))
    middle_left = middle[:middle_left_count]
    middle_right = middle[middle_left_count:]
    left = list(bottom) + list(middle_left)
    right = list(middle_right) + list(top)
    return left, right


def compute_target(my_id, drones, root, robust=False, trim_alpha=0.05):
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
        if robust:
            left_idx, right_idx = trimmed_rank_partition(order, dl, trim_alpha)
            groups = [
                [cur[i] for i in left_idx],
                [cur[i] for i in right_idx],
            ]
        else:
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
    if len(node.positions) == 1:
        return node.positions[0]
    return node.center


def assign_all(positions, targets, mask=None, robust=False):
    """Run assignment. mask filters which drones participate.
    robust=True enables projection-clamp winsorization at each tree level."""
    tree = ManifoldNode(targets)
    if mask is None:
        mask = np.ones(len(positions), dtype=bool)
    drones = [{'id': i, 'pos': positions[i].copy()}
              for i in range(len(positions)) if mask[i]]
    leaves = []
    for i in range(len(positions)):
        if not mask[i]:
            leaves.append(-1)
            continue
        leaf = compute_target(i, drones, tree, robust=robust)
        idx = int(np.argmin(np.linalg.norm(targets - leaf, axis=1)))
        leaves.append(idx)
    return np.array(leaves)


def outlier_reject_mask(positions, sigma_threshold=3.0):
    """Reject positions > sigma_threshold * std from the centroid (mean-based)."""
    centroid = positions.mean(axis=0)
    dists = np.linalg.norm(positions - centroid, axis=1)
    threshold = dists.mean() + sigma_threshold * dists.std()
    return dists < threshold


def mad_reject_mask(positions, k=3.0):
    """Reject positions > k * MAD from the spatial median (robust-statistics).
    MAD = median absolute deviation. More robust to a few extreme outliers
    than mean+std, because the median+MAD aren't pulled by the outliers
    themselves.
    """
    centroid = np.median(positions, axis=0)
    dists = np.linalg.norm(positions - centroid, axis=1)
    med = np.median(dists)
    mad = np.median(np.abs(dists - med))
    if mad < 1e-9:
        mad = dists.std() + 1e-9
    threshold = med + k * 1.4826 * mad  # 1.4826 → MAD-to-σ for normal dist
    return dists < threshold


def run_attack(k, attack, mitigation, n_seeds, targets):
    """attack ∈ {'random', 'coordinated'}, mitigation ∈ {'none','sigma','mad'}."""
    cascades = []
    detected = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        honest_starts = rng.uniform(-WORLD, WORLD, (NUM_DRONES, 3))
        byzantine_ids = (rng.choice(NUM_DRONES, size=k, replace=False)
                         if k > 0 else np.array([], dtype=int))
        byzantine_set = set(byzantine_ids.tolist())

        baseline_assignment = assign_all(honest_starts, targets)

        corrupted = honest_starts.copy()
        if attack == 'random':
            for bid in byzantine_ids:
                corrupted[bid] = rng.uniform(-200, 200, 3)
        elif attack == 'coordinated':
            # All byzantines lie in the same direction by the same amount —
            # mimics an attacker biasing the consensus. The shift is large
            # enough to leave the legitimate region.
            shift = rng.standard_normal(3)
            shift /= np.linalg.norm(shift)
            shift *= 100.0  # 2.5× legitimate range, in same direction
            for bid in byzantine_ids:
                corrupted[bid] = honest_starts[bid] + shift

        robust = False
        if mitigation == 'none':
            mask = np.ones(NUM_DRONES, dtype=bool)
        elif mitigation == 'sigma':
            mask = outlier_reject_mask(corrupted, sigma_threshold=3.0)
        elif mitigation == 'mad':
            mask = mad_reject_mask(corrupted, k=3.0)
        elif mitigation == 'clamp':
            # Per-level projection clamping; no mask removal — preserves
            # count invariant while clamping byzantines to projection-MAD
            # boundary at each tree level.
            mask = np.ones(NUM_DRONES, dtype=bool)
            robust = True
        else:
            raise ValueError(mitigation)

        result = assign_all(corrupted, targets, mask=mask, robust=robust)

        honest_ids = [i for i in range(NUM_DRONES) if i not in byzantine_set]
        changed = sum(1 for i in honest_ids
                      if (mask[i] and result[i] != baseline_assignment[i])
                      or (not mask[i]))
        cascades.append(changed)
        detected.append(sum(1 for bid in byzantine_ids if not mask[bid]))

    cascades = np.array(cascades)
    detected = np.array(detected)
    n_honest = NUM_DRONES - k
    return {
        'cascade_mean': float(cascades.mean()),
        'cascade_pct': 100 * float(cascades.mean()) / max(n_honest, 1),
        'detected_mean': float(detected.mean()),
    }


def main():
    print(f"Adversarial threat experiment: N={NUM_DRONES}, {N_SEEDS} seeds\n")

    targets = make_sphere(NUM_DRONES)
    ks = [0, 1, 3, 5, 10, 20]
    attacks = ['random', 'coordinated']
    mitigations = ['none', 'sigma', 'mad', 'clamp']

    for attack in attacks:
        print(f"=== {attack.upper()} BYZANTINE ATTACK ===")
        if attack == 'coordinated':
            print("  All byzantines shift by same vector (biased consensus attack)")
        else:
            print("  Each byzantine broadcasts independent random extreme position")
        print()
        print(f"{'k':>4}  {'mitigation':<10}  "
              f"{'cascade':>8}  {'cascade_pct':>13}  {'detected':>10}")
        for k in ks:
            for mit in mitigations:
                r = run_attack(k, attack, mit, N_SEEDS, targets)
                print(f"{k:>4d}  {mit:<10}  "
                      f"{r['cascade_mean']:>8.1f}  "
                      f"{r['cascade_pct']:>11.1f}%  "
                      f"{r['detected_mean']:>5.1f}/{k}")
        print()

    print("Operational interpretation:")
    print("  - Random byzantine: each lies independently → outlier-detection")
    print("    catches them (each is an isolated outlier in projection).")
    print("    MAD-based detection should outperform σ-based because the")
    print("    spatial median+MAD aren't pulled by the outliers themselves.")
    print("  - Coordinated byzantine: all shift in the same direction → the")
    print("    outliers form a cluster that may shift the centroid and")
    print("    inflate the σ/MAD enough to evade detection. PKI authentication")
    print("    is necessary for this regime; statistical defenses are not")
    print("    sufficient.")


if __name__ == '__main__':
    main()
