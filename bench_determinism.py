# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""FP determinism stress test for the assignment algorithm.

Theorem 1's "consensus by determinism" assumes every drone running ASSIGN
on the same broadcast input arrives at the same global assignment. The
implicit precondition is bit-identical floating-point arithmetic across
drones. In real heterogeneous-hardware deployment this is not free: SVD
implementations differ across BLAS libraries, SIMD widths, FMA
availability, compiler optimizations.

This script measures cross-implementation disagreement empirically by
running ASSIGN against the same broadcast under five conditions:

  1. Baseline: numpy.linalg.svd (LAPACK driver)
  2. SciPy: scipy.linalg.svd (different LAPACK routine)
  3. Order-perturbed: input rows shuffled then unshuffled (tests SVD's
     sensitivity to input ordering — a real cross-platform issue)
  4. Tiny perturbation: positions shifted by 1e-12 (machine-epsilon)
  5. Small perturbation: positions shifted by 1e-9 (sub-millimeter)

For each condition we count: (a) drones whose tree-path differs from
baseline, (b) drones whose final leaf differs from baseline, (c) Hamming
distance from baseline assignment.

Empirical result tells us how robust the determinism property is to
realistic numerical noise. The bound the paper actually needs:
heterogeneous hardware introduces ~1e-15 relative differences in SVD
output; if those propagate to >0% of drones disagreeing, then
deployment requires deterministic-SVD libraries or fixed-point
projection-rank computation as a precondition.
"""

import os
import numpy as np
import scipy.linalg

NUM_DRONES = int(os.environ.get("N", "100"))
N_SEEDS = int(os.environ.get("N_SEEDS", "10"))
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
    def __init__(self, positions, depth=0, svd_fn=None):
        self.positions = np.array(positions)
        self.center = np.mean(positions, axis=0)
        self.depth = depth
        self.split_axis = None
        self.left = self.right = None
        self.svd_fn = svd_fn or (lambda c: np.linalg.svd(c, full_matrices=False))
        if len(positions) > 1:
            self._split()

    def _split(self):
        c = self.positions - self.center
        _, _, Vt = self.svd_fn(c)
        self.split_axis = Vt[0]
        proj = c @ self.split_axis
        order = np.argsort(proj, kind='stable')
        mid = len(order) // 2
        self.left = ManifoldNode(self.positions[order[:mid]], self.depth + 1, self.svd_fn)
        self.right = ManifoldNode(self.positions[order[mid:]], self.depth + 1, self.svd_fn)


def compute_target(my_id, drones, root):
    """Strict-mode hierarchical assignment."""
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
    if len(node.positions) == 1:
        return node.positions[0]
    return node.center


def assign_all(starts, targets, svd_fn=None):
    """Run full assignment under specified SVD implementation."""
    tree = ManifoldNode(targets, svd_fn=svd_fn)
    drones = [{'id': i, 'pos': starts[i].copy()} for i in range(len(starts))]
    leaves = []
    for i in range(len(starts)):
        leaf = compute_target(i, drones, tree)
        # Map leaf to target index
        idx = int(np.argmin(np.linalg.norm(targets - leaf, axis=1)))
        leaves.append(idx)
    return np.array(leaves)


def main():
    print(f"FP determinism stress test: N={NUM_DRONES}, {N_SEEDS} seeds\n")

    # SVD implementations to test
    def svd_numpy(c):
        return np.linalg.svd(c, full_matrices=False)

    def svd_scipy(c):
        # scipy returns U, s, Vh — same convention as numpy
        return scipy.linalg.svd(c, full_matrices=False)

    def svd_numpy_perturbed(c):
        # Add 1e-12 noise before SVD
        c2 = c + np.random.default_rng(0).standard_normal(c.shape) * 1e-12
        return np.linalg.svd(c2, full_matrices=False)

    targets = make_sphere(NUM_DRONES)

    # Disagreement metrics
    metrics = {
        'numpy vs scipy': [],
        'baseline vs 1e-12 input perturbation': [],
        'baseline vs 1e-9 input perturbation': [],
        'baseline vs row-shuffled input': [],
        'baseline vs SVD-internal 1e-12 noise': [],
    }

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        starts = rng.uniform(-WORLD, WORLD, (NUM_DRONES, 3))

        # Baseline
        baseline = assign_all(starts, targets, svd_fn=svd_numpy)

        # numpy vs scipy
        scipy_result = assign_all(starts, targets, svd_fn=svd_scipy)
        diff = int(np.sum(baseline != scipy_result))
        metrics['numpy vs scipy'].append(diff)

        # 1e-12 input perturbation
        starts_p1 = starts + rng.standard_normal(starts.shape) * 1e-12
        result_p1 = assign_all(starts_p1, targets, svd_fn=svd_numpy)
        metrics['baseline vs 1e-12 input perturbation'].append(int(np.sum(baseline != result_p1)))

        # 1e-9 input perturbation
        starts_p2 = starts + rng.standard_normal(starts.shape) * 1e-9
        result_p2 = assign_all(starts_p2, targets, svd_fn=svd_numpy)
        metrics['baseline vs 1e-9 input perturbation'].append(int(np.sum(baseline != result_p2)))

        # Row-shuffled input (permutation of drone order — should be invariant)
        # The strict algorithm uses drone IDs not row order, so different drone IDs map to same role.
        # Here we test: same drone positions, different row-order in initial broadcast.
        # In our implementation, drone IDs are the iteration index, so shuffling rows changes IDs.
        # We test: does the final ID-to-leaf mapping match if we shuffle then unshuffle?
        perm = rng.permutation(NUM_DRONES)
        starts_perm = starts[perm]
        result_perm = assign_all(starts_perm, targets, svd_fn=svd_numpy)
        # Map back: result_perm[i] is the leaf for original drone perm[i]
        result_unperm = np.zeros(NUM_DRONES, dtype=int)
        for i in range(NUM_DRONES):
            result_unperm[perm[i]] = result_perm[i]
        metrics['baseline vs row-shuffled input'].append(int(np.sum(baseline != result_unperm)))

        # SVD-internal noise
        result_svd_noise = assign_all(starts, targets, svd_fn=svd_numpy_perturbed)
        metrics['baseline vs SVD-internal 1e-12 noise'].append(int(np.sum(baseline != result_svd_noise)))

    print(f"{'condition':<48} {'mean':>8} {'max':>6} {'fraction':>10}")
    for k, v in metrics.items():
        v = np.array(v)
        print(f"{k:<48} {v.mean():>8.2f} {v.max():>6} {v.mean()/NUM_DRONES*100:>8.2f}%")

    # Sweep perturbation magnitudes to find the breakeven threshold
    print()
    print("Perturbation-magnitude sweep (where does disagreement begin?):")
    print(f"{'noise σ':>10} {'mean disagreement':>20} {'fraction':>10}")
    sweep_levels = [1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1.0]
    for sigma in sweep_levels:
        diffs = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            starts = rng.uniform(-WORLD, WORLD, (NUM_DRONES, 3))
            baseline = assign_all(starts, targets)
            starts_p = starts + rng.standard_normal(starts.shape) * sigma
            result = assign_all(starts_p, targets)
            diffs.append(int(np.sum(baseline != result)))
        diffs = np.array(diffs)
        print(f"{sigma:>10.1e} {diffs.mean():>20.2f} {diffs.mean()/NUM_DRONES*100:>9.2f}%")

    print()
    print("Interpretation:")
    print("  - 'mean' is the average number of drones whose assigned leaf")
    print("    differs from the baseline across N_SEEDS runs.")
    print("  - 'max' is the worst case (largest disagreement) over seeds.")
    print("  - 'fraction' is mean / N (% of swarm disagreeing on average).")
    print()
    print("Operational implication: any non-zero disagreement means")
    print("Theorem 1's determinism precondition is not free; deployment")
    print("on heterogeneous hardware requires either (a) bit-deterministic")
    print("SVD+sort libraries with strict FP modes, or (b) fixed-point")
    print("arithmetic for the projection-rank computation, or (c) a")
    print("reconciliation protocol when drones detect disagreement.")


if __name__ == '__main__':
    main()
