# /// script
# dependencies = ["numpy<3"]
# ///
"""Manifold tree + per-drone divide-and-conquer assignment.

Lifted verbatim from the validated above-water work (simulator.py and
bench_attrition.py). compute_target() runs PER DRONE; each drone runs
the same code on its locally-known drone set and produces its own
assignment. Consensus is by determinism — same input, same output —
NOT by gossip-of-the-decision. This module has no global state.

The locally-known drone set is the only input. If two drones have
divergent local sets (because gossip hasn't fully converged or because
they're partitioned), they will compute different answers. That
divergence is the protocol's actual behavior under those conditions
and MUST be measured rather than hidden.
"""

from __future__ import annotations

import numpy as np


class ManifoldNode:
    """Binary tree decomposition of a target manifold via PCA splits.

    Pure geometry — depends only on the input target positions. Two
    drones with the same target list will build identical trees by
    deterministic SVD.
    """

    def __init__(self, positions: np.ndarray, depth: int = 0) -> None:
        self.positions = np.array(positions, dtype=np.float64)
        self.center = np.mean(self.positions, axis=0)
        self.depth = depth
        self.split_axis: np.ndarray | None = None
        self.left: "ManifoldNode | None" = None
        self.right: "ManifoldNode | None" = None
        if len(self.positions) > 1:
            self._split()

    def _split(self) -> None:
        centered = self.positions - self.center
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        self.split_axis = Vt[0]
        proj = centered @ self.split_axis
        order = np.argsort(proj, kind="stable")
        mid = len(order) // 2
        self.left = ManifoldNode(self.positions[order[:mid]], self.depth + 1)
        self.right = ManifoldNode(self.positions[order[mid:]], self.depth + 1)


def compute_target(
    my_id: int,
    drones: list[dict],
    root: ManifoldNode,
) -> tuple[np.ndarray, bool]:
    """Run the divide-and-conquer assignment for one drone.

    drones: [{'id': int, 'pos': np.ndarray(3,)}, ...] — the LOCALLY-KNOWN
            drone set as far as this drone can tell. NOT a global set.
    root:   ManifoldNode tree built from the target positions.

    Returns (target_position, is_primary_at_leaf).
        is_primary_at_leaf == True  -> this drone holds the leaf.
        is_primary_at_leaf == False -> this drone is surplus, target is
                                       a parent-centroid (interior).
    """
    node = root
    parent = root
    cur = list(drones)
    my_pos = None
    for d in drones:
        if d["id"] == my_id:
            my_pos = np.asarray(d["pos"], dtype=np.float64)
            break
    if my_pos is None:
        # Caller is not in the drone list. Pathological; return the root center.
        return root.center.copy(), False

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
        positions = np.array([d["pos"] for d in cur])
        proj = positions @ node.split_axis  # type: ignore[operator]
        order = np.argsort(proj, kind="stable")
        groups = [
            [cur[order[i]] for i in range(dl)],
            [cur[order[i]] for i in range(dl, n)],
        ]
        subs = [node.left, node.right]
        for i, group in enumerate(groups):
            if any(d["id"] == my_id for d in group):
                parent = node
                node = subs[i]
                cur = group
                break

    # Single-drone subtree may still have multi-leaf node; descend by nearest.
    while node.left is not None:
        parent = node
        dl_ = float(np.linalg.norm(my_pos - node.left.center))
        dr_ = float(np.linalg.norm(my_pos - node.right.center))
        node = node.left if dl_ <= dr_ else node.right

    leaf_pos = node.positions[0] if len(node.positions) == 1 else node.center

    if len(cur) == 1:
        return leaf_pos.copy(), True

    distances = sorted(
        (float(np.linalg.norm(np.array(d["pos"]) - leaf_pos)), d["id"])
        for d in cur
    )
    primary_id = distances[0][1]
    if my_id == primary_id:
        return leaf_pos.copy(), True
    return parent.center.copy(), False


# Sanity tests run only when executed directly.
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(20, 3)) * 5
    tree = ManifoldNode(targets)
    starts = rng.normal(size=(20, 3)) * 10
    drones = [{"id": i, "pos": starts[i].copy()} for i in range(20)]

    # Each drone independently computes its target.
    assignments = []
    for i in range(20):
        t, primary = compute_target(i, drones, tree)
        assignments.append((i, t, primary))

    # Sanity: all 20 leaves should be claimed by exactly one drone.
    leaf_claims = {}
    for i, t, primary in assignments:
        if primary:
            key = tuple(np.round(t, 6))
            leaf_claims.setdefault(key, []).append(i)
    n_unique_leaves = len(leaf_claims)
    n_contested = sum(1 for v in leaf_claims.values() if len(v) > 1)
    print(f"assigned {n_unique_leaves} distinct leaves; {n_contested} contested")

    # Falsifiability check: if I corrupt one drone's position in only ITS
    # local view, its assignment should differ from what the others see for it.
    drones_corrupted = [dict(d) for d in drones]
    drones_corrupted[5]["pos"] = np.array([100.0, 100.0, 100.0])
    t_self, _ = compute_target(5, drones_corrupted, tree)
    t_others, _ = compute_target(5, drones, tree)
    diff = float(np.linalg.norm(t_self - t_others))
    print(f"divergent-local-view test: drone 5 self-target vs others' view differs by {diff:.3f}m (expected non-zero)")
