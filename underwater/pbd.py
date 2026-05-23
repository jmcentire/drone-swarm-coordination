# /// script
# dependencies = ["numpy<3"]
# ///
"""Position-Based Dynamics for a drone-swarm lattice.

Each drone is a point mass; each lattice edge is a distance constraint
(stretch). Constraints are iteratively projected to enforce neighbor
spacing ~= k after each integration step.

Algorithm (per timestep):
  1. predicted_pos = pos + dt * velocity + dt^2 * accel
  2. for _ in range(iters): project_distance_constraints(predicted_pos)
  3. velocity = (predicted_pos - pos) / dt * damping
  4. pos = predicted_pos

Stretch compliance (compliance_xpbd >= 0):
  0   -> infinitely stiff (positions snap to rest length)
  >0  -> softer, allows some deformation (XPBD formulation)

The XPBD variant prevents iteration-count-dependent stiffness, so
behavior is stable across solver-iter changes.
"""

from __future__ import annotations

import numpy as np


class PBDLattice:
    """Position-based dynamics for a lattice of point masses with edge
    distance constraints.

    Args:
        positions: (N, 3) initial positions.
        edges: list of (i, j) index pairs with rest distance.
        rest_length: scalar rest length for all edges (typically the
            lattice spacing).
        compliance: XPBD compliance. 0 = infinitely stiff; ~1e-4 is a
            reasonable soft default.
        damping: per-step velocity multiplier (0 to 1). 0.99 leaves
            most energy; 0.5 strongly damped.
    """

    def __init__(
        self,
        positions: np.ndarray,
        edges: list[tuple[int, int]],
        rest_length: float,
        compliance: float = 0.0,
        damping: float = 0.98,
    ) -> None:
        self.pos = positions.astype(np.float64).copy()
        self.vel = np.zeros_like(self.pos)
        self.edges = np.asarray(edges, dtype=np.int64)
        self.rest_length = float(rest_length)
        self.compliance = float(compliance)
        self.damping = float(damping)

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    def step(
        self,
        dt: float,
        external_force: np.ndarray | None = None,
        iters: int = 8,
        alive: np.ndarray | None = None,
        locked: np.ndarray | None = None,
        edge_mask: np.ndarray | None = None,
    ) -> None:
        """Advance one timestep.

        external_force: (N, 3) acceleration array (per-drone) or None.
        iters: number of constraint-projection iterations.
        alive: (N,) bool array. Dead drones don't move and don't anchor
            constraints.
        locked: (N,) bool array. Locked drones don't move (zero inverse
            mass) but STILL anchor constraints -- their partner absorbs
            the full correction. This is the original work's rigid-
            scaffold trick: once a drone reaches its target it locks,
            providing an immovable reference for its neighbors.
        """
        if alive is None:
            alive = np.ones(self.n, dtype=bool)
        if locked is None:
            locked = np.zeros(self.n, dtype=bool)

        # 1) Predict positions; dead and locked drones don't propose motion.
        accel = external_force if external_force is not None else 0.0
        pred = self.pos + dt * self.vel + (dt * dt) * accel  # type: ignore
        pred[~alive] = self.pos[~alive]
        pred[locked] = self.pos[locked]

        # 2) Iteratively project distance constraints with mass weighting.
        # Inverse-mass weights: 0 = immovable, 1 = unit mass.
        # edge_mask: optional bool array per edge; False edges are skipped
        # this tick (used to drop cross-partition edges during bifurcation).
        inv_mass = np.where(alive & ~locked, 1.0, 0.0)
        alpha = self.compliance / (dt * dt)
        active_edges = self.edges
        if edge_mask is not None:
            active_edges = self.edges[edge_mask]
        for _ in range(iters):
            _project_distance(
                pred, active_edges, self.rest_length, alpha, alive, inv_mass
            )

        # 3) Velocity update
        new_vel = (pred - self.pos) / dt
        new_vel *= self.damping
        new_vel[~alive] = 0.0
        new_vel[locked] = 0.0
        self.vel = new_vel
        self.pos = pred

    def edge_strain(self) -> np.ndarray:
        """Per-edge fractional deviation from rest length: (d - L) / L."""
        a = self.pos[self.edges[:, 0]]
        b = self.pos[self.edges[:, 1]]
        d = np.linalg.norm(a - b, axis=1)
        return (d - self.rest_length) / self.rest_length


def _project_distance(
    pred: np.ndarray,
    edges: np.ndarray,
    rest: float,
    alpha: float,
    alive: np.ndarray,
    inv_mass: np.ndarray | None = None,
) -> None:
    """Apply one Gauss-Seidel pass of XPBD distance projection.

    inv_mass: (N,) inverse-mass weights. 0 = immovable (locked or dead);
        the movable endpoint absorbs the full correction. If None, both
        endpoints are treated as unit mass (legacy behaviour).
    """
    for k in range(edges.shape[0]):
        i, j = int(edges[k, 0]), int(edges[k, 1])
        if not (alive[i] and alive[j]):
            continue
        if inv_mass is not None:
            w_i = inv_mass[i]
            w_j = inv_mass[j]
        else:
            w_i = 1.0
            w_j = 1.0
        if w_i + w_j == 0:
            continue
        delta = pred[i] - pred[j]
        d = float(np.linalg.norm(delta))
        if d < 1e-12:
            continue
        C = d - rest
        denom = w_i + w_j + alpha
        s = C / (d * denom)
        corr = s * delta
        pred[i] -= w_i * corr
        pred[j] += w_j * corr


def edges_from_neighbors(neighbors: list[list[int]]) -> list[tuple[int, int]]:
    """Convert adjacency list to unique edge list (i < j)."""
    seen: set[tuple[int, int]] = set()
    for i, ns in enumerate(neighbors):
        for j in ns:
            if i < j:
                seen.add((i, j))
    return sorted(seen)


if __name__ == "__main__":
    # Smoke test: perturb a small HCP lattice, confirm PBD pulls it back.
    from lattice import (
        build_neighbor_graph,
        degree_distribution,
        hcp_positions,
    )

    spacing = 10.0
    pos = hcp_positions(hex_radius=2, n_layers=3, spacing=spacing)
    rng = np.random.default_rng(0)
    perturbed = pos + rng.normal(scale=0.3 * spacing, size=pos.shape)
    neighbors = build_neighbor_graph(pos, comms_range=spacing * 1.1)
    edges = edges_from_neighbors(neighbors)

    sim = PBDLattice(
        perturbed, edges, rest_length=spacing, compliance=0.0, damping=0.9
    )
    print(f"initial mean |strain|: {np.mean(np.abs(sim.edge_strain())):.4f}")
    for t in range(200):
        sim.step(dt=0.05, iters=8)
    print(f"final   mean |strain|: {np.mean(np.abs(sim.edge_strain())):.4f}")
    print(
        f"final degree (rebuilt): "
        f"min={degree_distribution(build_neighbor_graph(sim.pos, spacing*1.1)).min()}"
    )
