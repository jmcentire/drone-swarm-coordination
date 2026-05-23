# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Mission cycle FSM with convergence-based transitions.

The cycle: MOVE -> SETTLE -> REFORM -> MOVE -> ...

Phase transitions are convergence-gated -- a phase ends when the relevant
state has stopped changing, not when a timer expires. Same broadcast-
quiescence trick from the original divide-and-conquer simulator.

Loss recovery happens inside REFORM. There is no special panic phase --
REFORM simply runs until the topology invariant is restored and every
drone has reached its computed slot, however long that takes.

Per-drone isolation recovery is a subroutine, not a phase: a drone with
zero comms-range neighbors for K seconds overrides its phase-based force
with an attractor toward the last known swarm centroid until it picks up
neighbors again. The rejoined drone is then absorbed by whatever REFORM
or MOVE the swarm is in -- no global mode change required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from lattice import hcp_positions


# ---------------------------------------------------------------------------
# Recursive bisection target assignment (ported from the original
# drone_swarm work -- ManifoldNode + compute_leaf_target in simulator.py).
# Each drone independently descends the manifold tree, partitioning the
# drone set by spatial projection at every node. Local-recursive: losing
# a drone only reshuffles within its own subtree, not globally. Surplus
# drones (more drones than leaves in a subtree) go to the parent centroid.
# ---------------------------------------------------------------------------


class ManifoldNode:
    """Binary tree decomposition of a target manifold via PCA splits."""

    def __init__(self, positions: np.ndarray, depth: int = 0) -> None:
        self.positions = np.array(positions)
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


def compute_leaf_target(
    my_id: int, drones: list[dict], root: ManifoldNode
) -> np.ndarray | None:
    """Recursive divide-and-conquer assignment. Returns my target position.

    drones: [{'id': int, 'pos': (3,) ndarray}, ...] -- the live drone set.
    """
    node = root
    parent = root
    cur = list(drones)
    my_pos = None
    for d in drones:
        if d["id"] == my_id:
            my_pos = np.array(d["pos"])
            break
    if my_pos is None:
        return None

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

    # Singleton subtree: descend to a single leaf by nearest-center.
    while node.left is not None:
        parent = node
        dl_ = float(np.linalg.norm(my_pos - node.left.center))
        dr_ = float(np.linalg.norm(my_pos - node.right.center))
        node = node.left if dl_ <= dr_ else node.right

    leaf_pos = node.positions[0] if len(node.positions) == 1 else node.center
    if len(cur) == 1:
        return leaf_pos.copy()
    # Multiple drones at this leaf -- closest takes leaf, others go to parent
    # centroid (becomes "surplus" at an interior position).
    distances = sorted(
        (float(np.linalg.norm(np.array(d["pos"]) - leaf_pos)), d["id"])
        for d in cur
    )
    primary_id = distances[0][1]
    if my_id == primary_id:
        return leaf_pos.copy()
    return parent.center.copy()


class Phase(Enum):
    MOVE = "move"
    SETTLE = "settle"
    REFORM = "reform"


@dataclass
class MissionConfig:
    # Leg geometry
    leg_distance_m: float = 20.0
    velocity_mps: float = 1.0
    spacing: float = 10.0
    heading_init: tuple[float, float, float] = (1.0, 0.0, 0.0)

    # Drive gains
    drive_gain: float = 4.0
    attractor_gain: float = 2.0

    # Convergence thresholds
    velocity_quiescent_mps: float = 0.05
    # Locking thresholds (matches the original work's broadcast-quiescence
    # pattern): once a drone is within target_arrival_m of its target and
    # moving below lock_velocity_mps, it locks (snaps + freezes). REFORM
    # ends when all alive drones are locked.
    target_arrival_m: float = 1.0
    lock_velocity_mps: float = 0.15
    quiescent_ticks_required: int = 20

    # Generous hard cap on phase length (only reached if convergence fails)
    max_phase_ticks: int = 6000

    # Canonical lattice shape -- the swarm reforms to a sub-patch of this same
    # shape as drones die (shrinks inward, doesn't change aspect ratio).
    canonical_hex_radius: int = 3
    canonical_n_layers: int = 3

    # Spare-drone pool. Spares drift near the swarm centroid (relaxed,
    # no PBD edges) and get promoted into vacated lattice slots during
    # REFORM, restoring local topology. n_spares = 0 disables.
    n_spares: int = 0
    spare_hover_radius_m: float = 0.0  # set from spacing if 0 at init


@dataclass
class MissionState:
    phase: Phase = Phase.MOVE
    phase_start_tick: int = 0
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    last_known_centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    leg_index: int = 0
    leg_start_centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    targets: np.ndarray | None = None
    quiescent_streak: int = 0
    # Per-drone lock state: True means the drone has reached its target
    # during REFORM and is now frozen in place. Locked drones still anchor
    # PBD constraints (immovable reference points for their neighbors) but
    # don't themselves move. Unlocked at REFORM->MOVE transition.
    locked: np.ndarray | None = None
    isolation_events: int = 0  # kept for compatibility
    # Active/Spare designation: active drones get lattice slots; spares
    # drift near the centroid until promoted to fill a vacated slot.
    is_active: np.ndarray | None = None
    n_promotions: int = 0
    # The (alive_signature, n_active_signature) snapshot under which the
    # current targets were computed. Targets refresh whenever either changes.
    targets_alive_signature: int = -1
    targets_active_signature: int = -1
    # Snapshot of is_active at the moment REFORM started this cycle. Used
    # to decide convergence: only the drones already active at REFORM entry
    # are required to lock. Newly-promoted understudies don't block the
    # cycle -- they slot in over the next leg without pausing the show.
    pre_reform_active: np.ndarray | None = None


def compute_manifold_targets(
    alive_set: np.ndarray,
    centroid: np.ndarray,
    heading: np.ndarray,
    spacing: float,
    current_positions: np.ndarray | None = None,
    canonical_hex_radius: int = 3,
    canonical_n_layers: int = 3,
    use_bisection: bool = True,
    is_active: np.ndarray | None = None,
    spare_hover_radius: float | None = None,
) -> np.ndarray:
    """Deterministic target positions for each alive drone.

    The slot set is **always a sub-patch of the canonical HCP shape**
    -- we take the n_alive atoms closest to the canonical lattice's
    centroid. As drones die the manifold shrinks inward, preserving the
    original aspect ratio. (If we re-picked the shape per call, the
    chosen aspect ratio would jitter and Hungarian's best alignment
    would point drones at wildly different slots between consecutive
    ticks.)

    Drones are assigned to slots by minimum total displacement
    (Hungarian / linear_sum_assignment). Spatially-aware assignment
    matters: arbitrarily-indexed assignment would have drones swap
    across the swarm while PBD constraints keep them locked, and
    REFORM would never converge.

    Every drone running this function on the same inputs gets the same
    answer -- Hungarian is deterministic.
    """
    n = alive_set.shape[0]
    targets = np.tile(centroid, (n, 1))
    if is_active is None:
        is_active = np.ones(n, dtype=bool)
    alive_active = alive_set & is_active
    alive_spare = alive_set & ~is_active
    active_ids = np.where(alive_active)[0]
    spare_ids = np.where(alive_spare)[0]
    n_alive_active = len(active_ids)
    if n_alive_active == 0 and len(spare_ids) == 0:
        return targets

    # The manifold is always the canonical full lattice -- losses leave
    # holes rather than shrinking the lattice. Holes are tolerated because
    # HCP redundancy (12 NN interior, ≥4 non-coplanar everywhere) absorbs
    # them. Critically, the PBD edge rest-lengths stay matched to the
    # canonical arrangement, so surviving drones don't have to fight the
    # constraints to reach their slots.
    candidate = hcp_positions(
        canonical_hex_radius, canonical_n_layers, spacing=spacing
    )
    if candidate.shape[0] < n_alive_active:
        for r in range(canonical_hex_radius, 12):
            for L in range(canonical_n_layers, 12):
                cand2 = hcp_positions(r, L, spacing=spacing)
                if cand2.shape[0] >= n_alive_active:
                    candidate = cand2
                    break
            if candidate.shape[0] >= n_alive_active:
                break

    slots = candidate - candidate.mean(axis=0)
    slots = _rotate_to_heading(slots, heading)
    slots = slots + centroid

    if current_positions is not None and n_alive_active > 0:
        # Hungarian assignment of active drones onto lattice slots.
        cost = np.linalg.norm(
            current_positions[active_ids][:, None, :] - slots[None, :, :],
            axis=-1,
        )
        row_idx, col_idx = linear_sum_assignment(cost)
        for r, c in zip(row_idx, col_idx):
            targets[active_ids[r]] = slots[c]
    elif n_alive_active > 0:
        for idx, slot in zip(sorted(active_ids), slots[:n_alive_active]):
            targets[idx] = slot

    # Spare drones hover in a ring around the centroid (above and below the
    # lattice's primary plane), staggered so they don't pile up. They have
    # no PBD edges, so a small ring is sufficient. Each spare's hover slot
    # is keyed off its drone ID modulo a stable constant, NOT its index in
    # the currently-alive spare pool -- otherwise promotions reshuffle all
    # remaining spares' hover slots and REFORM can't converge.
    if len(spare_ids) > 0:
        radius = spare_hover_radius if spare_hover_radius and spare_hover_radius > 0 else (1.5 * spacing)
        ring_slots = 32  # fixed period; supports up to ~32 distinct hover angles
        for sid in spare_ids:
            local_idx = int(sid) % ring_slots
            angle = 2 * np.pi * local_idx / ring_slots
            z_offset = ((local_idx % 3) - 1) * 0.3 * spacing
            offset = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                z_offset,
            ])
            targets[sid] = centroid + offset
    return targets


def _rotate_to_heading(points: np.ndarray, heading: np.ndarray) -> np.ndarray:
    h = heading / max(np.linalg.norm(heading), 1e-12)
    x_ax = np.array([1.0, 0.0, 0.0])
    v = np.cross(x_ax, h)
    s = np.linalg.norm(v)
    c = float(np.dot(x_ax, h))
    if s < 1e-9:
        return points if c > 0 else points @ np.diag([-1.0, -1.0, 1.0]).T
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s * s))
    return points @ R.T


def update_mission(
    state: MissionState,
    config: MissionConfig,
    tick: int,
    dt: float,
    pos: np.ndarray,
    vel: np.ndarray,
    alive: np.ndarray,
    is_violating: np.ndarray,
    current_degree: np.ndarray,
) -> None:
    """Advance the mission FSM. Convergence-gated phase transitions."""
    elapsed_ticks = tick - state.phase_start_tick

    n = pos.shape[0]
    if state.locked is None or state.locked.shape[0] != n:
        state.locked = np.zeros(n, dtype=bool)
    if state.is_active is None or state.is_active.shape[0] != n:
        # Default: all drones active. The bench sets this explicitly when
        # using spares (last config.n_spares drones flagged as spare).
        state.is_active = np.ones(n, dtype=bool)
        if config.n_spares > 0 and config.n_spares < n:
            state.is_active[-config.n_spares:] = False

    # Centroid is the active body's center -- spares orbit around it, so
    # including them in the centroid would drift it outward.
    body = alive & state.is_active
    if body.any():
        centroid = pos[body].mean(axis=0)
    elif alive.any():
        centroid = pos[alive].mean(axis=0)
    else:
        centroid = state.last_known_centroid
    state.last_known_centroid = centroid

    if state.phase == Phase.MOVE:
        displacement = float(np.linalg.norm(centroid - state.leg_start_centroid))
        if displacement >= config.leg_distance_m:
            state.phase = Phase.SETTLE
            state.phase_start_tick = tick
            state.quiescent_streak = 0

    elif state.phase == Phase.SETTLE:
        if alive.any():
            mean_speed = float(np.linalg.norm(vel[alive], axis=1).mean())
        else:
            mean_speed = 0.0
        if mean_speed < config.velocity_quiescent_mps:
            state.quiescent_streak += 1
        else:
            state.quiescent_streak = 0
        if (
            state.quiescent_streak >= config.quiescent_ticks_required
            or elapsed_ticks > config.max_phase_ticks
        ):
            state.targets = compute_manifold_targets(
                alive, centroid, state.heading, config.spacing,
                current_positions=pos,
                canonical_hex_radius=config.canonical_hex_radius,
                canonical_n_layers=config.canonical_n_layers,
                is_active=state.is_active,
                spare_hover_radius=config.spare_hover_radius_m,
            )
            # Snapshot the active set as it stands at REFORM entry. Any
            # spares promoted during this REFORM cycle are NOT included --
            # they're understudies in flight, don't block convergence.
            state.pre_reform_active = state.is_active.copy()
            state.phase = Phase.REFORM
            state.phase_start_tick = tick
            state.quiescent_streak = 0

    elif state.phase == Phase.REFORM:
        # Patch-promotion: count active drones still alive. If fewer than
        # the canonical lattice size, promote live spares to fill the gap.
        # Promoted spares get a lattice slot in the next Hungarian pass.
        canon_count = hcp_positions(
            config.canonical_hex_radius, config.canonical_n_layers,
            spacing=config.spacing,
        ).shape[0]
        n_active_alive = int((state.is_active & alive).sum())
        n_spare_alive = int((~state.is_active & alive).sum())
        gap = canon_count - n_active_alive
        if gap > 0 and n_spare_alive > 0:
            promote_count = min(gap, n_spare_alive)
            spare_pool = np.where((~state.is_active) & alive)[0]
            # Promote the spares closest to the centroid first (they are
            # the most likely to be near a vacated interior slot).
            dists = np.linalg.norm(pos[spare_pool] - centroid, axis=1)
            order = np.argsort(dists)
            promoted = spare_pool[order[:promote_count]]
            state.is_active[promoted] = True
            state.n_promotions += int(promote_count)

        # Recompute targets when alive set OR active set changes.
        alive_sig = int(alive.sum())
        active_sig = int(state.is_active.sum())
        if (
            alive_sig != state.targets_alive_signature
            or active_sig != state.targets_active_signature
            or state.targets is None
        ):
            state.targets = compute_manifold_targets(
                alive, centroid, state.heading, config.spacing,
                current_positions=pos,
                canonical_hex_radius=config.canonical_hex_radius,
                canonical_n_layers=config.canonical_n_layers,
                is_active=state.is_active,
                spare_hover_radius=config.spare_hover_radius_m,
            )
            state.targets_alive_signature = alive_sig
            state.targets_active_signature = active_sig
            state.locked[:] = False

        # Update lock state: drones within target_arrival_m of their target
        # AND moving slowly snap into the lock.
        if state.targets is not None:
            speed = np.linalg.norm(vel, axis=1)
            dist_to_target = np.linalg.norm(state.targets - pos, axis=1)
            newly_locked = (
                alive
                & ~state.locked
                & (dist_to_target < config.target_arrival_m)
                & (speed < config.lock_velocity_mps)
            )
            state.locked[newly_locked] = True

        # Convergence: only the drones that were already active when this
        # REFORM started need to lock. Newly-promoted understudies are
        # in-flight and slot in over subsequent legs without pausing the
        # cycle. Idle spares (still hovering) also don't block.
        if state.pre_reform_active is not None:
            check_mask = alive & state.pre_reform_active
        else:
            check_mask = alive & state.is_active
        all_locked = (
            bool((state.locked[check_mask]).all())
            if check_mask.any()
            else True
        )
        if all_locked:
            state.quiescent_streak += 1
        else:
            state.quiescent_streak = 0
        if (
            state.quiescent_streak >= config.quiescent_ticks_required
            or elapsed_ticks > config.max_phase_ticks
        ):
            state.leg_index += 1
            state.leg_start_centroid = centroid.copy()
            state.phase = Phase.MOVE
            state.phase_start_tick = tick
            state.quiescent_streak = 0
            state.locked[:] = False  # unlock for next move leg


def mission_force(
    state: MissionState,
    config: MissionConfig,
    pos: np.ndarray,
    vel: np.ndarray,
    alive: np.ndarray,
    current_degree: np.ndarray,
) -> np.ndarray:
    """Compute external force per drone, with per-drone isolation override."""
    n = pos.shape[0]
    force = np.zeros_like(pos)

    if state.phase == Phase.MOVE:
        target_v = config.velocity_mps * state.heading
        for i in range(n):
            if alive[i]:
                force[i] = config.drive_gain * (target_v - vel[i])

    elif state.phase == Phase.SETTLE:
        for i in range(n):
            if alive[i]:
                force[i] = -config.drive_gain * vel[i]

    elif state.phase == Phase.REFORM:
        if state.targets is not None:
            for i in range(n):
                if alive[i]:
                    force[i] = (
                        config.attractor_gain * (state.targets[i] - pos[i])
                        - config.drive_gain * vel[i]
                    )

    # Locked drones contribute zero external force (they don't move regardless).
    if state.locked is not None:
        force[state.locked] = 0.0

    return force
