# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Per-drone Hungarian assignment + spare-hover for the target manifold.

Ported from the validated underwater work (~/Code/drone_swarm/underwater/
mission.py compute_manifold_targets) after a side trip through recursive
PCA bisection was found to under-perform on the surplus-fills-gaps
scenarios. The underwater design decision was explicit:

  "kept Hungarian over recursive bisection (bisection didn't fit HCP
   geometry). PBD mass-weighting (locked = zero inv mass) anchors the
   scaffold without spares getting jostled."  -- kindex 8a141dbfd90d

compute_target() runs PER DRONE; each drone runs the same code on its
locally-known drone set and produces its own assignment. Consensus is
by determinism (same input, same output), not by gossip-of-the-decision.
This module has no global state.

Two roles emerge from each per-drone call:
  - ACTIVE: top n_slots drones by ID (priority == ID by convention) occupy
    manifold slots; assigned by Hungarian / linear_sum_assignment over the
    locally-known active set to minimize total drone-to-slot distance.
  - SPARE: lower-priority drones beyond n_slots hover in a ring around the
    manifold centroid. Each spare's hover position is keyed off its drone
    ID modulo a stable constant -- so when an active dies and a spare is
    promoted, the OTHER spares' hover slots don't reshuffle (which would
    keep REFORM from ever converging).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


_RING_PERIOD = 32  # stable modulus for spare-ring slot assignment
_RING_RADIUS_X_SPACING = 1.5  # spare ring radius = this * spacing
_Z_OFFSET_FRACTION = 0.3      # spare ring vertical stagger


def _estimate_spacing(manifold: np.ndarray) -> float:
    """Approximate inter-slot spacing as the median nearest-neighbor
    distance over the manifold. Used to scale the spare-hover ring."""
    if len(manifold) < 2:
        return 8.0
    diffs = manifold[:, None, :] - manifold[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
    np.fill_diagonal(d2, np.inf)
    nearest = np.sqrt(d2.min(axis=1))
    return float(np.median(nearest))


def _spare_hover_position(drone_id: int, centroid: np.ndarray, spacing: float) -> np.ndarray:
    """Deterministic hover slot for a spare drone -- keyed by ID modulo a
    fixed constant so the slot is invariant under other spares' promotions."""
    local_idx = int(drone_id) % _RING_PERIOD
    angle = 2.0 * np.pi * local_idx / _RING_PERIOD
    z_offset = ((local_idx % 3) - 1) * _Z_OFFSET_FRACTION * spacing
    radius = _RING_RADIUS_X_SPACING * spacing
    return centroid + np.array(
        [radius * np.cos(angle), radius * np.sin(angle), z_offset],
        dtype=np.float64,
    )


def compute_target(
    my_id: int,
    drones: list[dict],
    manifold: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Run the per-drone Hungarian assignment.

    drones: [{'id': int, 'pos': np.ndarray(3,)}, ...] -- the LOCALLY-KNOWN
            alive drone set as far as this drone can tell. NOT a global set.
    manifold: (N_slots, 3) array of target positions.

    Returns (target_position, is_primary_at_slot).
        is_primary_at_slot == True  -> this drone is an active and holds
                                       a specific manifold slot.
        is_primary_at_slot == False -> this drone is a spare and is parked
                                       at a deterministic ring slot around
                                       the manifold centroid.

    Determinism: every drone that has the same `drones` set and same
    `manifold` will produce the same answer. Drones with divergent local
    views (gossip not converged, partitioned) will produce divergent
    answers -- that divergence IS the protocol's behavior under those
    conditions and is measured rather than hidden.
    """
    manifold = np.asarray(manifold, dtype=np.float64)
    n_slots = len(manifold)
    if n_slots == 0:
        return np.zeros(3, dtype=np.float64), False

    centroid = manifold.mean(axis=0)
    spacing = _estimate_spacing(manifold)

    by_id: dict[int, np.ndarray] = {
        int(d["id"]): np.asarray(d["pos"], dtype=np.float64)
        for d in drones
    }
    if my_id not in by_id:
        # Caller is not in their own known set -- pathological; hover at centroid.
        return centroid.copy(), False

    # Active = top n_slots IDs by priority (priority == ID by convention in
    # this protocol; the highest-priority drones occupy the lattice).
    sorted_ids = sorted(by_id.keys(), reverse=True)
    active_ids = sorted_ids[:n_slots]
    am_active = my_id in active_ids

    if am_active:
        # Hungarian over active drones x manifold slots.
        active_pos = np.array([by_id[aid] for aid in active_ids], dtype=np.float64)
        # cost[r, c] = distance from active drone r to slot c. If we have
        # fewer actives than slots, Hungarian assigns each active to its
        # closest free slot and leaves the rest empty.
        cost = np.linalg.norm(
            active_pos[:, None, :] - manifold[None, :, :], axis=-1
        )
        row_idx, col_idx = linear_sum_assignment(cost)
        my_row = active_ids.index(my_id)
        # Find which column my_row was assigned to.
        my_assignment = np.where(row_idx == my_row)[0]
        if my_assignment.size == 0:
            # Hungarian didn't assign me a slot (should not happen with
            # n_actives <= n_slots; defensive fall-through).
            return _spare_hover_position(my_id, centroid, spacing), False
        my_slot = int(col_idx[my_assignment[0]])
        return manifold[my_slot].copy(), True

    # Spare: ring hover around the manifold centroid.
    return _spare_hover_position(my_id, centroid, spacing), False


# ---------------------------------------------------------------------------
# Self tests.
# ---------------------------------------------------------------------------


def _tests() -> int:
    failed = 0
    rng = np.random.default_rng(0)

    # T1: equal-count case -- all drones get distinct slots, primary=True for all.
    manifold = rng.normal(size=(8, 3)) * 5.0
    starts = rng.normal(size=(8, 3)) * 10.0
    drones = [{"id": i, "pos": starts[i].copy()} for i in range(8)]
    primaries = 0
    slots_used = set()
    for i in range(8):
        t, primary = compute_target(i, drones, manifold)
        if primary:
            primaries += 1
            slots_used.add(tuple(np.round(t, 6)))
    if primaries != 8:
        print(f"FAIL T1: expected 8 primaries, got {primaries}")
        failed += 1
    if len(slots_used) != 8:
        print(f"FAIL T1: expected 8 distinct slots, got {len(slots_used)}")
        failed += 1

    # T2: surplus case -- 12 drones, 8 slots. Top-8 IDs (4..11) get slots,
    # IDs 0..3 hover.
    manifold = rng.normal(size=(8, 3)) * 5.0
    starts = rng.normal(size=(12, 3)) * 10.0
    drones = [{"id": i, "pos": starts[i].copy()} for i in range(12)]
    centroid = manifold.mean(axis=0)
    for i in range(12):
        t, primary = compute_target(i, drones, manifold)
        if i < 4:
            if primary:
                print(f"FAIL T2a: drone {i} (low ID) should be spare, got primary")
                failed += 1
            # Hover position should be near centroid, far from any slot.
            slot_min = float(min(np.linalg.norm(t - m) for m in manifold))
            cent_d = float(np.linalg.norm(t - centroid))
            if slot_min < 0.1:
                print(f"FAIL T2a: spare {i} too close to a manifold slot")
                failed += 1
            _ = cent_d  # quieter
        else:
            if not primary:
                print(f"FAIL T2b: drone {i} (high ID) should be primary, got spare")
                failed += 1

    # T3: promotion -- after a high-priority drone dies, the next-highest
    # spare gets promoted into the slot pool.
    manifold = rng.normal(size=(8, 3)) * 5.0
    starts = rng.normal(size=(12, 3)) * 10.0
    drones_full = [{"id": i, "pos": starts[i].copy()} for i in range(12)]
    drones_after_death = [d for d in drones_full if d["id"] != 11]  # ID 11 (highest) dies
    # Drone 3 was spare in full set; should now be primary (top 8 of {0..10}).
    t_before, p_before = compute_target(3, drones_full, manifold)
    t_after, p_after = compute_target(3, drones_after_death, manifold)
    if p_before:
        print("FAIL T3: drone 3 should have been spare in full set")
        failed += 1
    if not p_after:
        print("FAIL T3: drone 3 should have been promoted after ID 11 died")
        failed += 1

    # T4: hover slot for a spare is stable under other spares' promotion --
    # this is the property that makes REFORM converge (per underwater notes).
    manifold = rng.normal(size=(8, 3)) * 5.0
    starts = rng.normal(size=(13, 3)) * 10.0
    drones_a = [{"id": i, "pos": starts[i].copy()} for i in range(13)]
    drones_b = [d for d in drones_a if d["id"] != 4]  # ID 4 promoted
    t_a, p_a = compute_target(2, drones_a, manifold)   # ID 2 is spare in both
    t_b, p_b = compute_target(2, drones_b, manifold)
    if p_a or p_b:
        print("FAIL T4: ID 2 should be spare in both cases")
        failed += 1
    if not np.allclose(t_a, t_b):
        print(f"FAIL T4: spare hover slot moved when another spare was promoted: "
              f"{t_a} -> {t_b}")
        failed += 1

    # T5: divergent local views produce different answers.
    manifold = rng.normal(size=(8, 3)) * 5.0
    starts = rng.normal(size=(12, 3)) * 10.0
    drones_full = [{"id": i, "pos": starts[i].copy()} for i in range(12)]
    drones_partial = drones_full[:10]  # IDs 10, 11 not yet heard about by this drone
    # Drone 3 should be spare in full view (top 8 are 4..11) and ACTIVE in the
    # partial view (top 8 are 2..9).
    _, p_full = compute_target(3, drones_full, manifold)
    _, p_part = compute_target(3, drones_partial, manifold)
    if p_full:
        print("FAIL T5: full view -- drone 3 should be spare")
        failed += 1
    if not p_part:
        print("FAIL T5: partial view -- drone 3 should be primary")
        failed += 1

    return failed


if __name__ == "__main__":
    n_failed = _tests()
    print("manifold: all tests passed" if n_failed == 0 else f"manifold: {n_failed} tests failed")
