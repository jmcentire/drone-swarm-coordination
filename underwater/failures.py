# /// script
# dependencies = ["numpy<3"]
# ///
"""Failure-injection primitives for swarm benches.

Each failure type is a callable that mutates simulator state at a given
tick. Composable so a single scenario can stack multiple failure modes
(e.g. cluster loss + sensor noise spike + partition).

Conventions:
  * alive: (N,) bool array. False means dead -- drone holds last position,
    does not move, does not transmit. Edges to dead drones are pruned in
    metrics that should ignore them.
  * obstacles: list of (center, radius) tuples. Drones cannot enter the
    interior; collision is enforced by a projection constraint in PBD.
  * comms_partition: set of (i, j) pairs that cannot communicate, OR a
    pair of disjoint subsets representing a hard partition. Affects
    neighbor graph and gossip.

For Byzantine and obstacles we keep stubs here so the bench scaffold is
uniform; the next iteration fills in the missing pieces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FailureSchedule:
    """A timeline of failure events to apply during a simulation run."""

    kill_events: list[tuple[int, list[int]]] = field(default_factory=list)
    """List of (tick, drone_ids_to_kill)."""

    partition_events: list[tuple[int, int, set[int], set[int]]] = field(
        default_factory=list
    )
    """List of (start_tick, end_tick, side_a, side_b). During the interval,
    no communication is possible between side_a and side_b. Heals at end_tick."""

    diversion_events: list[tuple[int, int, list[int], np.ndarray]] = field(
        default_factory=list
    )
    """List of (start_tick, end_tick, drone_ids, displacement_direction).
    During the interval, those drones get a perpendicular drift force,
    simulating a current / disturbance that physically separates them
    from the main body. Heals (force removed) at end_tick."""

    obstacles: list[tuple[np.ndarray, float]] = field(default_factory=list)
    """Static obstacles: (center, radius). Constant for the whole mission."""


def select_corner_drones(
    positions: np.ndarray, n: int, neighbors: list[list[int]]
) -> list[int]:
    """Pick the n drones with the lowest neighbor degree (boundary/corner)."""
    deg = np.array([len(ns) for ns in neighbors])
    return list(np.argsort(deg)[:n])


def select_cluster_drones(
    positions: np.ndarray, center_idx: int, radius: float
) -> list[int]:
    """All drones within `radius` of positions[center_idx]."""
    dists = np.linalg.norm(positions - positions[center_idx], axis=1)
    return list(np.where(dists <= radius)[0])


def select_random_drones(
    n_total: int, n_victims: int, rng: np.random.Generator
) -> list[int]:
    return list(rng.choice(n_total, size=n_victims, replace=False))


def apply_kills_at_tick(
    schedule: FailureSchedule, tick: int, alive: np.ndarray
) -> int:
    """Apply any kill events scheduled at this tick. Returns number killed."""
    killed = 0
    for ev_tick, victims in schedule.kill_events:
        if ev_tick == tick:
            for v in victims:
                if alive[v]:
                    alive[v] = False
                    killed += 1
    return killed


def is_partitioned(
    schedule: FailureSchedule, tick: int, i: int, j: int
) -> bool:
    """Check if (i, j) cannot communicate at this tick due to a partition event."""
    for start, end, side_a, side_b in schedule.partition_events:
        if start <= tick < end:
            if (i in side_a and j in side_b) or (i in side_b and j in side_a):
                return True
    return False


def effective_neighbors(
    base_neighbors: list[list[int]],
    schedule: FailureSchedule,
    tick: int,
    alive: np.ndarray,
) -> list[list[int]]:
    """Filter neighbor list to remove dead drones and partition-blocked pairs."""
    result: list[list[int]] = []
    for i, ns in enumerate(base_neighbors):
        if not alive[i]:
            result.append([])
            continue
        filtered = [
            j for j in ns
            if alive[j] and not is_partitioned(schedule, tick, i, j)
        ]
        result.append(filtered)
    return result


def live_edge_strain(
    pos: np.ndarray, edges: np.ndarray, rest: float, alive: np.ndarray
) -> np.ndarray:
    """Edge strain considering only edges between two alive drones."""
    a = edges[:, 0]
    b = edges[:, 1]
    live_mask = alive[a] & alive[b]
    if not live_mask.any():
        return np.array([0.0])
    d = np.linalg.norm(pos[a[live_mask]] - pos[b[live_mask]], axis=1)
    return (d - rest) / rest


def apply_obstacle_projection(
    pos: np.ndarray, obstacles: list[tuple[np.ndarray, float]], alive: np.ndarray
) -> None:
    """Push any drone inside an obstacle to its surface. Mutates pos."""
    if not obstacles:
        return
    for center, radius in obstacles:
        offsets = pos - center
        dists = np.linalg.norm(offsets, axis=1)
        inside = (dists < radius) & alive & (dists > 1e-9)
        if inside.any():
            scale = (radius / dists[inside])[:, None]
            pos[inside] = center + offsets[inside] * scale


def diversion_force(
    schedule: FailureSchedule,
    tick: int,
    n: int,
    magnitude: float,
) -> np.ndarray:
    """Returns (N, 3) per-drone diversion force from any active diversion
    events. Drones not in any event get zero force. The force is constant
    magnitude in the event's direction, modeling a sustained current."""
    force = np.zeros((n, 3))
    for start, end, drone_ids, direction in schedule.diversion_events:
        if start <= tick < end:
            unit = direction / max(np.linalg.norm(direction), 1e-9)
            for d in drone_ids:
                if 0 <= d < n:
                    force[d] += magnitude * unit
    return force


def active_diversion_partitions(
    schedule: FailureSchedule, tick: int
) -> list[tuple[set[int], set[int]]]:
    """Return active diversion partitions at this tick.

    Each entry is (diverged_set, rest_set) -- edges between them should be
    suppressed in PBD (the drift exceeds the spring's coupling, so the
    sub-group physically separates) and in the comms graph (out of acoustic
    range)."""
    partitions: list[tuple[set[int], set[int]]] = []
    for start, end, drone_ids, _ in schedule.diversion_events:
        if start <= tick < end:
            partitions.append((set(int(d) for d in drone_ids), set()))
    return partitions


def filter_edges_for_partitions(
    edges: np.ndarray,
    partitions: list[tuple[set[int], set[int]]],
    n_total: int,
) -> np.ndarray:
    """Return a boolean mask for `edges`: True = keep, False = drop because
    the edge crosses an active partition boundary."""
    if not partitions:
        return np.ones(edges.shape[0], dtype=bool)
    rests = []
    for diverged, _ in partitions:
        rest = set(range(n_total)) - diverged
        rests.append((diverged, rest))
    keep = np.ones(edges.shape[0], dtype=bool)
    for k in range(edges.shape[0]):
        i, j = int(edges[k, 0]), int(edges[k, 1])
        for diverged, rest in rests:
            if (i in diverged and j in rest) or (i in rest and j in diverged):
                keep[k] = False
                break
    return keep
