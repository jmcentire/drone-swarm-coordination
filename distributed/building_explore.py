# /// script
# dependencies = ["numpy<3", "matplotlib"]
# ///
"""Observation-driven building exploration mission layer.

No central human operator. No prior terrain map.

This module consumes only what can be gossiped in the existing MAP/VOTE
pipeline: relative positions plus compact feature observations such as
"entrance seen", "intersection seen", "frontier here", and "target
detected". Given the same MAP-derived observation log, every drone can
derive the same roles:

  - sentinels hold entrances and only purpose-bearing intersections
  - target discoverers guard until extraction, then join transport duty
  - workers keep exploring frontier features, join escorts, then return

The current elected leader only emits the resulting Command payload. It
is not a permanent commander and does not need secret terrain knowledge;
if it dies, the next leader recomputes the same payload from the same
observations.

The output is compatible with manifold.compute_mission_target():
  station_targets + station_drone_ids + mission_objective
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from manifold import compute_mission_target


class ObservationKind(Enum):
    BUILDING = "building"
    ENTRANCE = "entrance"
    INTERSECTION = "intersection"
    FRONTIER = "frontier"
    ROOM = "room"
    TARGET = "target"
    CLEAR = "clear"


class MissionPhase(Enum):
    FIND_ENTRANCE = "find_entrance"
    EXPLORE = "explore"
    CALL_HELP_AND_EXPLORE = "call_help_and_explore"
    EXTRACT = "extract"
    RETURN_HOME = "return_home"
    COMPLETE = "complete"


@dataclass(frozen=True)
class BuildingObservation:
    """One compact observation piggybacked on MAP chatter.

    pos is in the swarm's current relative consensus frame, not a global
    building map.
    """
    tick: int
    drone_id: int
    kind: ObservationKind
    pos: np.ndarray
    confidence: float = 1.0


@dataclass(frozen=True)
class Feature:
    feature_id: str
    kind: ObservationKind
    pos: np.ndarray
    first_tick: int
    first_drone_id: int
    observer_ids: tuple[int, ...]
    observation_indices: tuple[int, ...]


@dataclass(frozen=True)
class BuildingPlan:
    phase: MissionPhase
    payload: dict[str, Any]
    features: tuple[Feature, ...]
    sentinel_feature_ids: tuple[str, ...]
    worker_ids: tuple[int, ...]


@dataclass(frozen=True)
class HiddenBuilding:
    """Truth model used only by the random simulator/visualizer."""
    positions: dict[str, np.ndarray]
    edges: dict[str, tuple[str, ...]]
    entrance_id: str
    target_ids: tuple[str, ...]
    home_pos: np.ndarray


@dataclass(frozen=True)
class TraceStep:
    step: int
    plan: BuildingPlan
    observations: tuple[BuildingObservation, ...]
    drones: tuple[dict, ...]
    visited_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class SmoothFloorplan:
    home_pos: np.ndarray
    building_center: np.ndarray
    walls: tuple[tuple[np.ndarray, np.ndarray, str], ...]
    doors: dict[str, np.ndarray]
    nodes: dict[str, np.ndarray]
    edges: dict[str, tuple[str, ...]]
    room_ids: tuple[str, ...]
    target_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class SmoothFrame:
    frame: int
    phase: str
    drones: np.ndarray
    roles: tuple[str, ...]
    known_wall_ids: frozenset[str]
    discovered_node_ids: frozenset[str]
    discovered_target_ids: frozenset[str]
    carried_targets: dict[str, np.ndarray]
    target_escorts: dict[str, tuple[int, ...]]
    failed_drone_ids: frozenset[int] = field(default_factory=frozenset)
    relay_drone_ids: frozenset[int] = field(default_factory=frozenset)
    comm_connected_ids: frozenset[int] = field(default_factory=frozenset)
    comm_degraded: bool = False


@dataclass(frozen=True)
class SmoothSharedMap:
    building_known: bool
    clear_seen: bool
    known_nodes: frozenset[str]
    known_doors: frozenset[str]
    known_edges: dict[str, tuple[str, ...]]
    known_wall_ids: frozenset[str]


def _qpos(pos: np.ndarray, scale: float = 10.0) -> tuple[int, int, int]:
    p = np.asarray(pos, dtype=np.float64)
    return tuple(int(round(float(x) * scale)) for x in p)


def _sorted_observations(observations: list[BuildingObservation]) -> list[tuple[int, BuildingObservation]]:
    indexed = list(enumerate(observations))
    indexed.sort(
        key=lambda item: (
            item[1].tick,
            item[1].kind.value,
            item[1].drone_id,
            _qpos(item[1].pos),
            item[0],
        )
    )
    return indexed


def derive_features(
    observations: list[BuildingObservation],
    *,
    merge_radius_m: float = 1.5,
) -> tuple[Feature, ...]:
    """Cluster same-kind observations into deterministic local features."""
    clusters: list[dict[str, Any]] = []
    for obs_idx, obs in _sorted_observations(observations):
        if obs.kind == ObservationKind.CLEAR:
            continue
        pos = np.asarray(obs.pos, dtype=np.float64)
        match_idx = None
        for i, c in enumerate(clusters):
            if c["kind"] != obs.kind:
                continue
            if float(np.linalg.norm(pos - c["pos"])) <= merge_radius_m:
                match_idx = i
                break
        if match_idx is None:
            clusters.append({
                "kind": obs.kind,
                "pos": pos.copy(),
                "first_tick": obs.tick,
                "first_drone_id": obs.drone_id,
                "observer_ids": {obs.drone_id},
                "observation_indices": [obs_idx],
            })
            continue
        c = clusters[match_idx]
        n_prev = len(c["observation_indices"])
        c["pos"] = (c["pos"] * n_prev + pos) / (n_prev + 1)
        c["observer_ids"].add(obs.drone_id)
        c["observation_indices"].append(obs_idx)

    features = []
    kind_counts: dict[ObservationKind, int] = {}
    for c in clusters:
        kind = c["kind"]
        ordinal = kind_counts.get(kind, 0)
        kind_counts[kind] = ordinal + 1
        features.append(Feature(
            feature_id=f"{kind.value}_{ordinal}",
            kind=kind,
            pos=np.asarray(c["pos"], dtype=np.float64),
            first_tick=int(c["first_tick"]),
            first_drone_id=int(c["first_drone_id"]),
            observer_ids=tuple(sorted(int(x) for x in c["observer_ids"])),
            observation_indices=tuple(int(x) for x in c["observation_indices"]),
        ))
    return tuple(features)


def _feature_sort_key(feature: Feature) -> tuple[int, str, int, int, tuple[int, int, int]]:
    kind_rank = {
        ObservationKind.ENTRANCE: 0,
        ObservationKind.INTERSECTION: 1,
        ObservationKind.TARGET: 2,
        ObservationKind.ROOM: 3,
        ObservationKind.FRONTIER: 4,
        ObservationKind.BUILDING: 5,
        ObservationKind.CLEAR: 6,
    }[feature.kind]
    return (
        kind_rank,
        feature.kind.value,
        feature.first_tick,
        feature.first_drone_id,
        _qpos(feature.pos),
    )


def _station_features(features: tuple[Feature, ...]) -> list[Feature]:
    return sorted(
        [
            f for f in features
            if f.kind in {
                ObservationKind.ENTRANCE,
                ObservationKind.INTERSECTION,
                ObservationKind.TARGET,
            }
        ],
        key=_feature_sort_key,
    )


def _frontier_features(features: tuple[Feature, ...]) -> list[Feature]:
    return sorted(
        [f for f in features if f.kind == ObservationKind.FRONTIER],
        key=lambda f: (f.first_tick, f.first_drone_id, _qpos(f.pos), f.feature_id),
    )


def _active_frontier_features(
    features: tuple[Feature, ...],
    *,
    merge_radius_m: float = 1.5,
) -> list[Feature]:
    resolved = [
        f for f in features
        if f.kind in {
            ObservationKind.ENTRANCE,
            ObservationKind.INTERSECTION,
            ObservationKind.ROOM,
            ObservationKind.TARGET,
        }
    ]
    active = []
    for frontier in _frontier_features(features):
        if any(float(np.linalg.norm(frontier.pos - station.pos)) <= merge_radius_m for station in resolved):
            continue
        active.append(frontier)
    return active


def _target_features(features: tuple[Feature, ...]) -> list[Feature]:
    return sorted(
        [f for f in features if f.kind == ObservationKind.TARGET],
        key=lambda f: (f.first_tick, f.first_drone_id, _qpos(f.pos), f.feature_id),
    )


def _distance_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return float(np.linalg.norm(point - a))
    t = max(0.0, min(1.0, float(np.dot(point - a, ab) / denom)))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def _is_downstream(
    root: np.ndarray,
    station: np.ndarray,
    point: np.ndarray,
    *,
    merge_radius_m: float,
) -> bool:
    root = np.asarray(root, dtype=np.float64)
    station = np.asarray(station, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    station_depth = float(np.linalg.norm(station - root))
    point_depth = float(np.linalg.norm(point - root))
    if point_depth <= station_depth + merge_radius_m:
        return False
    direct_support_radius = max(11.0, merge_radius_m * 7.0)
    corridor_radius = max(5.0, merge_radius_m * 3.5)
    return (
        float(np.linalg.norm(point - station)) <= direct_support_radius
        or _distance_to_segment(station, root, point) <= corridor_radius
    )


def _purpose_station_features(
    features: tuple[Feature, ...],
    drones: list[dict],
    *,
    phase: MissionPhase,
    active_frontiers: list[Feature],
    home_pos: np.ndarray,
    merge_radius_m: float,
) -> tuple[list[Feature], dict[str, str]]:
    entrances = [f for f in _station_features(features) if f.kind == ObservationKind.ENTRANCE]
    intersections = [f for f in _station_features(features) if f.kind == ObservationKind.INTERSECTION]
    targets = _target_features(features)
    root = entrances[0].pos if entrances else np.asarray(home_pos, dtype=np.float64)

    active_feature_points: list[np.ndarray] = []
    if phase in {MissionPhase.EXPLORE, MissionPhase.CALL_HELP_AND_EXPLORE}:
        active_feature_points.extend(f.pos for f in active_frontiers)
    if phase in {MissionPhase.CALL_HELP_AND_EXPLORE, MissionPhase.EXTRACT}:
        active_feature_points.extend(f.pos for f in targets)

    drone_points = [np.asarray(d["pos"], dtype=np.float64) for d in drones]
    downstream_points = active_feature_points + drone_points
    active_downstream = any(
        float(np.linalg.norm(p - root)) > merge_radius_m * 2.0
        for p in downstream_points
    )

    station_features: list[Feature] = []
    reasons: dict[str, str] = {}

    if phase in {
        MissionPhase.EXPLORE,
        MissionPhase.CALL_HELP_AND_EXPLORE,
        MissionPhase.EXTRACT,
    }:
        for entrance in entrances:
            if active_downstream:
                station_features.append(entrance)
                reasons[entrance.feature_id] = "gateway_for_downstream_drones"

    if phase == MissionPhase.CALL_HELP_AND_EXPLORE:
        for target in targets:
            station_features.append(target)
            reasons[target.feature_id] = "guard_target_until_transport"

    if phase in {
        MissionPhase.EXPLORE,
        MissionPhase.CALL_HELP_AND_EXPLORE,
        MissionPhase.EXTRACT,
    }:
        for intersection in intersections:
            if any(
                _is_downstream(root, intersection.pos, point, merge_radius_m=merge_radius_m)
                for point in downstream_points
            ):
                station_features.append(intersection)
                reasons[intersection.feature_id] = "downstream_drones_or_objectives"

    station_features.sort(key=_feature_sort_key)
    return station_features, reasons


def assign_sentinel_roles(
    drones: list[dict],
    features: tuple[Feature, ...],
    *,
    station_features: list[Feature] | None = None,
) -> dict[str, int]:
    """Assign station features to sentinel drones by discovery order.

    Primary rule: the first drone that discovered the entrance,
    intersection, or target becomes its sentinel. If that drone is gone
    or already holding an earlier purposeful station, fall back to another
    observer, then to the nearest available drone in the relative map.
    """
    by_id = {int(d["id"]): np.asarray(d["pos"], dtype=np.float64) for d in drones}
    assigned_drones: set[int] = set()
    out: dict[str, int] = {}
    ordered_station_features = station_features if station_features is not None else _station_features(features)
    for feature in ordered_station_features:
        candidates = [
            did for did in (feature.first_drone_id, *feature.observer_ids)
            if did in by_id and did not in assigned_drones
        ]
        if not candidates:
            nearest = []
            for did, pos in by_id.items():
                if did in assigned_drones:
                    continue
                dist = float(np.linalg.norm(pos - feature.pos))
                nearest.append((round(dist, 9), -did, did))
            if nearest:
                candidates = [min(nearest)[2]]
        if not candidates:
            continue
        chosen = candidates[0]
        out[feature.feature_id] = chosen
        assigned_drones.add(chosen)
    return out


def infer_phase(
    features: tuple[Feature, ...],
    observations: list[BuildingObservation],
    *,
    active_frontiers: list[Feature] | None = None,
    max_targets: int = 3,
    require_clear_for_extraction: bool = True,
    extraction_started: bool = False,
    extracted: bool = False,
    returned_home: bool = False,
) -> MissionPhase:
    if returned_home:
        return MissionPhase.COMPLETE
    if extracted:
        return MissionPhase.RETURN_HOME
    targets = _target_features(features)
    frontiers = active_frontiers if active_frontiers is not None else _active_frontier_features(features)
    clear_seen = any(obs.kind == ObservationKind.CLEAR for obs in observations)
    if clear_seen and targets:
        return MissionPhase.EXTRACT
    if clear_seen:
        return MissionPhase.RETURN_HOME
    if extraction_started:
        return MissionPhase.EXTRACT
    if targets:
        if require_clear_for_extraction:
            return MissionPhase.CALL_HELP_AND_EXPLORE
        if len(targets) >= max_targets:
            return MissionPhase.EXTRACT
        return MissionPhase.CALL_HELP_AND_EXPLORE if frontiers else MissionPhase.EXTRACT
    if any(f.kind == ObservationKind.ENTRANCE for f in features):
        return MissionPhase.EXPLORE if frontiers else MissionPhase.RETURN_HOME
    return MissionPhase.FIND_ENTRANCE


def plan_from_observations(
    drones: list[dict],
    observations: list[BuildingObservation],
    *,
    home_pos: np.ndarray,
    max_targets: int = 3,
    require_clear_for_extraction: bool = True,
    extraction_started: bool = False,
    extracted: bool = False,
    returned_home: bool = False,
    merge_radius_m: float = 1.5,
) -> BuildingPlan:
    """Compute deterministic sentinels/workers and Command payload."""
    features = derive_features(observations, merge_radius_m=merge_radius_m)
    active_frontiers = _active_frontier_features(features, merge_radius_m=merge_radius_m)
    phase = infer_phase(
        features,
        observations,
        active_frontiers=active_frontiers,
        max_targets=max_targets,
        require_clear_for_extraction=require_clear_for_extraction,
        extraction_started=extraction_started,
        extracted=extracted,
        returned_home=returned_home,
    )

    purpose_stations, station_reasons = _purpose_station_features(
        features,
        drones,
        phase=phase,
        active_frontiers=active_frontiers,
        home_pos=home_pos,
        merge_radius_m=merge_radius_m,
    )
    sentinel_by_feature = assign_sentinel_roles(
        drones,
        features,
        station_features=purpose_stations,
    )
    sentinel_ids = set(sentinel_by_feature.values())
    worker_ids = tuple(sorted((int(d["id"]) for d in drones if int(d["id"]) not in sentinel_ids), reverse=True))
    station_features = [f for f in purpose_stations if f.feature_id in sentinel_by_feature]
    station_targets = (
        np.array([f.pos for f in station_features], dtype=np.float64)
        if station_features
        else np.zeros((0, 3), dtype=np.float64)
    )
    station_drone_ids = [sentinel_by_feature[f.feature_id] for f in station_features]

    frontiers = active_frontiers
    targets = _target_features(features)
    if phase == MissionPhase.FIND_ENTRANCE:
        buildings = sorted(
            [f for f in features if f.kind == ObservationKind.BUILDING],
            key=lambda f: (f.first_tick, f.first_drone_id, _qpos(f.pos)),
        )
        center = buildings[0].pos if buildings else np.asarray(home_pos, dtype=np.float64)
        objective = {"kind": "perimeter", "center": center, "radius_m": 12.0}
        objective_feature_ids: list[str] = []
    elif phase in {MissionPhase.EXPLORE, MissionPhase.CALL_HELP_AND_EXPLORE}:
        objective_features = frontiers
        if phase == MissionPhase.CALL_HELP_AND_EXPLORE:
            # Keep searching for an unknown 0-3 target count while the
            # target stations stay guarded and help can route back through
            # entrance/intersection sentinels.
            objective_features = frontiers or targets
        objective = {
            "kind": "points",
            "points": np.array([f.pos for f in objective_features], dtype=np.float64)
                if objective_features else station_targets,
        }
        objective_feature_ids = [f.feature_id for f in objective_features]
    elif phase == MissionPhase.EXTRACT:
        objective = {
            "kind": "points",
            "points": np.array([f.pos for f in targets], dtype=np.float64)
                if targets else station_targets,
        }
        objective_feature_ids = [f.feature_id for f in targets]
    else:
        objective = {
            "kind": "point",
            "point": np.asarray(home_pos, dtype=np.float64),
            "standoff_radius_m": max(4.0, len(drones) * 0.75),
        }
        objective_feature_ids = []

    payload = {
        "mission_type": "building_exploration",
        "phase": phase.value,
        "station_targets": station_targets,
        "station_drone_ids": station_drone_ids,
        "sentinel_assignments": dict(sentinel_by_feature),
        "sentinel_feature_ids": [f.feature_id for f in station_features],
        "sentinel_reasons": {f.feature_id: station_reasons.get(f.feature_id, "purpose_station") for f in station_features},
        "worker_ids": list(worker_ids),
        "mission_objective": objective,
        "objective": objective,
        "objective_feature_ids": objective_feature_ids,
        "target_count_known": len(targets),
        "target_count_max": max_targets,
        "requires_clear_for_extraction": require_clear_for_extraction,
        "help_requested": phase in {
            MissionPhase.CALL_HELP_AND_EXPLORE,
            MissionPhase.EXTRACT,
        },
        "minimal_comms": "piggyback feature observations on MAP; roles recompute deterministically after MAP/VOTE",
    }
    return BuildingPlan(
        phase=phase,
        payload=payload,
        features=features,
        sentinel_feature_ids=tuple(f.feature_id for f in station_features),
        worker_ids=worker_ids,
    )


def sample_observations(target_count: int = 1, *, clear: bool = False) -> list[BuildingObservation]:
    """Synthetic MAP observation stream. The planner never sees a graph."""
    obs = [
        BuildingObservation(0, 0, ObservationKind.BUILDING, np.array([0.0, 0.0, 0.0])),
        BuildingObservation(2, 2, ObservationKind.ENTRANCE, np.array([0.0, -8.0, 0.0])),
        BuildingObservation(4, 3, ObservationKind.FRONTIER, np.array([0.0, -2.0, 0.0])),
        BuildingObservation(7, 4, ObservationKind.INTERSECTION, np.array([0.0, 5.0, 0.0])),
        BuildingObservation(8, 5, ObservationKind.FRONTIER, np.array([-6.0, 5.0, 0.0])),
        BuildingObservation(8, 6, ObservationKind.FRONTIER, np.array([6.0, 5.0, 0.0])),
        BuildingObservation(9, 7, ObservationKind.FRONTIER, np.array([0.0, 12.0, 0.0])),
    ]
    target_positions = [
        np.array([0.0, 18.0, 0.0]),
        np.array([-10.0, 12.0, 0.0]),
        np.array([10.0, 12.0, 0.0]),
    ]
    for i in range(max(0, min(3, target_count))):
        obs.append(BuildingObservation(12 + i, 5 + i, ObservationKind.TARGET, target_positions[i]))
    if clear:
        obs.append(BuildingObservation(20, 1, ObservationKind.CLEAR, np.array([0.0, 0.0, 0.0])))
    return obs


def _drones_at_home(n: int = 8) -> list[dict]:
    home = np.array([-20.0, 0.0, 0.0])
    return [
        {"id": i, "pos": home + np.array([0.0, i * 0.2, 0.0])}
        for i in range(n)
    ]


def generate_random_building(
    seed: int,
    *,
    n_nodes: int | None = None,
    max_targets: int = 3,
) -> HiddenBuilding:
    """Generate a random hidden topological building for the harness.

    The planner never receives this object. The simulator reveals it
    incrementally as observations.
    """
    rng = np.random.default_rng(seed)
    if n_nodes is None:
        n_nodes = int(rng.integers(8, 14))
    n_nodes = max(5, int(n_nodes))
    home_pos = np.array([-18.0, -10.0, 0.0])
    positions: dict[str, np.ndarray] = {
        "entrance": np.array([0.0, -8.0, 0.0]),
    }
    edges: dict[str, set[str]] = {"entrance": set()}
    occupied = {(0, -2)}
    grid_by_node = {"entrance": (0, -2)}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    frontier_parents = ["entrance"]
    for idx in range(1, n_nodes):
        placed = False
        for _ in range(200):
            parent = str(rng.choice(frontier_parents))
            px, py = grid_by_node[parent]
            dx, dy = dirs[int(rng.integers(0, len(dirs)))]
            coord = (px + dx, py + dy)
            if coord in occupied or coord[1] < -2:
                continue
            node_id = f"n{idx}"
            occupied.add(coord)
            grid_by_node[node_id] = coord
            positions[node_id] = np.array([coord[0] * 6.0, coord[1] * 6.0, 0.0])
            edges.setdefault(parent, set()).add(node_id)
            edges.setdefault(node_id, set()).add(parent)
            frontier_parents.append(node_id)
            if len(edges[parent]) >= 3 and parent in frontier_parents:
                # Still allow branches, but lower repeat probability.
                frontier_parents.append(parent)
            placed = True
            break
        if not placed:
            break

    leaves = sorted(
        nid for nid, nbs in edges.items()
        if nid != "entrance" and len(nbs) == 1
    )
    target_count = int(rng.integers(0, max_targets + 1))
    target_count = min(target_count, len(leaves))
    target_ids = tuple(sorted(rng.choice(leaves, size=target_count, replace=False).tolist()))
    return HiddenBuilding(
        positions={k: v.copy() for k, v in positions.items()},
        edges={k: tuple(sorted(v)) for k, v in edges.items()},
        entrance_id="entrance",
        target_ids=target_ids,
        home_pos=home_pos,
    )


def _hidden_kind(building: HiddenBuilding, node_id: str) -> ObservationKind:
    if node_id == building.entrance_id:
        return ObservationKind.ENTRANCE
    if node_id in building.target_ids:
        return ObservationKind.TARGET
    if len(building.edges.get(node_id, ())) >= 3:
        return ObservationKind.INTERSECTION
    return ObservationKind.FRONTIER


def simulate_random_exploration(
    seed: int,
    *,
    n_drones: int = 8,
    max_steps: int = 24,
) -> tuple[HiddenBuilding, list[TraceStep]]:
    """Run the terrain-blind planner against a random hidden building."""
    building = generate_random_building(seed)
    drones = [
        {"id": i, "pos": building.home_pos + np.array([0.0, i * 0.25, 0.0])}
        for i in range(n_drones)
    ]
    observations: list[BuildingObservation] = [
        BuildingObservation(0, 0, ObservationKind.BUILDING, np.zeros(3)),
    ]
    visited: set[str] = set()
    observed_nodes: set[str] = set()
    trace: list[TraceStep] = []

    for step in range(max_steps):
        plan = plan_from_observations(
            drones,
            observations,
            home_pos=building.home_pos,
            max_targets=3,
            extracted=False,
            returned_home=False,
        )
        trace.append(TraceStep(
            step=step,
            plan=plan,
            observations=tuple(observations),
            drones=tuple({"id": int(d["id"]), "pos": np.asarray(d["pos"], dtype=np.float64).copy()} for d in drones),
            visited_node_ids=tuple(sorted(visited)),
        ))

        # Move the discrete drones directly to their assigned targets; this
        # is a mission-layer harness, not a flight dynamics simulation.
        next_drones = []
        for d in drones:
            target, _ = compute_mission_target(int(d["id"]), drones, plan.payload)
            next_drones.append({"id": int(d["id"]), "pos": target.copy()})
        drones = next_drones

        if building.entrance_id not in observed_nodes:
            discoverer = plan.worker_ids[0] if plan.worker_ids else 0
            observations.append(BuildingObservation(
                step + 1,
                int(discoverer),
                ObservationKind.ENTRANCE,
                building.positions[building.entrance_id],
            ))
            observed_nodes.add(building.entrance_id)
            visited.add(building.entrance_id)
            for nb in building.edges[building.entrance_id]:
                observations.append(BuildingObservation(
                    step + 1,
                    int(discoverer),
                    ObservationKind.FRONTIER,
                    building.positions[nb],
                ))
                observed_nodes.add(nb)
            continue

        unvisited_frontiers = sorted(
            nid for nid in observed_nodes
            if nid not in visited and nid in building.positions
        )
        if unvisited_frontiers:
            node_id = unvisited_frontiers[0]
            discoverer = plan.worker_ids[step % len(plan.worker_ids)] if plan.worker_ids else 0
            kind = _hidden_kind(building, node_id)
            observations.append(BuildingObservation(
                step + 1,
                int(discoverer),
                kind,
                building.positions[node_id],
            ))
            visited.add(node_id)
            for nb in building.edges[node_id]:
                if nb in visited or nb in observed_nodes:
                    continue
                observations.append(BuildingObservation(
                    step + 1,
                    int(discoverer),
                    ObservationKind.FRONTIER,
                    building.positions[nb],
                ))
                observed_nodes.add(nb)
            continue

        observations.append(BuildingObservation(
            step + 1,
            0,
            ObservationKind.CLEAR,
            np.zeros(3),
        ))
        break

    final_plan = plan_from_observations(
        drones,
        observations,
        home_pos=building.home_pos,
        returned_home=True,
    )
    trace.append(TraceStep(
        step=len(trace),
        plan=final_plan,
        observations=tuple(observations),
        drones=tuple({"id": int(d["id"]), "pos": np.asarray(d["pos"], dtype=np.float64).copy()} for d in drones),
        visited_node_ids=tuple(sorted(visited)),
    ))
    return building, trace


def render_exploration_trace(
    building: HiddenBuilding,
    trace: list[TraceStep],
    output_path: str,
    *,
    max_panels: int = 6,
) -> None:
    """Render a compact multi-panel PNG of exploration progress."""
    import matplotlib.pyplot as plt

    if not trace:
        return
    indices = np.linspace(0, len(trace) - 1, num=min(max_panels, len(trace)), dtype=int)
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    axes_list = axes[0]
    for ax, idx in zip(axes_list, indices):
        _draw_trace_frame(ax, building, trace[int(idx)])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_bounds(building: HiddenBuilding) -> tuple[float, float, float, float]:
    all_pts = np.array(list(building.positions.values()) + [building.home_pos], dtype=np.float64)
    x_min, y_min = all_pts[:, :2].min(axis=0) - 8
    x_max, y_max = all_pts[:, :2].max(axis=0) + 8
    return float(x_min), float(x_max), float(y_min), float(y_max)


def _hidden_node_style(building: HiddenBuilding, node_id: str) -> tuple[str, str, float]:
    if node_id == building.entrance_id:
        return "door", "#bfc9d6", 80.0
    if node_id in building.target_ids:
        return "target", "#d8d8d8", 90.0
    degree = len(building.edges.get(node_id, ()))
    if degree >= 3:
        return "intersection", "#d6cfdf", 85.0
    if degree == 1:
        return "room", "#dddddd", 95.0
    return "corridor", "#dddddd", 36.0


def _draw_trace_frame(ax: Any, building: HiddenBuilding, step: TraceStep) -> None:
    from matplotlib.patches import Circle, Rectangle

    x_min, x_max, y_min, y_max = _plot_bounds(building)
    ax.clear()
    ax.set_facecolor("#fbfbfa")

    # Hidden truth rendered softly as the building substrate. This is for
    # the viewer only; the planner never receives this full graph.
    for a, nbs in building.edges.items():
        for b in nbs:
            if a > b:
                continue
            pa = building.positions[a]
            pb = building.positions[b]
            ax.plot(
                [pa[0], pb[0]], [pa[1], pb[1]],
                color="#d8d8d5", lw=9.0, solid_capstyle="round", zorder=0,
            )
            ax.plot(
                [pa[0], pb[0]], [pa[1], pb[1]],
                color="#eeeeeb", lw=5.0, solid_capstyle="round", zorder=1,
            )
    for node_id, pos in building.positions.items():
        style, color, size = _hidden_node_style(building, node_id)
        if style == "room":
            ax.add_patch(Rectangle(
                (pos[0] - 1.7, pos[1] - 1.7), 3.4, 3.4,
                facecolor=color, edgecolor="#c6c6c1", lw=0.8, zorder=2,
            ))
        elif style == "door":
            ax.add_patch(Rectangle(
                (pos[0] - 1.5, pos[1] - 0.25), 3.0, 0.5,
                facecolor=color, edgecolor="#adb8c6", lw=0.9, zorder=2,
            ))
        elif style == "intersection":
            ax.scatter([pos[0]], [pos[1]], s=size, c=color, marker="D", edgecolors="#c1bacb", zorder=2)
        elif style == "target":
            ax.scatter([pos[0]], [pos[1]], s=size, c=color, marker="s", edgecolors="#c3c3be", zorder=2)
        else:
            ax.add_patch(Circle((pos[0], pos[1]), 0.7, facecolor=color, edgecolor="#cdcdc8", lw=0.6, zorder=2))

    colors = {
        ObservationKind.BUILDING: "#8a8a8a",
        ObservationKind.ENTRANCE: "#1f77b4",
        ObservationKind.INTERSECTION: "#9467bd",
        ObservationKind.FRONTIER: "#ff9f1c",
        ObservationKind.ROOM: "#666666",
        ObservationKind.TARGET: "#d62728",
        ObservationKind.CLEAR: "#2ca02c",
    }
    markers = {
        ObservationKind.BUILDING: "o",
        ObservationKind.ENTRANCE: "s",
        ObservationKind.INTERSECTION: "D",
        ObservationKind.FRONTIER: "o",
        ObservationKind.ROOM: "o",
        ObservationKind.TARGET: "s",
        ObservationKind.CLEAR: "*",
    }
    sizes = {
        ObservationKind.BUILDING: 70,
        ObservationKind.ENTRANCE: 105,
        ObservationKind.INTERSECTION: 105,
        ObservationKind.FRONTIER: 70,
        ObservationKind.ROOM: 70,
        ObservationKind.TARGET: 135,
        ObservationKind.CLEAR: 105,
    }
    for feature in step.plan.features:
        edge = "#222222" if feature.kind in {ObservationKind.ENTRANCE, ObservationKind.INTERSECTION, ObservationKind.TARGET} else "#775200"
        ax.scatter(
            [feature.pos[0]], [feature.pos[1]],
            s=sizes[feature.kind],
            c=colors[feature.kind],
            marker=markers[feature.kind],
            edgecolors=edge,
            linewidths=0.8,
            zorder=5,
        )

    station_by_id = step.plan.payload.get("sentinel_assignments", {})
    sentinel_ids = set(station_by_id.values())
    for drone in step.drones:
        pos = np.asarray(drone["pos"], dtype=np.float64)
        did = int(drone["id"])
        face = "#111111" if did in sentinel_ids else "#ffffff"
        edge = "#111111"
        ax.scatter([pos[0]], [pos[1]], s=92, c=face, edgecolors=edge, marker="o", linewidths=1.3, zorder=8)
        ax.text(
            pos[0], pos[1], str(did),
            color="#ffffff" if did in sentinel_ids else "#111111",
            ha="center", va="center", fontsize=7, zorder=9,
        )

    ax.scatter([building.home_pos[0]], [building.home_pos[1]], s=135, c="#2ca02c", marker="*", zorder=7)
    ax.set_title(
        f"step {step.step} | {step.plan.phase.value} | targets {step.plan.payload.get('target_count_known', 0)}",
        fontsize=11,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#ededeb", linewidth=0.6)


def render_exploration_animation(
    building: HiddenBuilding,
    trace: list[TraceStep],
    output_path: str,
    *,
    fps: float = 1.5,
) -> None:
    """Render the trace as an animated GIF using Pillow."""
    import matplotlib.pyplot as plt
    from PIL import Image

    if not trace:
        return
    frames: list[Image.Image] = []
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    try:
        for step in trace:
            _draw_trace_frame(ax, building, step)
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE))
    finally:
        plt.close(fig)
    duration_ms = max(80, int(round(1000.0 / max(0.1, fps))))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def generate_smooth_floorplan(seed: int, *, max_targets: int = 3) -> SmoothFloorplan:
    rng = np.random.default_rng(seed)
    cx = float(rng.uniform(36.0, 48.0))
    cy = float(rng.uniform(-5.0, 7.0))
    center = np.array([cx, cy, 0.0])
    width = 34.0
    height = 28.0
    left = cx - width / 2
    right = cx + width / 2
    bottom = cy - height / 2
    top = cy + height / 2
    hall_x0 = cx - 3.0
    hall_x1 = cx + 3.0
    rows = 4
    row_h = height / rows
    home = np.array([0.0, 0.0, 0.0])

    walls: list[tuple[np.ndarray, np.ndarray, str]] = []

    def add_wall(a: tuple[float, float], b: tuple[float, float], wid: str) -> None:
        walls.append((np.array([a[0], a[1], 0.0]), np.array([b[0], b[1], 0.0]), wid))

    def add_vertical_wall_with_gaps(
        x: float,
        y0: float,
        y1: float,
        gaps: list[tuple[float, float]],
        wid_prefix: str,
    ) -> None:
        cursor = y0
        segment = 0
        for gap0, gap1 in sorted(gaps):
            gap0 = max(y0, float(gap0))
            gap1 = min(y1, float(gap1))
            if gap0 > cursor:
                add_wall((x, cursor), (x, gap0), f"{wid_prefix}_{segment}")
                segment += 1
            cursor = max(cursor, gap1)
        if cursor < y1:
            add_wall((x, cursor), (x, y1), f"{wid_prefix}_{segment}")

    # Outer shell with front/back door gaps.
    door_w = 4.0
    add_wall((left, bottom), (cx - door_w / 2, bottom), "outer_front_l")
    add_wall((cx + door_w / 2, bottom), (right, bottom), "outer_front_r")
    add_wall((left, top), (cx - door_w / 2, top), "outer_back_l")
    add_wall((cx + door_w / 2, top), (right, top), "outer_back_r")
    add_wall((left, bottom), (left, top), "outer_left")
    add_wall((right, bottom), (right, top), "outer_right")

    # Hallway walls and room separators. Room-door gaps line up with the
    # topological hall<->room edges, so graph-routed drones do not appear to
    # pass through walls.
    room_door_h = 2.7
    room_door_gaps = [
        (
            bottom + (i + 0.5) * row_h - room_door_h / 2,
            bottom + (i + 0.5) * row_h + room_door_h / 2,
        )
        for i in range(rows)
    ]
    add_vertical_wall_with_gaps(hall_x0, bottom, top, room_door_gaps, "hall_left")
    add_vertical_wall_with_gaps(hall_x1, bottom, top, room_door_gaps, "hall_right")
    for i in range(1, rows):
        y = bottom + i * row_h
        add_wall((left, y), (hall_x0, y), f"left_room_sep_{i}")
        add_wall((hall_x1, y), (right, y), f"right_room_sep_{i}")

    nodes: dict[str, np.ndarray] = {
        "front_door": np.array([cx, bottom, 0.0]),
        "back_door": np.array([cx, top, 0.0]),
    }
    edges: dict[str, set[str]] = {nid: set() for nid in nodes}
    room_ids: list[str] = []
    prev_hall = "front_door"
    for i in range(rows):
        y = bottom + (i + 0.5) * row_h
        hall = f"hall_{i}"
        left_room = f"room_L{i}"
        right_room = f"room_R{i}"
        nodes[hall] = np.array([cx, y, 0.0])
        nodes[left_room] = np.array([(left + hall_x0) / 2, y, 0.0])
        nodes[right_room] = np.array([(right + hall_x1) / 2, y, 0.0])
        edges.setdefault(hall, set()).update({left_room, right_room, prev_hall})
        edges.setdefault(prev_hall, set()).add(hall)
        edges.setdefault(left_room, set()).add(hall)
        edges.setdefault(right_room, set()).add(hall)
        room_ids.extend([left_room, right_room])
        prev_hall = hall
    edges.setdefault(prev_hall, set()).add("back_door")
    edges.setdefault("back_door", set()).add(prev_hall)

    target_count = int(rng.integers(0, max_targets + 1))
    target_ids = tuple(sorted(rng.choice(room_ids, size=target_count, replace=False).tolist()))
    return SmoothFloorplan(
        home_pos=home,
        building_center=center,
        walls=tuple(walls),
        doors={
            "front_door": nodes["front_door"].copy(),
            "back_door": nodes["back_door"].copy(),
        },
        nodes={k: v.copy() for k, v in nodes.items()},
        edges={k: tuple(sorted(v)) for k, v in edges.items()},
        room_ids=tuple(room_ids),
        target_node_ids=target_ids,
    )


def _nearest_node(plan: SmoothFloorplan, pos: np.ndarray, candidates: set[str] | None = None) -> str:
    ids = sorted(candidates if candidates is not None else set(plan.nodes))
    return min(ids, key=lambda nid: (float(np.linalg.norm(plan.nodes[nid] - pos)), nid))


def _shortest_node_path(plan: SmoothFloorplan, start: str, goal: str, allowed: set[str]) -> list[str]:
    from collections import deque

    if start not in allowed or goal not in allowed:
        return []
    q: deque[str] = deque([start])
    parent: dict[str, str | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nb in plan.edges.get(cur, ()):
            if nb not in allowed or nb in parent:
                continue
            parent[nb] = cur
            q.append(nb)
    if goal not in parent:
        return []
    path = []
    cur: str | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def _smooth_nodes_from_observations(
    plan: SmoothFloorplan,
    observations: list[BuildingObservation],
    discovered_nodes: set[str],
) -> set[str]:
    known = set(discovered_nodes)
    for obs in observations:
        if obs.kind in {ObservationKind.BUILDING, ObservationKind.CLEAR}:
            continue
        nearest = _nearest_node(plan, obs.pos)
        if float(np.linalg.norm(plan.nodes[nearest] - obs.pos)) <= 2.0:
            known.add(nearest)
    return known


def _build_smooth_shared_map(
    plan: SmoothFloorplan,
    observations: list[BuildingObservation],
    discovered_nodes: set[str],
) -> SmoothSharedMap:
    """Build the static map every drone can derive from MAP observations."""
    known_nodes = frozenset(_smooth_nodes_from_observations(plan, observations, discovered_nodes))
    known_edges = {
        node_id: tuple(nb for nb in plan.edges.get(node_id, ()) if nb in known_nodes)
        for node_id in sorted(known_nodes)
    }
    known_doors = frozenset(node_id for node_id in known_nodes if node_id in plan.doors)
    return SmoothSharedMap(
        building_known=any(obs.kind == ObservationKind.BUILDING for obs in observations),
        clear_seen=any(obs.kind == ObservationKind.CLEAR for obs in observations),
        known_nodes=known_nodes,
        known_doors=known_doors,
        known_edges=known_edges,
        known_wall_ids=_wall_ids_for_nodes(plan, set(known_nodes)),
    )


def _shortest_known_path(shared_map: SmoothSharedMap, start: str, goal: str) -> list[str]:
    from collections import deque

    if start not in shared_map.known_nodes or goal not in shared_map.known_nodes:
        return []
    q: deque[str] = deque([start])
    parent: dict[str, str | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nb in shared_map.known_edges.get(cur, ()):
            if nb in parent:
                continue
            parent[nb] = cur
            q.append(nb)
    if goal not in parent:
        return []
    path = []
    cur: str | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def _smooth_bounds(plan: SmoothFloorplan, *, margin: float = 0.0) -> tuple[float, float, float, float]:
    pts = np.array([p for a, b, _ in plan.walls for p in (a, b)], dtype=np.float64)
    x_min, y_min = pts[:, :2].min(axis=0) - margin
    x_max, y_max = pts[:, :2].max(axis=0) + margin
    return float(x_min), float(x_max), float(y_min), float(y_max)


def _inside_smooth_building(plan: SmoothFloorplan, pos: np.ndarray, *, margin: float = 0.0) -> bool:
    x_min, x_max, y_min, y_max = _smooth_bounds(plan, margin=margin)
    return x_min <= float(pos[0]) <= x_max and y_min <= float(pos[1]) <= y_max


def _distance_to_smooth_walls(plan: SmoothFloorplan, pos: np.ndarray) -> float:
    return min(_distance_to_segment(pos, a, b) for a, b, _ in plan.walls)


def _perimeter_vertices(plan: SmoothFloorplan, *, margin: float = 2.6) -> list[np.ndarray]:
    x_min, x_max, y_min, y_max = _smooth_bounds(plan, margin=margin)
    return [
        np.array([x_min, y_min, 0.0]),
        np.array([x_max, y_min, 0.0]),
        np.array([x_max, y_max, 0.0]),
        np.array([x_min, y_max, 0.0]),
    ]


def _perimeter_lengths(vertices: list[np.ndarray]) -> tuple[list[float], float]:
    lengths = []
    total = 0.0
    for idx, a in enumerate(vertices):
        b = vertices[(idx + 1) % len(vertices)]
        total += float(np.linalg.norm(b - a))
        lengths.append(total)
    return lengths, total


def _perimeter_point_at(plan: SmoothFloorplan, distance_m: float, *, margin: float = 2.6) -> np.ndarray:
    vertices = _perimeter_vertices(plan, margin=margin)
    cumulative, total = _perimeter_lengths(vertices)
    d = float(distance_m) % total
    prev_total = 0.0
    for idx, seg_total in enumerate(cumulative):
        a = vertices[idx]
        b = vertices[(idx + 1) % len(vertices)]
        seg_len = seg_total - prev_total
        if d <= seg_total:
            frac = (d - prev_total) / max(1e-9, seg_len)
            return a * (1.0 - frac) + b * frac
        prev_total = seg_total
    return vertices[0].copy()


def _perimeter_projection(plan: SmoothFloorplan, pos: np.ndarray, *, margin: float = 2.6) -> tuple[np.ndarray, float, float]:
    vertices = _perimeter_vertices(plan, margin=margin)
    best_point = vertices[0]
    best_distance_along = 0.0
    best_distance = float("inf")
    distance_along = 0.0
    for idx, a in enumerate(vertices):
        b = vertices[(idx + 1) % len(vertices)]
        ab = b - a
        seg_len = float(np.linalg.norm(ab))
        if seg_len < 1e-9:
            continue
        t = max(0.0, min(1.0, float(np.dot(pos - a, ab) / (seg_len * seg_len))))
        point = a + ab * t
        distance = float(np.linalg.norm(pos - point))
        if distance < best_distance:
            best_point = point
            best_distance = distance
            best_distance_along = distance_along + seg_len * t
        distance_along += seg_len
    return best_point.copy(), best_distance_along, best_distance


def _perimeter_search_goal(
    plan: SmoothFloorplan,
    drone_id: int,
    frame: int,
    n_drones: int,
) -> np.ndarray:
    vertices = _perimeter_vertices(plan)
    _, total = _perimeter_lengths(vertices)
    phase = total * (drone_id / max(1, n_drones))
    progress = phase + frame * 0.42
    return _perimeter_point_at(plan, progress)


def _door_exit_point(plan: SmoothFloorplan, door_id: str, *, margin: float = 2.6) -> np.ndarray:
    door = plan.nodes[door_id]
    _, _, y_min, y_max = _smooth_bounds(plan, margin=margin)
    bottom_dist = abs(float(door[1]) - _smooth_bounds(plan, margin=0.0)[2])
    top_dist = abs(float(door[1]) - _smooth_bounds(plan, margin=0.0)[3])
    if bottom_dist <= top_dist:
        return np.array([door[0], y_min, 0.0], dtype=np.float64)
    return np.array([door[0], y_max, 0.0], dtype=np.float64)


def _route_exterior_perimeter_step(
    plan: SmoothFloorplan,
    pos: np.ndarray,
    goal: np.ndarray,
    speed: float,
) -> np.ndarray:
    on_perimeter, cur_s, cur_d = _perimeter_projection(plan, pos)
    goal_on_perimeter, goal_s, _ = _perimeter_projection(plan, goal)
    if cur_d > speed * 1.8:
        return _move_toward_with_walls(plan, pos, on_perimeter, speed, entry_allowed=False)
    _, total = _perimeter_lengths(_perimeter_vertices(plan))
    clockwise = (goal_s - cur_s) % total
    counter = (cur_s - goal_s) % total
    if min(clockwise, counter) <= speed:
        return _move_toward_with_walls(plan, pos, goal_on_perimeter, speed, entry_allowed=False)
    next_s = cur_s + speed if clockwise <= counter else cur_s - speed
    return _perimeter_point_at(plan, next_s)


def _route_exterior_home_step(
    plan: SmoothFloorplan,
    pos: np.ndarray,
    home: np.ndarray,
    speed: float,
) -> np.ndarray:
    if _candidate_respects_walls(plan, pos, home, entry_allowed=False):
        return _move_toward_with_walls(plan, pos, home, speed, entry_allowed=False)
    return _route_exterior_perimeter_step(plan, pos, home, speed)


def _target_home_slot(plan: SmoothFloorplan, target_id: str) -> np.ndarray:
    ordered = sorted(plan.target_node_ids)
    if target_id not in ordered:
        return plan.home_pos.copy()
    idx = ordered.index(target_id)
    offset = (idx - (len(ordered) - 1) / 2.0) * 3.0
    return plan.home_pos + np.array([0.0, offset, 0.0], dtype=np.float64)


def _role_mobility(role: str) -> float:
    if role == "failed":
        return 0.0
    if role == "sentinel":
        return 0.05
    if role == "extractor":
        return 0.65
    return 1.0


def _apply_collision_avoidance(
    plan: SmoothFloorplan,
    positions: np.ndarray,
    roles: tuple[str, ...],
    shared_map: SmoothSharedMap,
    *,
    min_distance_m: float = 0.95,
    previous_positions: np.ndarray | None = None,
) -> np.ndarray:
    adjusted = positions.copy()
    entry_allowed = bool(shared_map.known_doors)
    previous = previous_positions if previous_positions is not None else positions
    for _ in range(2):
        for i in range(len(adjusted)):
            for j in range(i + 1, len(adjusted)):
                diff = adjusted[j] - adjusted[i]
                dist = float(np.linalg.norm(diff))
                if dist >= min_distance_m:
                    continue
                if dist < 1e-9:
                    theta = 2.0 * np.pi * (((i + 1) * 37 + (j + 1) * 17) % 97) / 97.0
                    direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
                    dist = 0.0
                else:
                    direction = diff / dist
                correction = (min_distance_m - dist) + 1e-6
                wi = _role_mobility(roles[i] if i < len(roles) else "worker")
                wj = _role_mobility(roles[j] if j < len(roles) else "worker")
                total = wi + wj
                if total <= 1e-9:
                    continue
                cand_i = adjusted[i] - direction * correction * (wi / total)
                cand_j = adjusted[j] + direction * correction * (wj / total)
                if (
                    _candidate_respects_walls(plan, adjusted[i], cand_i, entry_allowed=entry_allowed)
                    and _candidate_respects_walls(plan, previous[i], cand_i, entry_allowed=entry_allowed)
                ):
                    adjusted[i] = cand_i
                if (
                    _candidate_respects_walls(plan, adjusted[j], cand_j, entry_allowed=entry_allowed)
                    and _candidate_respects_walls(plan, previous[j], cand_j, entry_allowed=entry_allowed)
                ):
                    adjusted[j] = cand_j
    return adjusted


def _candidate_respects_walls(
    plan: SmoothFloorplan,
    pos: np.ndarray,
    candidate: np.ndarray,
    *,
    entry_allowed: bool,
) -> bool:
    if _segment_crosses_wall(plan, pos, candidate, allow_wall_escape=True):
        return False
    was_inside = _inside_smooth_building(plan, pos, margin=-0.2)
    would_be_inside = _inside_smooth_building(plan, candidate, margin=-0.2)
    if was_inside != would_be_inside and not _segment_near_door(plan, pos, candidate):
        return False
    if not entry_allowed:
        if would_be_inside and not was_inside:
            return False
    return True


def _segment_near_door(plan: SmoothFloorplan, a: np.ndarray, b: np.ndarray) -> bool:
    return any(_distance_to_segment(door_pos, a, b) <= 2.3 for door_pos in plan.doors.values())


def _move_toward_with_walls(
    plan: SmoothFloorplan,
    pos: np.ndarray,
    target: np.ndarray,
    speed: float,
    *,
    entry_allowed: bool,
) -> np.ndarray:
    direct = _move_toward(pos, target, speed)
    if _candidate_respects_walls(plan, pos, direct, entry_allowed=entry_allowed):
        return direct

    diff = target - pos
    dist = float(np.linalg.norm(diff))
    if dist < 1e-9:
        return pos.copy()
    base_angle = float(np.arctan2(diff[1], diff[0]))
    direction = diff / dist
    angle_offsets = [
        15, -15, 30, -30, 45, -45, 60, -60, 90, -90,
        120, -120, 150, -150, 180,
    ]
    candidates: list[tuple[float, np.ndarray]] = []
    for speed_scale in (1.0, 0.5, 0.25):
        step_speed = speed * speed_scale
        for offset_deg in angle_offsets:
            theta = base_angle + np.deg2rad(offset_deg)
            step = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64) * step_speed
            candidate = pos + step
            if not _candidate_respects_walls(plan, pos, candidate, entry_allowed=entry_allowed):
                continue
            progress = float(np.dot(step, direction))
            candidates.append((progress, candidate))
    if candidates:
        candidates.sort(key=lambda item: (item[0], -float(np.linalg.norm(item[1] - target))))
        return candidates[-1][1].copy()
    return pos.copy()


def _nearest_allowed_node(
    plan: SmoothFloorplan,
    pos: np.ndarray,
    allowed: set[str],
) -> tuple[str, float]:
    node_id = _nearest_node(plan, pos, allowed)
    return node_id, float(np.linalg.norm(plan.nodes[node_id] - pos))


def _interior_anchor_node(
    plan: SmoothFloorplan,
    pos: np.ndarray,
    allowed: set[str],
) -> str:
    cx = float(plan.nodes["front_door"][0])
    room_side_margin = 2.2
    if float(pos[0]) < cx - room_side_margin:
        left_rooms = {node_id for node_id in allowed if node_id.startswith("room_L")}
        if left_rooms:
            return _nearest_node(plan, pos, left_rooms)
    if float(pos[0]) > cx + room_side_margin:
        right_rooms = {node_id for node_id in allowed if node_id.startswith("room_R")}
        if right_rooms:
            return _nearest_node(plan, pos, right_rooms)
    return _nearest_node(plan, pos, allowed)


def _containing_smooth_cell_node(
    plan: SmoothFloorplan,
    pos: np.ndarray,
    allowed: set[str],
) -> str:
    cx = float(plan.nodes["front_door"][0])
    hall_x0 = cx - 3.0
    hall_x1 = cx + 3.0
    hall_rows = sorted(
        (node_id for node_id in plan.nodes if node_id.startswith("hall_")),
        key=lambda node_id: float(plan.nodes[node_id][1]),
    )
    if not hall_rows:
        return _nearest_node(plan, pos, allowed)
    row_id = min(hall_rows, key=lambda node_id: (abs(float(plan.nodes[node_id][1] - pos[1])), node_id))
    row = row_id.split("_", 1)[1]
    if float(pos[0]) < hall_x0:
        candidate = f"room_L{row}"
    elif float(pos[0]) > hall_x1:
        candidate = f"room_R{row}"
    else:
        candidate = row_id
    if candidate in allowed:
        return candidate
    return _nearest_node(plan, pos, allowed)


def _route_visual_step(
    plan: SmoothFloorplan,
    pos: np.ndarray,
    target: np.ndarray,
    shared_map: SmoothSharedMap,
    speed: float,
) -> np.ndarray:
    """Move one visual step without cutting across known building walls.

    The mission target is still the consensus-computed point. This helper
    only turns that point into a plausible visual waypoint through the
    discovered/observed floorplan graph.
    """
    entry_allowed = bool(shared_map.known_doors)
    if not shared_map.known_nodes:
        return _move_toward_with_walls(plan, pos, target, speed, entry_allowed=entry_allowed)

    allowed = set(shared_map.known_nodes)
    known_doors = sorted(shared_map.known_doors)
    target_node, target_dist = _nearest_allowed_node(plan, target, allowed)
    inside = _inside_smooth_building(plan, pos, margin=0.4)
    target_inside = _inside_smooth_building(plan, target, margin=-0.2)
    if (
        shared_map.building_known
        and not inside
        and not target_inside
        and not _candidate_respects_walls(
            plan,
            pos,
            target,
            entry_allowed=entry_allowed,
        )
    ):
        return _route_exterior_perimeter_step(plan, pos, target, speed)
    if known_doors and inside and not target_inside:
        target_node = min(known_doors, key=lambda nid: (float(np.linalg.norm(plan.nodes[nid] - pos)), nid))
        target_dist = 0.0

    if known_doors and not inside and target_inside:
        entry_id = min(
            known_doors,
            key=lambda nid: (float(np.linalg.norm(_door_exit_point(plan, nid) - pos)), nid),
        )
        entry_stage = _door_exit_point(plan, entry_id)
        door_pos = plan.nodes[entry_id]
        if float(np.linalg.norm(pos - entry_stage)) > 2.4:
            return _route_exterior_perimeter_step(plan, pos, entry_stage, speed)
        if float(np.linalg.norm(pos - door_pos)) > 0.45:
            return _move_toward_with_walls(
                plan,
                pos,
                door_pos,
                speed,
                entry_allowed=entry_allowed,
            )
    elif target_dist > 3.5:
        return _move_toward_with_walls(plan, pos, target, speed, entry_allowed=entry_allowed)

    wall_distance = _distance_to_smooth_walls(plan, pos)
    cx = float(plan.nodes["front_door"][0])
    room_side = float(pos[0]) < cx - 3.0 or float(pos[0]) > cx + 3.0
    needs_cell_anchor = wall_distance < 0.9 or room_side
    if inside and needs_cell_anchor:
        start_node = _containing_smooth_cell_node(plan, pos, allowed)
        start_dist = float(np.linalg.norm(plan.nodes[start_node] - pos))
    else:
        start_node, start_dist = _nearest_allowed_node(plan, pos, allowed)
    if known_doors and inside and not target_inside:
        door_pos = plan.nodes[target_node]
        if float(np.linalg.norm(pos - door_pos)) > 0.8:
            exit_start = start_node
            if start_node == target_node:
                interior_nodes = allowed - set(plan.doors)
                if interior_nodes:
                    exit_start = _interior_anchor_node(plan, pos, interior_nodes)
            if exit_start != target_node:
                path_to_door = _shortest_known_path(shared_map, exit_start, target_node)
                if len(path_to_door) >= 2:
                    return _move_toward_with_walls(
                        plan,
                        pos,
                        plan.nodes[path_to_door[1]],
                        speed,
                        entry_allowed=entry_allowed,
                    )
            return _move_toward_with_walls(plan, pos, door_pos, speed, entry_allowed=entry_allowed)
        return _move_toward_with_walls(
            plan,
            pos,
            _door_exit_point(plan, target_node),
            speed,
            entry_allowed=entry_allowed,
        )

    if start_node == target_node:
        return _move_toward_with_walls(plan, pos, target, speed, entry_allowed=entry_allowed)

    path = _shortest_known_path(shared_map, start_node, target_node)
    if len(path) >= 2:
        next_waypoint = plan.nodes[path[1]]
        if (
            inside
            and needs_cell_anchor
            and start_dist > 1.2
            and not _candidate_respects_walls(
                plan,
                pos,
                _move_toward(pos, next_waypoint, speed),
                entry_allowed=entry_allowed,
            )
        ):
            return _move_toward_with_walls(
                plan,
                pos,
                plan.nodes[start_node],
                speed,
                entry_allowed=entry_allowed,
            )
        return _move_toward_with_walls(
            plan,
            pos,
            next_waypoint,
            speed,
            entry_allowed=entry_allowed,
        )
    if start_dist > 1.2:
        return _move_toward_with_walls(
            plan,
            pos,
            plan.nodes[start_node],
            speed,
            entry_allowed=entry_allowed,
        )
    return _move_toward_with_walls(plan, pos, target, speed, entry_allowed=entry_allowed)


def _smooth_observation_kind(plan: SmoothFloorplan, node_id: str) -> ObservationKind:
    if node_id in plan.doors:
        return ObservationKind.ENTRANCE
    if node_id in plan.target_node_ids:
        return ObservationKind.TARGET
    if node_id in plan.room_ids:
        return ObservationKind.ROOM
    if len(plan.edges.get(node_id, ())) >= 3:
        return ObservationKind.INTERSECTION
    return ObservationKind.FRONTIER


def _target_discoverers(
    plan: SmoothFloorplan,
    observations: list[BuildingObservation],
    target_ids: set[str],
) -> dict[str, int]:
    discoverers: dict[str, int] = {}
    for target_id in sorted(target_ids):
        node_pos = plan.nodes[target_id]
        candidates = [
            obs for obs in observations
            if obs.kind == ObservationKind.TARGET
            and float(np.linalg.norm(obs.pos - node_pos)) <= 2.0
        ]
        if not candidates:
            continue
        first = min(candidates, key=lambda obs: (obs.tick, obs.drone_id))
        discoverers[target_id] = int(first.drone_id)
    return discoverers


def _assign_extractor_groups(
    plan: SmoothFloorplan,
    target_ids: set[str],
    observations: list[BuildingObservation],
    *,
    n_drones: int,
    fixed_sentinels: set[int],
    available_ids: set[int] | None = None,
    drone_positions: np.ndarray | None = None,
    group_size: int = 3,
) -> dict[str, tuple[int, ...]]:
    """Deterministically assign transport groups from shared MAP facts."""
    discoverers = _target_discoverers(plan, observations, target_ids)
    pool = available_ids if available_ids is not None else set(range(n_drones))
    available = [i for i in sorted(pool, reverse=True) if i not in fixed_sentinels]
    groups: dict[str, tuple[int, ...]] = {}
    for target_id in sorted(target_ids):
        group: list[int] = []
        discoverer = discoverers.get(target_id)
        if discoverer in available:
            group.append(discoverer)
            available.remove(discoverer)
        while available and len(group) < group_size:
            if drone_positions is None:
                group.append(available.pop(0))
                continue
            target_pos = plan.nodes[target_id]
            chosen = min(
                available,
                key=lambda did: (
                    round(float(np.linalg.norm(drone_positions[did] - target_pos)), 9),
                    -did,
                ),
            )
            group.append(chosen)
            available.remove(chosen)
        groups[target_id] = tuple(group)
    return groups


def _wall_ids_for_nodes(plan: SmoothFloorplan, node_ids: set[str]) -> frozenset[str]:
    if not node_ids:
        return frozenset()
    pts = np.array([plan.nodes[nid] for nid in node_ids if nid in plan.nodes], dtype=np.float64)
    if len(pts) == 0:
        return frozenset()
    known = set()
    for a, b, wid in plan.walls:
        mid = (a + b) * 0.5
        if float(np.linalg.norm(pts[:, :2] - mid[:2], axis=1).min()) <= 9.0:
            known.add(wid)
    return frozenset(known)


def _search_target(home: np.ndarray, drone_id: int, frame: int, n_drones: int) -> np.ndarray:
    spread = np.deg2rad(58.0)
    frac = 0.5 if n_drones <= 1 else drone_id / (n_drones - 1)
    angle = -spread / 2 + spread * frac
    radius = min(60.0, 5.0 + 0.55 * frame)
    wobble = 2.0 * np.sin(0.05 * frame + drone_id)
    return home + np.array([radius * np.cos(angle), radius * np.sin(angle) + wobble, 0.0])


def _move_toward(pos: np.ndarray, target: np.ndarray, speed: float) -> np.ndarray:
    diff = target - pos
    dist = float(np.linalg.norm(diff))
    if dist <= speed or dist < 1e-9:
        return target.copy()
    return pos + diff / dist * speed


def _escort_offsets(n: int) -> list[np.ndarray]:
    if n <= 0:
        return []
    radius = 1.2
    return [
        np.array([radius * np.cos(2 * np.pi * i / n), radius * np.sin(2 * np.pi * i / n), 0.0])
        for i in range(n)
    ]


def _escort_offsets_for_position(plan: SmoothFloorplan, pos: np.ndarray, n: int) -> list[np.ndarray]:
    if n <= 0:
        return []
    if _inside_smooth_building(plan, pos, margin=-0.2):
        return _escort_offsets(n)
    x_min, x_max, y_min, y_max = _smooth_bounds(plan, margin=0.0)
    radius = 1.2
    if float(pos[1]) <= y_min + 0.8:
        angles = np.linspace(-140.0, -40.0, n)
    elif float(pos[1]) >= y_max - 0.8:
        angles = np.linspace(40.0, 140.0, n)
    elif float(pos[0]) <= x_min + 0.8:
        angles = np.linspace(140.0, 220.0, n)
    elif float(pos[0]) >= x_max - 0.8:
        angles = np.linspace(-40.0, 40.0, n)
    else:
        return _escort_offsets(n)
    return [
        np.array([radius * np.cos(np.deg2rad(angle)), radius * np.sin(np.deg2rad(angle)), 0.0])
        for angle in angles
    ]


def _segments_intersect_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    if abs(o1) <= 1e-9 and on_segment(a, c, b):
        return True
    if abs(o2) <= 1e-9 and on_segment(a, d, b):
        return True
    if abs(o3) <= 1e-9 and on_segment(c, a, d):
        return True
    if abs(o4) <= 1e-9 and on_segment(c, b, d):
        return True
    return False


def _segments_cross_strict_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    eps = 1e-9
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return o1 * o2 < -eps and o3 * o4 < -eps


def _segment_crosses_wall(
    plan: SmoothFloorplan,
    a: np.ndarray,
    b: np.ndarray,
    *,
    allow_wall_escape: bool = False,
) -> bool:
    for wall_a, wall_b, _ in plan.walls:
        if not _segments_intersect_2d(a, b, wall_a, wall_b):
            continue
        if (
            allow_wall_escape
            and not _segments_cross_strict_2d(a, b, wall_a, wall_b)
            and _distance_to_segment(a, wall_a, wall_b) <= 0.08
            and _distance_to_segment(b, wall_a, wall_b) > 0.18
        ):
            continue
        return True
    return False


def _stress_comm_link_clear(plan: SmoothFloorplan, a: np.ndarray, b: np.ndarray) -> bool:
    if not _segment_crosses_wall(plan, a, b):
        return True
    return _segment_near_door(plan, a, b)


def _stress_comm_range_limit(plan: SmoothFloorplan, a: np.ndarray, b: np.ndarray, *, home_link: bool) -> float:
    if home_link:
        return 70.0
    a_inside = _inside_smooth_building(plan, a, margin=-0.2)
    b_inside = _inside_smooth_building(plan, b, margin=-0.2)
    return 14.0 if a_inside or b_inside else 34.0


def _stress_connected_drones(
    plan: SmoothFloorplan,
    drones: np.ndarray,
    alive: np.ndarray,
    relay_ids: set[int],
) -> frozenset[int]:
    """Drones whose MAP observations can reach home through relay sentinels."""
    from collections import deque

    alive_ids = {int(i) for i, ok in enumerate(alive) if ok}
    connected: set[int] = set()
    q: deque[int] = deque()

    for did in sorted(alive_ids):
        pos = drones[did]
        dist = float(np.linalg.norm(pos - plan.home_pos))
        if dist <= _stress_comm_range_limit(plan, pos, plan.home_pos, home_link=True) and _stress_comm_link_clear(
            plan,
            pos,
            plan.home_pos,
        ):
            connected.add(did)
            q.append(did)

    while q:
        cur = q.popleft()
        for did in sorted(alive_ids - connected):
            if cur not in relay_ids and did not in relay_ids:
                continue
            dist = float(np.linalg.norm(drones[cur] - drones[did]))
            if dist > _stress_comm_range_limit(plan, drones[cur], drones[did], home_link=False):
                continue
            if not _stress_comm_link_clear(plan, drones[cur], drones[did]):
                continue
            connected.add(did)
            q.append(did)

    return frozenset(connected)


def simulate_smooth_building_mission(
    seed: int,
    *,
    n_drones: int = 15,
    n_frames: int = 220,
    sensor_range_m: float = 5.0,
    map_interval_frames: int = 6,
    stress: bool = False,
    failure_frame: int = 170,
) -> tuple[SmoothFloorplan, list[SmoothFrame]]:
    """Continuous visual harness: no prior map, local discovery only."""
    plan = generate_smooth_floorplan(seed)
    rng = np.random.default_rng(seed + 1000)
    drones = np.array([
        plan.home_pos + np.array([rng.normal(scale=0.7), rng.normal(scale=0.7), 0.0])
        for _ in range(n_drones)
    ], dtype=np.float64)
    shared_obs: list[BuildingObservation] = []
    pending_obs: list[BuildingObservation] = []
    pending_obs_by_drone: dict[int, list[BuildingObservation]] = {i: [] for i in range(n_drones)}
    discovered_nodes: set[str] = set()
    known_walls: frozenset[str] = frozenset()
    alive = np.ones(n_drones, dtype=bool)
    failed_drone_ids: set[int] = set()
    failure_applied = False
    last_connected_ids: frozenset[int] = frozenset(range(n_drones))
    target_positions: dict[str, np.ndarray] = {
        nid: plan.nodes[nid].copy() for nid in plan.target_node_ids
    }
    target_progress: dict[str, int] = {nid: 0 for nid in plan.target_node_ids}
    target_escorts: dict[str, tuple[int, ...]] = {}
    target_boarded: dict[str, bool] = {nid: False for nid in plan.target_node_ids}
    target_exited: dict[str, bool] = {nid: False for nid in plan.target_node_ids}
    extracting = False
    targets_home = False
    clear_announced = False
    returned_home = False
    frames: list[SmoothFrame] = []
    current_payload: dict[str, Any] | None = None
    current_phase = "find_entrance"

    def add_pending(did: int, obs: BuildingObservation) -> None:
        if stress:
            pending_obs_by_drone.setdefault(did, []).append(obs)
        else:
            pending_obs.append(obs)

    def pending_for(did: int) -> list[BuildingObservation]:
        if stress:
            return shared_obs + pending_obs_by_drone.get(did, [])
        return shared_obs + pending_obs

    def local_map(did: int) -> SmoothSharedMap:
        obs = pending_for(did)
        if stress:
            nodes = _smooth_nodes_from_observations(plan, obs, set())
            return _build_smooth_shared_map(plan, obs, nodes)
        return _build_smooth_shared_map(plan, obs, discovered_nodes)

    def global_map(include_pending: bool = False) -> SmoothSharedMap:
        if stress:
            obs = shared_obs
            if include_pending:
                obs = shared_obs + [
                    item
                    for did in sorted(last_connected_ids)
                    for item in pending_obs_by_drone.get(did, [])
                ]
            nodes = _smooth_nodes_from_observations(plan, obs, set())
            return _build_smooth_shared_map(plan, obs, nodes)
        obs = shared_obs + pending_obs if include_pending else shared_obs
        return _build_smooth_shared_map(plan, obs, discovered_nodes)

    def share_pending_on_map_tick(frame_idx: int) -> None:
        nonlocal pending_obs
        if frame_idx % map_interval_frames != 0:
            return
        if stress:
            delivered: list[BuildingObservation] = []
            for did in sorted(last_connected_ids):
                delivered.extend(pending_obs_by_drone.get(did, []))
                pending_obs_by_drone[did] = []
            if delivered:
                shared_obs.extend(delivered)
        elif pending_obs:
            shared_obs.extend(pending_obs)
            pending_obs = []

    def confirmed_nodes_from_observations(
        observations: list[BuildingObservation],
        kinds: set[ObservationKind],
    ) -> set[str]:
        confirmed: set[str] = set()
        for obs in observations:
            if obs.kind not in kinds:
                continue
            nearest = _nearest_node(plan, obs.pos)
            if float(np.linalg.norm(plan.nodes[nearest] - obs.pos)) <= 2.0:
                confirmed.add(nearest)
        return confirmed

    for frame in range(n_frames):
        alive_ids = {int(i) for i, ok in enumerate(alive) if ok}
        drone_dicts = [{"id": i, "pos": drones[i].copy()} for i in sorted(alive_ids)]
        shared_map = global_map(include_pending=not stress)

        # Local sensing. Observations are only shared on MAP ticks below.
        for did, pos in enumerate(drones):
            if not alive[did]:
                continue
            locally_observed_map = local_map(did)
            if not locally_observed_map.building_known and _distance_to_smooth_walls(plan, pos) <= sensor_range_m:
                add_pending(did, BuildingObservation(frame, did, ObservationKind.BUILDING, plan.building_center.copy()))
                locally_observed_map = local_map(did)
            if len(locally_observed_map.known_doors) < len(plan.doors):
                for door_id, door_pos in plan.doors.items():
                    if door_id in discovered_nodes:
                        continue
                    if float(np.linalg.norm(pos - door_pos)) <= sensor_range_m:
                        add_pending(did, BuildingObservation(frame, did, ObservationKind.ENTRANCE, door_pos.copy()))
                        discovered_nodes.add(door_id)
                        for nb in plan.edges[door_id]:
                            add_pending(did, BuildingObservation(frame, did, ObservationKind.FRONTIER, plan.nodes[nb].copy()))
                        break
            locally_observed_map = local_map(did)
            interior_candidates = set(locally_observed_map.known_nodes) - set(plan.doors)
            nearest = _nearest_node(plan, pos, interior_candidates) if interior_candidates else None
            if (
                nearest is not None
                and float(np.linalg.norm(pos - plan.nodes[nearest])) <= sensor_range_m
                and nearest not in discovered_nodes
            ):
                kind = _smooth_observation_kind(plan, nearest)
                add_pending(did, BuildingObservation(frame, did, kind, plan.nodes[nearest].copy()))
                discovered_nodes.add(nearest)
                for nb in plan.edges[nearest]:
                    if nb not in discovered_nodes:
                        add_pending(did, BuildingObservation(frame, did, ObservationKind.FRONTIER, plan.nodes[nb].copy()))

        share_pending_on_map_tick(frame)
        shared_map = global_map(include_pending=not stress)

        confirmed_obs = shared_obs if stress else shared_obs + pending_obs
        confirmed_rooms = confirmed_nodes_from_observations(
            confirmed_obs,
            {ObservationKind.ROOM, ObservationKind.TARGET},
        )
        target_seen = {
            nid for nid in plan.target_node_ids
            if nid in confirmed_nodes_from_observations(confirmed_obs, {ObservationKind.TARGET})
        }
        all_rooms_touched = all(room_id in confirmed_rooms for room_id in plan.room_ids)
        if all_rooms_touched and not clear_announced:
            announcer = min(alive_ids) if alive_ids else 0
            add_pending(announcer, BuildingObservation(frame, announcer, ObservationKind.CLEAR, plan.building_center.copy()))
            clear_announced = True

        share_pending_on_map_tick(frame)
        shared_map = global_map(include_pending=not stress)

        clear_shared = shared_map.clear_seen
        if clear_shared and target_seen:
            extracting = True

        if frame % map_interval_frames == 0 or current_payload is None:
            building_plan = plan_from_observations(
                drone_dicts,
                shared_obs,
                home_pos=plan.home_pos,
                max_targets=3,
                extraction_started=extracting,
                extracted=targets_home,
                returned_home=returned_home,
            )
            current_payload = building_plan.payload
            current_phase = building_plan.phase.value
            if extracting and target_seen and not targets_home:
                current_phase = "extract"
                assignments = current_payload.get("sentinel_assignments", {})
                fixed_sentinels = {
                    int(did)
                    for feature_id, did in assignments.items()
                    if not str(feature_id).startswith("target")
                }
                if set(target_escorts) != target_seen:
                    target_escorts = _assign_extractor_groups(
                        plan,
                        target_seen,
                        shared_obs,
                        n_drones=n_drones,
                        fixed_sentinels=fixed_sentinels,
                        available_ids=alive_ids,
                        drone_positions=drones,
                    )
                    for target_id in target_seen:
                        target_boarded.setdefault(target_id, False)
                        target_exited.setdefault(target_id, False)

        known_walls = shared_map.known_wall_ids
        roles = ["worker"] * n_drones
        if current_payload is not None:
            for did in current_payload.get("sentinel_assignments", {}).values():
                if 0 <= int(did) < n_drones and alive[int(did)]:
                    roles[int(did)] = "sentinel"
        if not targets_home:
            for group in target_escorts.values():
                for did in group:
                    if 0 <= did < n_drones and alive[did]:
                        roles[did] = "extractor"

        if stress and not failure_applied and frame >= failure_frame and shared_map.building_known and alive_ids:
            escorting = {did for group in target_escorts.values() for did in group}
            sentinel_candidates = [
                did for did, role in enumerate(roles)
                if role == "sentinel" and alive[did] and did not in escorting
            ]
            worker_candidates = [
                did for did, role in enumerate(roles)
                if role == "worker" and alive[did] and did not in escorting
            ]
            candidates = sentinel_candidates or worker_candidates or [did for did in sorted(alive_ids) if did not in escorting]
            if candidates:
                failed = candidates[len(candidates) // 2]
                alive[failed] = False
                failed_drone_ids.add(int(failed))
                failure_applied = True
                roles[failed] = "failed"
                if any(failed in group for group in target_escorts.values()):
                    target_escorts = {}
                    target_boarded = {nid: False for nid in target_boarded}
                current_payload = None
                alive_ids = {int(i) for i, ok in enumerate(alive) if ok}
        for did in failed_drone_ids:
            if 0 <= did < n_drones:
                roles[did] = "failed"

        # Move targets and compute per-drone goals.
        escort_goal_by_drone: dict[int, np.ndarray] = {}
        if extracting and target_escorts and not targets_home:
            allowed = set(discovered_nodes) | {"front_door"}
            for target_id, group in target_escorts.items():
                valid_group = tuple(did for did in group if 0 <= did < n_drones and alive[did])
                if (
                    len(valid_group) == len(group)
                    and len(valid_group) >= min(3, n_drones)
                    and all(float(np.linalg.norm(drones[did] - target_positions[target_id])) <= 4.2 for did in valid_group)
                ):
                    target_boarded[target_id] = True
                can_advance = (
                    target_boarded.get(target_id, False)
                    and valid_group
                    and all(float(np.linalg.norm(drones[did] - target_positions[target_id])) <= 5.0 for did in valid_group)
                )
                if can_advance:
                    path = _shortest_node_path(plan, target_id, "front_door", allowed)
                    if not path:
                        path = [target_id, "front_door"]
                    progress = target_progress[target_id]
                    if progress < len(path):
                        target_goal = plan.nodes[path[progress]]
                        target_positions[target_id] = _move_toward_with_walls(
                            plan,
                            target_positions[target_id],
                            target_goal,
                            0.42,
                            entry_allowed=True,
                        )
                        if float(np.linalg.norm(target_positions[target_id] - target_goal)) < 0.45:
                            target_progress[target_id] = min(progress + 1, len(path))
                    else:
                        exit_stage = _door_exit_point(plan, "front_door")
                        if not target_exited.get(target_id, False):
                            target_positions[target_id] = _move_toward_with_walls(
                                plan,
                                target_positions[target_id],
                                exit_stage,
                                0.42,
                                entry_allowed=True,
                            )
                            if float(np.linalg.norm(target_positions[target_id] - exit_stage)) <= 0.45:
                                target_exited[target_id] = True
                        else:
                            target_positions[target_id] = _route_exterior_home_step(
                                plan,
                                target_positions[target_id],
                                _target_home_slot(plan, target_id),
                                0.42,
                            )
                offsets = _escort_offsets_for_position(plan, target_positions[target_id], len(valid_group))
                for did, off in zip(valid_group, offsets):
                    escort_goal_by_drone[did] = target_positions[target_id] + off

        shared_map = global_map(include_pending=False)
        next_drones = drones.copy()
        for did in range(n_drones):
            if not alive[did]:
                next_drones[did] = drones[did]
                continue
            if shared_map.building_known and not shared_map.known_doors:
                target = _perimeter_search_goal(plan, did, frame, n_drones)
                next_drones[did] = _route_exterior_perimeter_step(plan, drones[did], target, 0.85)
                continue
            if did in escort_goal_by_drone:
                target = escort_goal_by_drone[did]
            elif current_payload is not None and discovered_nodes:
                target, _ = compute_mission_target(did, drone_dicts, current_payload)
            else:
                target = _search_target(plan.home_pos, did, frame, n_drones)
            if current_phase in {"complete", "return_home"}:
                home_slot = plan.home_pos + np.array([0.0, (did - n_drones / 2) * 0.45, 0.0])
                if current_phase == "return_home":
                    x_min, _, _, _ = _smooth_bounds(plan, margin=3.2)
                    if _inside_smooth_building(plan, drones[did], margin=-0.2) or float(drones[did][0]) > x_min:
                        target = np.array([x_min - 2.0, drones[did][1], 0.0], dtype=np.float64)
                    else:
                        target = home_slot
                else:
                    target = home_slot
            next_drones[did] = _route_visual_step(
                plan,
                drones[did],
                target,
                shared_map,
                0.85,
            )
        avoidance_distance = 0.35 if current_phase in {"return_home", "complete"} else 0.95
        next_drones = _apply_collision_avoidance(
            plan,
            next_drones,
            tuple(roles),
            shared_map,
            min_distance_m=avoidance_distance,
            previous_positions=drones,
        )
        for did in failed_drone_ids:
            if 0 <= did < n_drones:
                next_drones[did] = drones[did]
        if extracting and target_escorts and not targets_home:
            for target_id, group in target_escorts.items():
                if not target_boarded.get(target_id, False):
                    continue
                valid_group = tuple(did for did in group if 0 <= did < n_drones and alive[did])
                offsets = _escort_offsets_for_position(plan, target_positions[target_id], len(valid_group))
                for did, off in zip(valid_group, offsets):
                    if 0 <= did < n_drones and alive[did]:
                        next_drones[did] = target_positions[target_id] + off
        drones = next_drones

        if extracting and target_escorts and not targets_home:
            all_home = all(
                float(np.linalg.norm(pos - _target_home_slot(plan, target_id))) < 1.4
                for target_id, pos in target_positions.items()
            )
            if all_home:
                targets_home = True
                for target_id in list(target_positions):
                    target_positions[target_id] = plan.home_pos.copy()
                target_escorts = {}
                current_phase = "return_home"
        if targets_home:
            all_drones_home = all(
                (not alive[did]) or float(np.linalg.norm(pos - plan.home_pos)) < 5.0
                for did, pos in enumerate(drones)
            )
            if all_drones_home:
                returned_home = True
                current_phase = "complete"

        relay_ids = {
            did for did, role in enumerate(roles)
            if role == "sentinel" and alive[did]
        }
        connected_ids = (
            _stress_connected_drones(plan, drones, alive, relay_ids)
            if stress
            else frozenset(int(i) for i, ok in enumerate(alive) if ok)
        )
        last_connected_ids = connected_ids
        display_nodes = shared_map.known_nodes if stress else frozenset(discovered_nodes)

        frames.append(SmoothFrame(
            frame=frame,
            phase=current_phase,
            drones=drones.copy(),
            roles=tuple(roles),
            known_wall_ids=known_walls,
            discovered_node_ids=frozenset(display_nodes),
            discovered_target_ids=frozenset(target_seen),
            carried_targets={k: v.copy() for k, v in target_positions.items() if k in target_seen},
            target_escorts={k: tuple(v) for k, v in target_escorts.items()},
            failed_drone_ids=frozenset(failed_drone_ids),
            relay_drone_ids=frozenset(relay_ids),
            comm_connected_ids=connected_ids,
            comm_degraded=stress and shared_map.building_known,
        ))
        if returned_home and frame > n_frames * 0.65:
            break

    return plan, frames


def _draw_smooth_frame(ax: Any, plan: SmoothFloorplan, frame: SmoothFrame) -> None:
    from matplotlib.patches import Rectangle

    ax.clear()
    ax.set_facecolor("#ffffff")
    pts = np.array(list(plan.nodes.values()) + [plan.home_pos], dtype=np.float64)
    x_min, y_min = pts[:, :2].min(axis=0) - 8
    x_max, y_max = pts[:, :2].max(axis=0) + 8

    for a, b, wid in plan.walls:
        known = wid in frame.known_wall_ids
        ax.plot(
            [a[0], b[0]], [a[1], b[1]],
            color="#111111" if known else "#d8d8d8",
            lw=2.8 if known else 1.4,
            solid_capstyle="round",
            zorder=2 if known else 1,
        )
    for door_id, pos in plan.doors.items():
        known = door_id in frame.discovered_node_ids
        ax.add_patch(Rectangle(
            (pos[0] - 1.7, pos[1] - 0.32), 3.4, 0.64,
            facecolor="#111111" if known else "#d8d8d8",
            edgecolor="#111111" if known else "#cfcfcf",
            lw=1.0,
            zorder=4,
        ))
    for node_id, pos in plan.nodes.items():
        if node_id in plan.doors:
            continue
        discovered = node_id in frame.discovered_node_ids
        if len(plan.edges.get(node_id, ())) >= 3:
            ax.scatter([pos[0]], [pos[1]], s=70, marker="D",
                       c="#111111" if discovered else "#d8d8d8",
                       edgecolors="#111111" if discovered else "#cfcfcf", zorder=4)
        elif node_id in plan.room_ids:
            ax.scatter([pos[0]], [pos[1]], s=44, marker="o",
                       c="#f7f7f7" if discovered else "#eeeeee",
                       edgecolors="#111111" if discovered else "#d4d4d4", zorder=3)

    for target_id, pos in frame.carried_targets.items():
        ax.scatter([pos[0]], [pos[1]], s=115, marker="s", c="#d62728",
                   edgecolors="#5a0000", linewidths=1.0, zorder=8)
        group = frame.target_escorts.get(target_id, ())
        if group:
            ax.text(pos[0] + 0.9, pos[1] + 0.9, f"x{len(group)}", fontsize=8, color="#5a0000", zorder=9)

    if frame.comm_degraded and frame.relay_drone_ids:
        relay_points = [
            (did, frame.drones[did])
            for did in sorted(frame.relay_drone_ids)
            if did < len(frame.drones) and did in frame.comm_connected_ids
        ]
        for did, pos in relay_points:
            if _stress_comm_link_clear(plan, plan.home_pos, pos):
                ax.plot(
                    [plan.home_pos[0], pos[0]], [plan.home_pos[1], pos[1]],
                    color="#3f9fb5",
                    lw=0.55,
                    alpha=0.22,
                    zorder=1,
                )
        for idx, (did_a, pos_a) in enumerate(relay_points):
            for did_b, pos_b in relay_points[idx + 1:]:
                dist = float(np.linalg.norm(pos_a - pos_b))
                if dist <= _stress_comm_range_limit(plan, pos_a, pos_b, home_link=False) and _stress_comm_link_clear(
                    plan,
                    pos_a,
                    pos_b,
                ):
                    ax.plot(
                        [pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]],
                        color="#3f9fb5",
                        lw=0.75,
                        alpha=0.32,
                        zorder=1,
                    )

    role_colors = {
        "worker": "#ffffff",
        "sentinel": "#111111",
        "extractor": "#f6c744",
        "failed": "#b8b8b8",
    }
    for did, pos in enumerate(frame.drones):
        role = frame.roles[did] if did < len(frame.roles) else "worker"
        edge = "#d94848" if frame.comm_degraded and did not in frame.comm_connected_ids and role != "failed" else "#111111"
        if role == "failed":
            ax.scatter([pos[0]], [pos[1]], s=82, marker="x",
                       c="#777777", linewidths=1.4, zorder=10)
        else:
            ax.scatter([pos[0]], [pos[1]], s=72, marker="o",
                       c=role_colors.get(role, "#ffffff"),
                       edgecolors=edge, linewidths=1.1, zorder=10)
        if role == "sentinel" and frame.comm_degraded:
            ax.scatter([pos[0]], [pos[1]], s=120, marker="o",
                       facecolors="none", edgecolors="#3f9fb5", linewidths=1.1, zorder=9)
        ax.text(pos[0], pos[1], str(did), ha="center", va="center",
                fontsize=6.5, color="#ffffff" if role == "sentinel" else "#111111", zorder=11)
    ax.scatter([plan.home_pos[0]], [plan.home_pos[1]], s=140, marker="*", c="#2ca02c", zorder=9)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#f0f0f0", linewidth=0.6)
    rooms_seen = len(set(plan.room_ids) & set(frame.discovered_node_ids))
    stress_note = ""
    if frame.comm_degraded:
        stress_note = (
            f" | stress lost {len(frame.failed_drone_ids)}"
            f" | relay {len(frame.relay_drone_ids)}"
            f" | linked {len(frame.comm_connected_ids)}"
        )
    ax.set_title(
        f"frame {frame.frame} | {frame.phase} | targets found {len(frame.discovered_target_ids)} | rooms seen {rooms_seen}{stress_note}",
        fontsize=11,
    )


def render_smooth_building_animation(
    plan: SmoothFloorplan,
    frames: list[SmoothFrame],
    output_path: str,
    *,
    fps: float = 12.0,
    stride: int = 2,
) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    selected = frames[::max(1, stride)]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    images: list[Image.Image] = []
    try:
        for frame in selected:
            _draw_smooth_frame(ax, plan, frame)
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            images.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE))
    finally:
        plt.close(fig)
    duration_ms = max(40, int(round(1000.0 / max(0.1, fps))))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _tests() -> int:
    failed = 0
    home = np.array([-20.0, 0.0, 0.0])
    drones = _drones_at_home(8)

    plan = plan_from_observations(drones, sample_observations(0)[:1], home_pos=home)
    if plan.phase != MissionPhase.FIND_ENTRANCE:
        print(f"FAIL T1: expected find_entrance, got {plan.phase}")
        failed += 1

    obs = sample_observations(0)[:4]
    plan = plan_from_observations(drones, obs, home_pos=home)
    if plan.phase != MissionPhase.EXPLORE:
        print(f"FAIL T2: entrance/frontier observations should explore, got {plan.phase}")
        failed += 1
    sent = plan.payload["sentinel_assignments"]
    if sent.get("entrance_0") != 2:
        print(f"FAIL T2b: entrance discoverer drone 2 should be sentinel, got {sent}")
        failed += 1
    if 2 in plan.worker_ids:
        print(f"FAIL T2c: sentinel drone 2 should not be worker, workers={plan.worker_ids}")
        failed += 1

    obs = sample_observations(1)
    plan = plan_from_observations(drones, obs, home_pos=home)
    sent = plan.payload["sentinel_assignments"]
    if plan.phase != MissionPhase.CALL_HELP_AND_EXPLORE:
        print(f"FAIL T3: one target with frontiers should call help and keep exploring, got {plan.phase}")
        failed += 1
    if sent.get("intersection_0") != 4:
        print(f"FAIL T3b: intersection discoverer drone 4 should be sentinel, got {sent}")
        failed += 1
    if sent.get("target_0") != 5:
        print(f"FAIL T3c: target discoverer drone 5 should be sentinel, got {sent}")
        failed += 1
    if not plan.payload["help_requested"]:
        print("FAIL T3d: target detection should request help")
        failed += 1

    stationed = [
        {"id": 2, "pos": np.array([0.0, -8.0, 0.0])},
        {"id": 4, "pos": np.array([0.0, 5.0, 0.0])},
        {"id": 5, "pos": np.array([0.0, 18.0, 0.0])},
        {"id": 7, "pos": np.array([2.0, 2.0, 0.0])},
    ]
    t2, p2 = compute_mission_target(2, stationed, plan.payload)
    t4, p4 = compute_mission_target(4, stationed, plan.payload)
    t5, p5 = compute_mission_target(5, stationed, plan.payload)
    t7, p7 = compute_mission_target(7, stationed, plan.payload)
    if not (p2 and np.allclose(t2, np.array([0.0, -8.0, 0.0]))):
        print(f"FAIL T4a: entrance sentinel did not hold, target={t2} primary={p2}")
        failed += 1
    if not (p4 and np.allclose(t4, np.array([0.0, 5.0, 0.0]))):
        print(f"FAIL T4b: intersection sentinel did not hold, target={t4} primary={p4}")
        failed += 1
    if not (p5 and np.allclose(t5, np.array([0.0, 18.0, 0.0]))):
        print(f"FAIL T4c: target sentinel did not hold, target={t5} primary={p5}")
        failed += 1
    if p7:
        print(f"FAIL T4d: worker should not be primary station holder, target={t7}")
        failed += 1
    if plan.payload["sentinel_reasons"].get("intersection_0") != "downstream_drones_or_objectives":
        print(f"FAIL T4e: intersection sentinel should be purpose-labeled, got {plan.payload['sentinel_reasons']}")
        failed += 1

    plan3 = plan_from_observations(drones, sample_observations(3), home_pos=home)
    if plan3.phase != MissionPhase.CALL_HELP_AND_EXPLORE:
        print(f"FAIL T5: uncleared building should keep exploring after three targets, got {plan3.phase}")
        failed += 1
    if plan3.payload["target_count_known"] != 3:
        print(f"FAIL T5b: expected three known targets, got {plan3.payload['target_count_known']}")
        failed += 1
    plan3_clear = plan_from_observations(drones, sample_observations(3, clear=True), home_pos=home)
    if plan3_clear.phase != MissionPhase.EXTRACT:
        print(f"FAIL T5c: clear building with targets should extract, got {plan3_clear.phase}")
        failed += 1
    if any(fid.startswith("target") for fid in plan3_clear.payload["sentinel_assignments"]):
        print(f"FAIL T5d: targets should release from sentinel duty during extraction, got {plan3_clear.payload['sentinel_assignments']}")
        failed += 1

    plan0_clear = plan_from_observations(drones, sample_observations(0, clear=True), home_pos=home)
    if plan0_clear.phase != MissionPhase.RETURN_HOME:
        print(f"FAIL T6: no-target clear sample should return home, got {plan0_clear.phase}")
        failed += 1
    returned = plan_from_observations(
        drones, sample_observations(0, clear=True), home_pos=home, extracted=True,
    )
    if returned.phase != MissionPhase.RETURN_HOME:
        print(f"FAIL T6b: extracted mission should return home, got {returned.phase}")
        failed += 1
    if returned.payload["sentinel_assignments"]:
        print(f"FAIL T6bb: return-home phase should release sentinels, got {returned.payload['sentinel_assignments']}")
        failed += 1
    complete = plan_from_observations(
        drones, sample_observations(0, clear=True), home_pos=home, returned_home=True,
    )
    if complete.phase != MissionPhase.COMPLETE:
        print(f"FAIL T6c: returned-home mission should complete, got {complete.phase}")
        failed += 1
    stale_obs = [
        BuildingObservation(0, 0, ObservationKind.BUILDING, np.array([0.0, 0.0, 0.0])),
        BuildingObservation(2, 2, ObservationKind.ENTRANCE, np.array([0.0, -8.0, 0.0])),
        BuildingObservation(7, 4, ObservationKind.INTERSECTION, np.array([0.0, 5.0, 0.0])),
    ]
    stale_drones = [
        {"id": i, "pos": np.array([0.0, -8.0 + i * 0.2, 0.0])}
        for i in range(8)
    ]
    stale_plan = plan_from_observations(stale_drones, stale_obs, home_pos=home)
    if "intersection_0" in stale_plan.payload["sentinel_assignments"]:
        print(f"FAIL T6d: stale intersection should release sentinel, got {stale_plan.payload['sentinel_assignments']}")
        failed += 1
    if 4 not in stale_plan.worker_ids:
        print(f"FAIL T6e: released intersection discoverer should be worker, workers={stale_plan.worker_ids}")
        failed += 1

    # Determinism under observation row-order changes.
    obs_reordered = list(reversed(sample_observations(1)))
    p_a = plan_from_observations(drones, sample_observations(1), home_pos=home)
    p_b = plan_from_observations(drones, obs_reordered, home_pos=home)
    if p_a.payload["sentinel_assignments"] != p_b.payload["sentinel_assignments"]:
        print(f"FAIL T7: role assignments changed with row order: {p_a.payload['sentinel_assignments']} vs {p_b.payload['sentinel_assignments']}")
        failed += 1

    building, trace = simulate_random_exploration(seed=11, n_drones=8)
    if len(trace) < 3:
        print(f"FAIL T8: random trace too short: {len(trace)}")
        failed += 1
    if trace[-1].plan.phase != MissionPhase.COMPLETE:
        print(f"FAIL T8b: random trace should end complete, got {trace[-1].plan.phase}")
        failed += 1
    if len(building.target_ids) > 3:
        print(f"FAIL T8c: target count should be 0-3, got {building.target_ids}")
        failed += 1

    floor = generate_smooth_floorplan(seed=4)
    for a, nbs in floor.edges.items():
        for b in nbs:
            if a > b:
                continue
            if _segment_crosses_wall(floor, floor.nodes[a], floor.nodes[b]):
                print(f"FAIL T9a: floorplan edge {a}->{b} crosses a rendered wall")
                failed += 1
                break
    full_shared_map = SmoothSharedMap(
        building_known=True,
        clear_seen=False,
        known_nodes=frozenset(floor.nodes),
        known_doors=frozenset(floor.doors),
        known_edges={node_id: tuple(nbs) for node_id, nbs in floor.edges.items()},
        known_wall_ids=frozenset(),
    )
    route_step = _route_visual_step(
        floor,
        floor.nodes["front_door"],
        floor.nodes["room_R3"],
        full_shared_map,
        1.0,
    )
    first_hall = floor.nodes["hall_0"]
    expected = _move_toward(floor.nodes["front_door"], first_hall, 1.0)
    if float(np.linalg.norm(route_step - expected)) > 1e-6:
        print(f"FAIL T9b: door-to-room route should enter hallway first, got {route_step}, expected {expected}")
        failed += 1
    overlap = np.array([floor.home_pos.copy(), floor.home_pos.copy()], dtype=np.float64)
    separated = _apply_collision_avoidance(floor, overlap, ("sentinel", "worker"), full_shared_map)
    if float(np.linalg.norm(separated[1] - separated[0])) < 0.8:
        print(f"FAIL T9c: collision avoidance did not separate overlapped drones: {separated}")
        failed += 1
    if float(np.linalg.norm(separated[0] - overlap[0])) >= float(np.linalg.norm(separated[1] - overlap[1])):
        print(f"FAIL T9d: worker should give way more than sentinel: {separated}")
        failed += 1

    smooth_plan, smooth_frames = simulate_smooth_building_mission(seed=4, n_drones=15, n_frames=700)
    if len(smooth_frames) < 40:
        print(f"FAIL T10: smooth trace too short: {len(smooth_frames)}")
        failed += 1
    if not any(frame.discovered_node_ids for frame in smooth_frames):
        print("FAIL T10b: smooth trace never discovered building nodes")
        failed += 1
    if len(smooth_plan.target_node_ids) > 3:
        print(f"FAIL T10c: smooth target count >3: {smooth_plan.target_node_ids}")
        failed += 1
    if smooth_plan.target_node_ids and not any(frame.discovered_target_ids for frame in smooth_frames):
        print(f"FAIL T10d: smooth trace has targets but discovered none: {smooth_plan.target_node_ids}")
        failed += 1
    extraction_frames = [frame for frame in smooth_frames if frame.target_escorts]
    for frame in smooth_frames:
        if "front_door" in frame.discovered_node_ids:
            continue
        if any(_inside_smooth_building(smooth_plan, pos, margin=-0.2) for pos in frame.drones):
            print(f"FAIL T10e: drone entered building before entrance discovery at frame {frame.frame}")
            failed += 1
            break
    if smooth_plan.target_node_ids and extraction_frames:
        rooms = set(smooth_plan.room_ids)
        for frame in extraction_frames:
            if not rooms.issubset(frame.discovered_node_ids):
                print(f"FAIL T10f: extraction before all rooms explored at frame {frame.frame}")
                failed += 1
                break
            for target_id, group in frame.target_escorts.items():
                if len(group) != min(3, 15):
                    print(f"FAIL T10g: target {target_id} should have 3 escorts, got {group}")
                    failed += 1
                    break
                if float(np.linalg.norm(frame.carried_targets[target_id] - smooth_plan.nodes[target_id])) > 1.0:
                    max_group_dist = max(
                        float(np.linalg.norm(frame.drones[did] - frame.carried_targets[target_id]))
                        for did in group
                    )
                    if max_group_dist > 1.8:
                        print(
                            f"FAIL T10g2: moving target {target_id} should visibly keep "
                            f"three escorts, max distance {max_group_dist:.2f} at frame {frame.frame}"
                        )
                        failed += 1
                        break
        for idx in range(1, len(smooth_frames)):
            prev = smooth_frames[idx - 1]
            cur = smooth_frames[idx]
            for did in range(len(cur.drones)):
                if _segment_crosses_wall(smooth_plan, prev.drones[did], cur.drones[did]):
                    print(f"FAIL T10g3: drone {did} crossed a wall at frame {cur.frame}")
                    failed += 1
                    break
                crossed_boundary = (
                    _inside_smooth_building(smooth_plan, prev.drones[did], margin=-0.2)
                    != _inside_smooth_building(smooth_plan, cur.drones[did], margin=-0.2)
                )
                if crossed_boundary and not _segment_near_door(smooth_plan, prev.drones[did], cur.drones[did]):
                    print(f"FAIL T10g4: drone {did} crossed building boundary away from a door at frame {cur.frame}")
                    failed += 1
                    break
            for target_id, target_pos in cur.carried_targets.items():
                if target_id not in prev.carried_targets:
                    continue
                prev_target_pos = prev.carried_targets[target_id]
                if _segment_crosses_wall(smooth_plan, prev_target_pos, target_pos):
                    print(f"FAIL T10g5: target {target_id} crossed a wall at frame {cur.frame}")
                    failed += 1
                    break
                crossed_boundary = (
                    _inside_smooth_building(smooth_plan, prev_target_pos, margin=-0.2)
                    != _inside_smooth_building(smooth_plan, target_pos, margin=-0.2)
                )
                if crossed_boundary and not _segment_near_door(smooth_plan, prev_target_pos, target_pos):
                    print(f"FAIL T10g6: target {target_id} crossed building boundary away from a door at frame {cur.frame}")
                    failed += 1
                    break
    complete_frames = [frame for frame in smooth_frames if frame.phase == "complete"]
    if not complete_frames:
        print("FAIL T10h: smooth mission should reach complete after drones return home")
        failed += 1
    else:
        max_home_dist = max(float(np.linalg.norm(pos - smooth_plan.home_pos)) for pos in complete_frames[0].drones)
        if max_home_dist > 5.0:
            print(f"FAIL T10i: complete before all drones returned home, max distance {max_home_dist:.2f}")
            failed += 1

    stress_plan, stress_frames = simulate_smooth_building_mission(seed=4, n_drones=15, n_frames=900, stress=True)
    if not any(frame.failed_drone_ids for frame in stress_frames):
        print("FAIL T11: stress trace never applied deterministic drone loss")
        failed += 1
    if not any(frame.comm_degraded and frame.relay_drone_ids for frame in stress_frames):
        print("FAIL T11b: stress trace never exposed relay-gated communication")
        failed += 1
    if not any(frame.phase == "complete" for frame in stress_frames):
        print("FAIL T11c: stress mission should complete after reassignment")
        failed += 1
    stress_failed_ids = set().union(*(set(frame.failed_drone_ids) for frame in stress_frames))
    for frame in stress_frames:
        for target_id, group in frame.target_escorts.items():
            if stress_failed_ids & set(group):
                print(f"FAIL T11d: failed drone assigned to extraction group {target_id}: {group}")
                failed += 1
                break
            if len(group) != min(3, 15 - len(stress_failed_ids)):
                print(f"FAIL T11e: stress target {target_id} should have 3 live escorts, got {group}")
                failed += 1
                break
    for idx in range(1, len(stress_frames)):
        prev = stress_frames[idx - 1]
        cur = stress_frames[idx]
        for did in range(len(cur.drones)):
            if _segment_crosses_wall(stress_plan, prev.drones[did], cur.drones[did]):
                print(f"FAIL T11f: stress drone {did} crossed a wall at frame {cur.frame}")
                failed += 1
                break
            crossed_boundary = (
                _inside_smooth_building(stress_plan, prev.drones[did], margin=-0.2)
                != _inside_smooth_building(stress_plan, cur.drones[did], margin=-0.2)
            )
            if crossed_boundary and not _segment_near_door(stress_plan, prev.drones[did], cur.drones[did]):
                print(f"FAIL T11g: stress drone {did} crossed building boundary away from a door at frame {cur.frame}")
                failed += 1
                break

    return failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Render a random exploration trace PNG.")
    ap.add_argument("--animate", action="store_true", help="Render a random exploration trace GIF.")
    ap.add_argument("--smooth", action="store_true", help="Render a smooth random building mission GIF.")
    ap.add_argument("--stress", action="store_true", help="Add deterministic drone loss and relay-gated MAP sharing.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="/Users/jmcentire/Code/drone_swarm/figures/building_explore_demo.png")
    ap.add_argument("--fps", type=float, default=1.5)
    ap.add_argument("--frames", type=int, default=220)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    n_failed = _tests()
    print(
        "building_explore: all tests passed"
        if n_failed == 0
        else f"building_explore: {n_failed} tests failed"
    )
    if n_failed != 0:
        raise SystemExit(1)
    if args.smooth:
        plan, frames = simulate_smooth_building_mission(seed=args.seed, n_frames=args.frames, stress=args.stress)
        render_smooth_building_animation(plan, frames, args.output, fps=args.fps, stride=args.stride)
        print(
            f"building_explore: wrote {args.output} "
            f"(seed={args.seed}, stress={args.stress}, targets={len(plan.target_node_ids)}, "
            f"frames={len(frames)}, stride={args.stride})"
        )
    elif args.demo or args.animate:
        building, trace = simulate_random_exploration(seed=args.seed)
        if args.animate:
            render_exploration_animation(building, trace, args.output, fps=args.fps)
        else:
            render_exploration_trace(building, trace, args.output)
        print(
            f"building_explore: wrote {args.output} "
            f"(seed={args.seed}, targets={len(building.target_ids)}, steps={len(trace)})"
        )


if __name__ == "__main__":
    main()
