# /// script
# dependencies = ["numpy<3"]
# ///
"""Mission-cycle bench: stop-and-go legs with vote/map/reform.

The mission FSM (mission.py) is convergence-gated and has no special
panic mode. All loss recovery happens inside REFORM. Per-drone isolation
recovery is the only fallback mechanism.

Scenarios per seed:
  * baseline:              no failures, run cleanly
  * loss_mid_move:         kill 3 random drones during a MOVE phase
  * loss_mid_reform:       kill 5 random drones during a REFORM phase
  * catastrophic:          kill 20% of swarm at once
  * trickle_long:          1 random drone every ~30s over 300s
  * cluster_then_continue: cluster (depth-charge-style) loss

Metrics per run:
  * legs_completed
  * isolation_events:      number of times a drone entered isolation recovery
  * final_topology_ok
  * total_distance_m
  * n_survivors
  * phase_time_pct
  * pct_ticks_violating_topology
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from failures import (
    FailureSchedule,
    active_diversion_partitions,
    apply_obstacle_projection,
    diversion_force,
    filter_edges_for_partitions,
    select_cluster_drones,
    select_random_drones,
)
from lattice import (
    build_neighbor_graph,
    hcp_positions,
    non_coplanar_neighbors,
)
from mission import (
    MissionConfig,
    MissionState,
    Phase,
    mission_force,
    update_mission,
)
from pbd import PBDLattice, edges_from_neighbors

SPACING = 10.0
COMMS_DELTA = 0.15
HEX_RADIUS = 3
N_LAYERS = 3
DT = 0.05
MISSION_S = 300.0
SEEDS = 3
N_SPARES = 12       # spare-drone pool (drafts near centroid; promotes on loss)


@dataclass
class MissionRun:
    name: str
    seed: int
    n_initial: int
    n_survivors: int
    legs_completed: int
    isolation_events: int
    n_promotions: int
    final_topology_ok: bool
    # Graceful-degradation metrics at end of mission:
    pct_degraded_at_end: float        # % of survivors with ncp < 4
    mean_degree_at_end: float         # mean comms-degree of survivors
    median_degree_at_end: float
    min_degree_at_end: int
    mean_ncp_at_end: float            # mean non-coplanar-neighbor count
    n_isolated_at_end: int            # survivors with degree == 0
    max_components_observed: int      # peak connected components during mission
    n_components_at_end: int
    max_obstacle_penetration_m: float # peak depth any drone was inside an obstacle
    total_distance_m: float
    intended_distance_m: float
    phase_time_pct: dict[str, float]
    pct_ticks_violating_topology: float
    final_phase: str


@dataclass
class MissionStats:
    name: str
    n_seeds: int
    mean_legs: float
    mean_isolation_events: float
    n_final_ok: int
    mean_distance_m: float
    mean_survivors: float
    mean_pct_degraded_at_end: float
    mean_mean_degree_at_end: float
    mean_min_degree_at_end: float
    mean_isolated_at_end: float
    mean_promotions: float
    mean_max_components: float
    mean_components_at_end: float
    max_obstacle_penetration_m: float


def build_schedule(name: str, n: int, n_lattice: int, seed: int,
                    positions: np.ndarray) -> tuple[list, FailureSchedule]:
    """Returns (kill_events_list, full_failure_schedule)."""
    rng = np.random.default_rng(seed)
    kills: list[tuple[int, list[int]]] = []
    sched = FailureSchedule()
    if name == "baseline":
        pass
    elif name == "loss_mid_move":
        kills.append((1200, select_random_drones(n_lattice, 3, rng)))
    elif name == "loss_mid_reform":
        kills.append((1800, select_random_drones(n_lattice, 5, rng)))
    elif name == "catastrophic":
        n_kill = int(0.2 * n_lattice)
        kills.append((1600, select_random_drones(n_lattice, n_kill, rng)))
    elif name == "trickle_long":
        for k in range(10):
            kills.append((400 + k * 600, select_random_drones(n_lattice, 1, rng)))
    elif name == "cluster_then_continue":
        center = int(rng.integers(n_lattice))
        victims = select_cluster_drones(positions, center, 1.8 * SPACING)
        kills.append((1500, victims))
    elif name == "bifurcation":
        # Split the swarm spatially: drones with x > median get pushed
        # perpendicular to the mission heading for ~60s, then the current
        # subsides. The swarm should split, operate independently, then
        # reunite via mission_force pulling everyone back to the centroid.
        median_x = float(np.median(positions[:n_lattice, 0]))
        diverged = [i for i in range(n_lattice) if positions[i, 0] > median_x]
        # Perpendicular to mission heading (heading is +x; perpendicular is +y)
        direction = np.array([0.0, 1.0, 0.0])
        # Apply between tick 800 and tick 2000 (60s of current)
        sched.diversion_events.append((800, 2000, diverged, direction))
    elif name == "bifurcation_long":
        # Longer separation that exceeds comms range, then heals.
        median_x = float(np.median(positions[:n_lattice, 0]))
        diverged = [i for i in range(n_lattice) if positions[i, 0] > median_x]
        direction = np.array([0.0, 1.0, 0.0])
        sched.diversion_events.append((800, 3200, diverged, direction))
    elif name == "obstacle_small":
        # Small sphere in the swarm's path (heading is +x). The swarm
        # should deform locally around it and continue past.
        swarm_centroid = positions[:n_lattice].mean(axis=0)
        sched.obstacles.append((swarm_centroid + np.array([60.0, 0.0, 0.0]), 5.0))
    elif name == "obstacle_medium":
        swarm_centroid = positions[:n_lattice].mean(axis=0)
        sched.obstacles.append((swarm_centroid + np.array([60.0, 0.0, 0.0]), 15.0))
    elif name == "obstacle_large":
        # Roughly half the swarm's cross-section. Tests deformation under
        # significant blockage; some drones may be forced into degraded
        # connectivity as they squeeze past.
        swarm_centroid = positions[:n_lattice].mean(axis=0)
        sched.obstacles.append((swarm_centroid + np.array([80.0, 0.0, 0.0]), 25.0))
    elif name == "obstacle_field":
        # Multiple smaller obstacles staggered through the path.
        swarm_centroid = positions[:n_lattice].mean(axis=0)
        sched.obstacles.append((swarm_centroid + np.array([40.0, -10.0, 0.0]), 6.0))
        sched.obstacles.append((swarm_centroid + np.array([80.0,  12.0, 0.0]), 7.0))
        sched.obstacles.append((swarm_centroid + np.array([120.0, -5.0, 5.0]), 8.0))
    else:
        raise ValueError(name)
    return kills, sched


def build_kill_schedule(name: str, n: int, n_lattice: int, seed: int,
                          positions: np.ndarray):
    """Back-compat shim returning just the kill list."""
    kills, _ = build_schedule(name, n, n_lattice, seed, positions)
    return kills


def run_one(name: str, seed: int) -> MissionRun:
    rng = np.random.default_rng(seed)
    # Active drones occupy the canonical HCP lattice. Spares hover in a
    # ring just inside the centroid -- no PBD edges, they drift with the
    # swarm and promote into vacated lattice slots on demand.
    lattice_pos = hcp_positions(HEX_RADIUS, N_LAYERS, spacing=SPACING)
    n_lattice = lattice_pos.shape[0]
    initial_centroid_full = lattice_pos.mean(axis=0)
    spare_pos = np.zeros((N_SPARES, 3))
    for i in range(N_SPARES):
        angle = 2 * np.pi * i / max(N_SPARES, 1)
        z_off = ((i % 3) - 1) * 0.3 * SPACING
        spare_pos[i] = initial_centroid_full + np.array([
            1.5 * SPACING * np.cos(angle),
            1.5 * SPACING * np.sin(angle),
            z_off,
        ])
    pos0 = np.vstack([lattice_pos, spare_pos])
    n = pos0.shape[0]
    pos0 = pos0 + rng.normal(scale=0.02 * SPACING, size=pos0.shape)

    comms_range = SPACING * (1 + COMMS_DELTA)
    # PBD edges only among the lattice (active) drones. Spares are free
    # to move under mission_force alone; once promoted they slot into
    # position and are held there by the attractor (no dynamic edges).
    lattice_neighbors = build_neighbor_graph(pos0[:n_lattice], comms_range)
    edges = edges_from_neighbors(lattice_neighbors)

    cfg = MissionConfig(spacing=SPACING, n_spares=N_SPARES)

    initial_centroid = pos0[:n_lattice].mean(axis=0)
    state = MissionState(
        leg_start_centroid=initial_centroid.copy(),
        last_known_centroid=initial_centroid.copy(),
    )

    alive = np.ones(n, dtype=bool)
    sim = PBDLattice(pos0, edges, rest_length=SPACING, compliance=1e-5, damping=0.95)

    n_steps = int(MISSION_S / DT)
    kills, schedule = build_schedule(name, n, n_lattice, seed, pos0)
    kill_dict: dict[int, list[int]] = {}
    for tick, victims in kills:
        kill_dict.setdefault(tick, []).extend(victims)

    phase_ticks: dict[str, int] = {p.value: 0 for p in Phase}
    topology_violation_ticks = 0
    max_components_observed = 1
    components_at_end = 1
    max_obstacle_penetration_m = 0.0  # peak distance any drone was inside an obstacle

    def _count_components(ngh: list[list[int]]) -> int:
        seen = [False] * len(ngh)
        count = 0
        for s in range(len(ngh)):
            if seen[s]:
                continue
            count += 1
            stack = [s]
            while stack:
                u = stack.pop()
                if seen[u]:
                    continue
                seen[u] = True
                for v in ngh[u]:
                    if not seen[v]:
                        stack.append(v)
        return count

    for tick in range(n_steps):
        if tick in kill_dict:
            for v in kill_dict[tick]:
                alive[v] = False

        # Determine which edges/comms are dropped this tick by any active
        # diversion partition (sub-group physically separated, out of range).
        partitions_now = active_diversion_partitions(schedule, tick)

        is_violating = np.zeros(n, dtype=bool)
        current_degree = np.zeros(n, dtype=np.int64)
        live_idx = np.where(alive)[0]
        if len(live_idx) >= 4:
            live_pos = sim.pos[live_idx]
            ngh = build_neighbor_graph(live_pos, comms_range)
            # Strip cross-partition comms edges from the local indices.
            if partitions_now:
                idx_to_did = {k: int(did) for k, did in enumerate(live_idx)}
                for k in range(len(ngh)):
                    filtered = []
                    for nb_k in ngh[k]:
                        u = idx_to_did[k]
                        v = idx_to_did[nb_k]
                        crosses = False
                        for diverged, _ in partitions_now:
                            rest = set(range(n)) - diverged
                            if (u in diverged and v in rest) or (u in rest and v in diverged):
                                crosses = True
                                break
                        if not crosses:
                            filtered.append(nb_k)
                    ngh[k] = filtered
            ncp = non_coplanar_neighbors(live_pos, ngh)
            for k, did in enumerate(live_idx):
                current_degree[did] = len(ngh[k])
                if ncp[k] < 4:
                    is_violating[did] = True
            if tick % 50 == 0:
                comps = _count_components(ngh)
                if comps > max_components_observed:
                    max_components_observed = comps
                components_at_end = comps
        else:
            is_violating[live_idx] = True

        if is_violating.any():
            topology_violation_ticks += 1

        update_mission(
            state, cfg, tick, DT, sim.pos, sim.vel, alive, is_violating, current_degree
        )
        phase_ticks[state.phase.value] += 1

        force = mission_force(state, cfg, sim.pos, sim.vel, alive, current_degree)
        # Add any current/disturbance force from active diversion events.
        force = force + diversion_force(schedule, tick, n, magnitude=2.0)
        # Drop PBD edges that cross an active diversion partition: the
        # current physically separates the sub-group beyond spring range.
        edge_keep = filter_edges_for_partitions(np.asarray(edges), partitions_now, n)
        sim.step(
            dt=DT, external_force=force, iters=8, alive=alive,
            locked=state.locked if state.locked is not None else None,
            edge_mask=edge_keep,
        )

        # Obstacle handling: measure peak penetration BEFORE projecting drones
        # back out. Then project. This tracks how much physical work the
        # collision constraint had to do.
        if schedule.obstacles:
            for center, radius in schedule.obstacles:
                offsets = sim.pos - center
                dists = np.linalg.norm(offsets, axis=1)
                inside_amount = np.where(
                    (dists < radius) & alive,
                    radius - dists,
                    0.0,
                )
                if inside_amount.size and inside_amount.max() > max_obstacle_penetration_m:
                    max_obstacle_penetration_m = float(inside_amount.max())
            apply_obstacle_projection(sim.pos, schedule.obstacles, alive)

    # Final topology check with graceful-degradation metrics
    live_idx = np.where(alive)[0]
    if len(live_idx) >= 4:
        live_pos = sim.pos[live_idx]
        ngh = build_neighbor_graph(live_pos, comms_range)
        ncp = non_coplanar_neighbors(live_pos, ngh)
        degs = np.array([len(g) for g in ngh])
        n_degraded = int((ncp < 4).sum())
        final_ok = n_degraded == 0
        pct_degraded = 100.0 * n_degraded / len(live_idx)
        mean_deg = float(degs.mean())
        median_deg = float(np.median(degs))
        min_deg = int(degs.min())
        mean_ncp = float(ncp.mean())
        n_isolated = int((degs == 0).sum())
    else:
        final_ok = False
        pct_degraded = 100.0
        mean_deg = 0.0
        median_deg = 0.0
        min_deg = 0
        mean_ncp = 0.0
        n_isolated = int(len(live_idx))

    final_centroid = sim.pos[alive].mean(axis=0) if alive.any() else initial_centroid
    total_distance = float(np.linalg.norm(final_centroid - initial_centroid))
    intended_distance = state.leg_index * cfg.leg_distance_m

    total_phase = sum(phase_ticks.values()) or 1
    phase_pct = {k: 100.0 * v / total_phase for k, v in phase_ticks.items()}

    return MissionRun(
        name=name,
        seed=seed,
        n_initial=n,
        n_survivors=int(alive.sum()),
        legs_completed=state.leg_index,
        isolation_events=state.isolation_events,
        n_promotions=int(state.n_promotions),
        final_topology_ok=bool(final_ok),
        pct_degraded_at_end=pct_degraded,
        mean_degree_at_end=mean_deg,
        median_degree_at_end=median_deg,
        min_degree_at_end=min_deg,
        mean_ncp_at_end=mean_ncp,
        n_isolated_at_end=n_isolated,
        max_components_observed=int(max_components_observed),
        n_components_at_end=int(components_at_end),
        max_obstacle_penetration_m=max_obstacle_penetration_m,
        total_distance_m=total_distance,
        intended_distance_m=intended_distance,
        phase_time_pct=phase_pct,
        pct_ticks_violating_topology=100.0 * topology_violation_ticks / n_steps,
        final_phase=state.phase.value,
    )


def aggregate(runs: list[MissionRun]) -> MissionStats:
    legs = np.array([r.legs_completed for r in runs])
    iso = np.array([r.isolation_events for r in runs])
    dist = np.array([r.total_distance_m for r in runs])
    survivors = np.array([r.n_survivors for r in runs])
    pct_deg = np.array([r.pct_degraded_at_end for r in runs])
    mean_deg = np.array([r.mean_degree_at_end for r in runs])
    min_deg = np.array([r.min_degree_at_end for r in runs])
    iso_end = np.array([r.n_isolated_at_end for r in runs])
    max_comp = np.array([r.max_components_observed for r in runs])
    end_comp = np.array([r.n_components_at_end for r in runs])
    pene = np.array([r.max_obstacle_penetration_m for r in runs])
    promo = np.array([r.n_promotions for r in runs])
    n_ok = sum(1 for r in runs if r.final_topology_ok)
    return MissionStats(
        name=runs[0].name,
        n_seeds=len(runs),
        mean_legs=float(legs.mean()),
        mean_isolation_events=float(iso.mean()),
        n_final_ok=int(n_ok),
        mean_distance_m=float(dist.mean()),
        mean_survivors=float(survivors.mean()),
        mean_pct_degraded_at_end=float(pct_deg.mean()),
        mean_mean_degree_at_end=float(mean_deg.mean()),
        mean_min_degree_at_end=float(min_deg.mean()),
        mean_isolated_at_end=float(iso_end.mean()),
        mean_promotions=float(promo.mean()),
        mean_max_components=float(max_comp.mean()),
        mean_components_at_end=float(end_comp.mean()),
        max_obstacle_penetration_m=float(pene.max()),
    )


def main() -> None:
    t0 = time.perf_counter()
    scenarios = [
        "baseline",
        "loss_mid_move",
        "loss_mid_reform",
        "catastrophic",
        "trickle_long",
        "cluster_then_continue",
        "bifurcation",
        "bifurcation_long",
        "obstacle_small",
        "obstacle_medium",
        "obstacle_large",
        "obstacle_field",
    ]
    all_runs: list[MissionRun] = []
    all_stats: list[MissionStats] = []
    for sc in scenarios:
        runs = [run_one(sc, seed=k + 2000) for k in range(SEEDS)]
        all_runs.extend(runs)
        all_stats.append(aggregate(runs))

    out = {
        "scenarios": [asdict(s) for s in all_stats],
        "runs": [asdict(r) for r in all_runs],
        "wall_time_s": time.perf_counter() - t0,
    }
    with open("bench_mission_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"{'scenario':<22s} {'legs':>5s} {'surv':>5s} {'dist':>7s} "
        f"{'%deg':>6s} {'<deg>':>6s} {'mindeg':>7s} {'iso_end':>8s} "
        f"{'promo':>6s} {'maxC':>5s} {'endC':>5s} {'pene':>6s}"
    )
    for s in all_stats:
        print(
            f"{s.name:<22s} {s.mean_legs:5.1f} {s.mean_survivors:5.1f} "
            f"{s.mean_distance_m:7.2f} "
            f"{s.mean_pct_degraded_at_end:6.2f} "
            f"{s.mean_mean_degree_at_end:6.2f} "
            f"{s.mean_min_degree_at_end:7.2f} "
            f"{s.mean_isolated_at_end:8.2f} "
            f"{s.mean_promotions:6.1f} "
            f"{s.mean_max_components:5.1f} "
            f"{s.mean_components_at_end:5.1f} "
            f"{s.max_obstacle_penetration_m:6.3f}"
        )

    print("\nper-run detail:")
    print(
        f"{'scenario':<22s} {'seed':>5s} {'legs':>5s} {'surv':>5s} "
        f"{'%deg':>6s} {'<deg>':>6s} {'mindeg':>6s} {'iso':>4s} "
        f"{'promo':>6s} {'dist':>7s} {'final':>7s}"
    )
    for r in all_runs:
        print(
            f"{r.name:<22s} {r.seed:5d} {r.legs_completed:5d} {r.n_survivors:5d} "
            f"{r.pct_degraded_at_end:6.2f} "
            f"{r.mean_degree_at_end:6.2f} "
            f"{r.min_degree_at_end:6d} "
            f"{r.n_isolated_at_end:4d} "
            f"{r.n_promotions:6d} "
            f"{r.total_distance_m:7.2f} "
            f"{r.final_phase:>7s}"
        )
    print(f"\nwall time: {time.perf_counter()-t0:.1f}s ({len(all_runs)} runs)")


if __name__ == "__main__":
    main()
