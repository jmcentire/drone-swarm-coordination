# /// script
# dependencies = ["numpy<3"]
# ///
"""Multi-seed loss-scenario bench.

Scenarios:
  * baseline:        no losses, baseline metrics
  * single_random:   kill 1 random drone mid-mission
  * single_edge:     kill 1 corner/boundary drone
  * multi_random_5:  kill 5 random drones simultaneously
  * multi_random_10: kill 10 random drones simultaneously
  * multi_random_20: kill 20 random drones simultaneously
  * cluster_small:   kill all drones within 1.5 * spacing of a random victim
                     (~7 drones — one HCP ring)
  * cluster_large:   kill all drones within 2.5 * spacing of a random victim
                     (~19 drones — depth-charge sim)
  * edge_burst:      kill 5 boundary drones simultaneously
  * trickle:         kill 10 drones one at a time, spaced over 10 seconds

Each scenario runs SEEDS=5 times; metrics are aggregated.

Output: bench_losses_results.json with per-scenario {mean, std, min, max}
on each metric. Console table reports the headline numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from failures import (
    FailureSchedule,
    apply_kills_at_tick,
    effective_neighbors,
    live_edge_strain,
    select_cluster_drones,
    select_corner_drones,
    select_random_drones,
)
from lattice import (
    build_neighbor_graph,
    hcp_positions,
    non_coplanar_neighbors,
)
from pbd import PBDLattice, edges_from_neighbors

SPACING = 10.0
COMMS_DELTA = 0.15
HEX_RADIUS = 3
N_LAYERS = 3
DT = 0.05
MISSION_S = 60.0
KILL_TICK = 400
TRICKLE_DURATION_TICKS = 200
SEEDS = 5
VELOCITY = 1.0


@dataclass
class ScenarioRun:
    name: str
    seed: int
    n_initially_alive: int
    n_killed: int
    n_finally_alive: int
    pct_ticks_topology_violated_after_kill: float
    mean_violators_after_kill: float
    max_violators_after_kill: int
    mean_abs_strain_after_kill: float
    max_abs_strain_after_kill: float
    recovery_ticks: int             # -1 if never recovered (violations persisted)
    centroid_target_track_err_m: float


@dataclass
class ScenarioStats:
    name: str
    n_seeds: int
    mean_pct_violated: float
    std_pct_violated: float
    max_pct_violated: float
    mean_recovery_ticks: float
    max_recovery_ticks: int
    n_recovered: int
    mean_strain: float
    max_strain: float
    mean_track_err_m: float


def build_schedule(name: str, positions: np.ndarray,
                   neighbors: list[list[int]], n: int,
                   rng: np.random.Generator) -> FailureSchedule:
    sched = FailureSchedule()
    if name == "baseline":
        pass
    elif name == "single_random":
        v = select_random_drones(n, 1, rng)
        sched.kill_events.append((KILL_TICK, v))
    elif name == "single_edge":
        v = select_corner_drones(positions, 1, neighbors)
        sched.kill_events.append((KILL_TICK, v))
    elif name == "multi_random_5":
        v = select_random_drones(n, 5, rng)
        sched.kill_events.append((KILL_TICK, v))
    elif name == "multi_random_10":
        v = select_random_drones(n, 10, rng)
        sched.kill_events.append((KILL_TICK, v))
    elif name == "multi_random_20":
        v = select_random_drones(n, 20, rng)
        sched.kill_events.append((KILL_TICK, v))
    elif name == "cluster_small":
        # Pick a random center, kill all within 1.5 * spacing.
        center = int(rng.integers(n))
        v = select_cluster_drones(positions, center, 1.5 * SPACING)
        sched.kill_events.append((KILL_TICK, v))
    elif name == "cluster_large":
        center = int(rng.integers(n))
        v = select_cluster_drones(positions, center, 2.5 * SPACING)
        sched.kill_events.append((KILL_TICK, v))
    elif name == "edge_burst":
        v = select_corner_drones(positions, 5, neighbors)
        sched.kill_events.append((KILL_TICK, v))
    elif name == "trickle":
        # 10 random victims spread over TRICKLE_DURATION_TICKS.
        victims = select_random_drones(n, 10, rng)
        for k, v in enumerate(victims):
            tick = KILL_TICK + k * (TRICKLE_DURATION_TICKS // 10)
            sched.kill_events.append((tick, [v]))
    else:
        raise ValueError(f"unknown scenario: {name}")
    return sched


def run_one(name: str, seed: int) -> ScenarioRun:
    rng = np.random.default_rng(seed)
    pos0 = hcp_positions(HEX_RADIUS, N_LAYERS, spacing=SPACING)
    n = pos0.shape[0]
    pos0 = pos0 + rng.normal(scale=0.02 * SPACING, size=pos0.shape)
    comms_range = SPACING * (1 + COMMS_DELTA)
    base_neighbors = build_neighbor_graph(pos0, comms_range)
    edges = edges_from_neighbors(base_neighbors)
    edges_arr = np.asarray(edges, dtype=np.int64)

    schedule = build_schedule(name, pos0, base_neighbors, n, rng)
    alive = np.ones(n, dtype=bool)
    sim = PBDLattice(
        pos0,
        edges,
        rest_length=SPACING,
        compliance=1e-5,
        damping=0.95,
    )
    target_vel = np.array([VELOCITY, 0.0, 0.0])

    n_steps = int(MISSION_S / DT)
    initial_centroid_x = pos0[:, 0].mean()

    violators_after_kill: list[int] = []
    strains_after_kill: list[float] = []
    max_strain_seen = 0.0
    recovery_ticks = -1
    recovered = False
    last_kill_tick = -1
    final_centroid_x = initial_centroid_x

    for tick in range(n_steps):
        # Kill scheduled drones at this tick.
        n_killed_this_tick = apply_kills_at_tick(schedule, tick, alive)
        if n_killed_this_tick > 0:
            last_kill_tick = tick
            recovered = False  # restart recovery timer

        # Drive force toward target velocity.
        vel_error = target_vel[None, :] - sim.vel
        force = 0.5 * vel_error
        force[~alive] = 0.0

        sim.step(dt=DT, external_force=force, iters=8, alive=alive)

        # Metrics after the first kill event.
        if last_kill_tick >= 0:
            live_idx = np.where(alive)[0]
            live_pos = sim.pos[live_idx]
            eff = effective_neighbors(
                build_neighbor_graph(live_pos, comms_range),
                FailureSchedule(),  # no partitions in these scenarios
                tick,
                np.ones(len(live_idx), dtype=bool),
            )
            ncp = non_coplanar_neighbors(live_pos, eff)
            n_violators = int((ncp < 4).sum())
            violators_after_kill.append(n_violators)
            strains = live_edge_strain(sim.pos, edges_arr, SPACING, alive)
            strains_after_kill.append(float(np.mean(np.abs(strains))))
            max_strain_seen = max(max_strain_seen, float(np.max(np.abs(strains))))

            if not recovered and tick > last_kill_tick + 5 and n_violators == 0:
                strain_now = float(np.mean(np.abs(strains)))
                if strain_now < 0.02:  # within 2% strain = converged
                    recovery_ticks = tick - last_kill_tick
                    recovered = True

        final_centroid_x = float(sim.pos[alive, 0].mean()) if alive.any() else final_centroid_x

    intended_x = initial_centroid_x + VELOCITY * MISSION_S
    track_err = abs(final_centroid_x - intended_x)

    n_initial = n
    n_killed = int((~alive).sum())
    n_final = int(alive.sum())
    pct_viol = (
        100.0 * np.mean([v > 0 for v in violators_after_kill])
        if violators_after_kill else 0.0
    )
    mean_viol = float(np.mean(violators_after_kill)) if violators_after_kill else 0.0
    max_viol = int(max(violators_after_kill)) if violators_after_kill else 0
    mean_strain = float(np.mean(strains_after_kill)) if strains_after_kill else 0.0

    return ScenarioRun(
        name=name,
        seed=seed,
        n_initially_alive=n_initial,
        n_killed=n_killed,
        n_finally_alive=n_final,
        pct_ticks_topology_violated_after_kill=pct_viol,
        mean_violators_after_kill=mean_viol,
        max_violators_after_kill=max_viol,
        mean_abs_strain_after_kill=mean_strain,
        max_abs_strain_after_kill=max_strain_seen,
        recovery_ticks=recovery_ticks,
        centroid_target_track_err_m=track_err,
    )


def aggregate(runs: list[ScenarioRun]) -> ScenarioStats:
    pcts = np.array([r.pct_ticks_topology_violated_after_kill for r in runs])
    recs = np.array([r.recovery_ticks for r in runs])
    strains = np.array([r.mean_abs_strain_after_kill for r in runs])
    max_strains = np.array([r.max_abs_strain_after_kill for r in runs])
    tracks = np.array([r.centroid_target_track_err_m for r in runs])
    n_rec = int((recs >= 0).sum())
    return ScenarioStats(
        name=runs[0].name,
        n_seeds=len(runs),
        mean_pct_violated=float(pcts.mean()),
        std_pct_violated=float(pcts.std()),
        max_pct_violated=float(pcts.max()),
        mean_recovery_ticks=float(recs[recs >= 0].mean()) if n_rec else -1.0,
        max_recovery_ticks=int(recs[recs >= 0].max()) if n_rec else -1,
        n_recovered=n_rec,
        mean_strain=float(strains.mean()),
        max_strain=float(max_strains.max()),
        mean_track_err_m=float(tracks.mean()),
    )


def main() -> None:
    t0 = time.perf_counter()
    scenarios = [
        "baseline",
        "single_random",
        "single_edge",
        "multi_random_5",
        "multi_random_10",
        "multi_random_20",
        "cluster_small",
        "cluster_large",
        "edge_burst",
        "trickle",
    ]
    all_runs: list[ScenarioRun] = []
    all_stats: list[ScenarioStats] = []
    for sc in scenarios:
        runs = [run_one(sc, seed=k + 1000) for k in range(SEEDS)]
        all_runs.extend(runs)
        all_stats.append(aggregate(runs))

    out = {
        "scenarios": [asdict(s) for s in all_stats],
        "runs": [asdict(r) for r in all_runs],
        "wall_time_s": time.perf_counter() - t0,
        "config": {
            "spacing": SPACING,
            "comms_delta": COMMS_DELTA,
            "hex_radius": HEX_RADIUS,
            "n_layers": N_LAYERS,
            "velocity": VELOCITY,
            "mission_s": MISSION_S,
            "seeds": SEEDS,
        },
    }
    with open("bench_losses_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"{'scenario':<18s} {'killed':>7s} {'pct_v':>7s} {'std':>6s} "
        f"{'max_v':>6s} {'recov':>7s} {'n_rec':>6s} {'strain':>7s} {'track':>7s}"
    )
    for s in all_stats:
        killed_str = (
            f"{int(np.mean([r.n_killed for r in all_runs if r.name == s.name]))}"
        )
        recov_str = (
            f"{s.mean_recovery_ticks:.1f}" if s.mean_recovery_ticks >= 0 else "-"
        )
        print(
            f"{s.name:<18s} {killed_str:>7s} "
            f"{s.mean_pct_violated:7.2f} {s.std_pct_violated:6.2f} "
            f"{s.max_pct_violated:6.2f} {recov_str:>7s} "
            f"{s.n_recovered:>4d}/{s.n_seeds:<1d} "
            f"{s.mean_strain:7.4f} {s.mean_track_err_m:7.3f}"
        )
    print(f"wall time: {time.perf_counter()-t0:.1f}s ({len(all_runs)} runs)")


if __name__ == "__main__":
    main()
