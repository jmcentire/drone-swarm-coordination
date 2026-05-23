# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Debug: run baseline scenario one tick at a time, print phase state."""

from __future__ import annotations

import numpy as np

from lattice import build_neighbor_graph, hcp_positions, non_coplanar_neighbors
from mission import (
    MissionConfig,
    MissionState,
    Phase,
    compute_manifold_targets,
    mission_force,
    update_mission,
)
from pbd import PBDLattice, edges_from_neighbors

SPACING = 10.0
COMMS_DELTA = 0.15
DT = 0.05


def main() -> None:
    rng = np.random.default_rng(2000)
    pos0 = hcp_positions(3, 3, spacing=SPACING)
    n = pos0.shape[0]
    pos0 = pos0 + rng.normal(scale=0.02 * SPACING, size=pos0.shape)
    comms_range = SPACING * (1 + COMMS_DELTA)
    base_neighbors = build_neighbor_graph(pos0, comms_range)
    edges = edges_from_neighbors(base_neighbors)

    cfg = MissionConfig(spacing=SPACING)
    initial_centroid = pos0.mean(axis=0)
    state = MissionState(
        leg_start_centroid=initial_centroid.copy(),
        last_known_centroid=initial_centroid.copy(),
    )
    alive = np.ones(n, dtype=bool)
    sim = PBDLattice(pos0, edges, rest_length=SPACING, compliance=1e-5, damping=0.95)

    prev_phase = state.phase
    for tick in range(3000):
        is_violating = np.zeros(n, dtype=bool)
        current_degree = np.zeros(n, dtype=np.int64)
        live_idx = np.where(alive)[0]
        live_pos = sim.pos[live_idx]
        ngh = build_neighbor_graph(live_pos, comms_range)
        ncp = non_coplanar_neighbors(live_pos, ngh)
        for k, did in enumerate(live_idx):
            current_degree[did] = len(ngh[k])
            if ncp[k] < 4:
                is_violating[did] = True

        update_mission(state, cfg, tick, DT, sim.pos, sim.vel, alive, is_violating, current_degree)

        if state.phase != prev_phase:
            elapsed = tick - state.phase_start_tick
            print(f"tick {tick}: {prev_phase.value} -> {state.phase.value} (after {elapsed} ticks in old phase)")
            if state.phase == Phase.REFORM and state.targets is not None:
                print(f"  current centroid: {sim.pos[alive].mean(axis=0)}")
                print(f"  targets centroid: {state.targets[alive].mean(axis=0)}")
                print(f"  drone 0 current: {sim.pos[0]} target: {state.targets[0]}")
                print(f"  drone 55 current: {sim.pos[55]} target: {state.targets[55]}")
                # Lattice spread
                p_alive = sim.pos[alive]
                print(f"  position spread: x=[{p_alive[:,0].min():.1f},{p_alive[:,0].max():.1f}] "
                      f"y=[{p_alive[:,1].min():.1f},{p_alive[:,1].max():.1f}] "
                      f"z=[{p_alive[:,2].min():.1f},{p_alive[:,2].max():.1f}]")
                t_alive = state.targets[alive]
                print(f"  target spread: x=[{t_alive[:,0].min():.1f},{t_alive[:,0].max():.1f}] "
                      f"y=[{t_alive[:,1].min():.1f},{t_alive[:,1].max():.1f}] "
                      f"z=[{t_alive[:,2].min():.1f},{t_alive[:,2].max():.1f}]")
            prev_phase = state.phase

        # Print REFORM state every 100 ticks
        if state.phase == Phase.REFORM and tick % 50 == 0 and state.targets is not None:
            offsets = state.targets[alive] - sim.pos[alive]
            errs = np.linalg.norm(offsets, axis=1)
            speeds = np.linalg.norm(sim.vel[alive], axis=1)
            n_locked = int(state.locked.sum()) if state.locked is not None else 0
            print(
                f"  tick {tick} REFORM: max_err={errs.max():.3f}m mean_err={errs.mean():.3f}m "
                f"max_speed={speeds.max():.4f}m/s mean_speed={speeds.mean():.4f}m/s "
                f"locked={n_locked}/{alive.sum()} qstreak={state.quiescent_streak} leg={state.leg_index}"
            )

        force = mission_force(state, cfg, sim.pos, sim.vel, alive, current_degree)
        sim.step(
            dt=DT, external_force=force, iters=8, alive=alive,
            locked=state.locked if state.locked is not None else None,
        )


if __name__ == "__main__":
    main()
