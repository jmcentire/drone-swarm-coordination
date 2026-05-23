# /// script
# dependencies = ["numpy<3"]
# ///
"""Topology-under-motion bench.

Question: does a 3D HCP lattice held together by PBD distance constraints
preserve its topology invariant (every drone has >=4 non-coplanar neighbors
within comms range) while the whole swarm translates in a fixed direction?

Sub-questions covered:
  (a) Topology violation rate vs. swarm velocity.
  (b) Mean edge strain (formation distortion) vs. velocity.
  (c) Recovery time after a synthetic node loss.
  (d) Sensitivity to comms-range slack: k * (1 + delta_ratio).

Method:
  - Generate HCP lattice (default 111 drones: hex_radius=3, n_layers=3).
  - Apply uniform target velocity as external force; PBD maintains spacing.
  - At each tick, recompute neighbor graph at comms_range = k * (1 + delta).
  - Record: per-tick degree min/median, fraction of drones with <4
    non-coplanar neighbors, mean |edge strain|, centroid position.
  - In the loss scenario, kill drone N//2 at t=KILL_TICK and record how
    many ticks until topology recovers (no violators).

Output: bench_topology_results.json with one record per scenario.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from lattice import (
    build_neighbor_graph,
    hcp_positions,
    non_coplanar_neighbors,
)
from pbd import PBDLattice, edges_from_neighbors

SPACING = 10.0          # meters; nearest-neighbor lattice spacing
COMMS_DELTA = 0.15      # comms range = SPACING * (1 + COMMS_DELTA)
HEX_RADIUS = 3
N_LAYERS = 3
DT = 0.05               # 20 Hz physics
MISSION_S = 60.0
KILL_TICK = 400         # ~20s into the mission


@dataclass
class ScenarioResult:
    name: str
    velocity_mps: float
    delta_ratio: float
    kill_event: bool
    n_drones: int
    n_edges: int
    centroid_drift_m: float
    centroid_target_m: float
    mean_abs_strain: float
    max_abs_strain: float
    pct_ticks_topology_violated: float
    min_degree_observed: int
    recovery_ticks: int  # -1 if no kill or never recovered
    wall_time_s: float


def run_scenario(
    name: str,
    velocity_mps: float,
    delta_ratio: float = COMMS_DELTA,
    kill_event: bool = False,
    seed: int = 0,
) -> ScenarioResult:
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)

    pos0 = hcp_positions(HEX_RADIUS, N_LAYERS, spacing=SPACING)
    # Light initial perturbation so we don't start in a degenerate
    # zero-strain crystalline state (more realistic launch).
    pos0 = pos0 + rng.normal(scale=0.02 * SPACING, size=pos0.shape)
    comms_range = SPACING * (1.0 + delta_ratio)
    initial_neighbors = build_neighbor_graph(pos0, comms_range)
    edges = edges_from_neighbors(initial_neighbors)
    n = pos0.shape[0]

    sim = PBDLattice(
        pos0,
        edges,
        rest_length=SPACING,
        compliance=1e-5,
        damping=0.95,
    )

    # Steady-state motion as a constant acceleration ramp to target velocity.
    target_vel = np.array([velocity_mps, 0.0, 0.0])
    alive = np.ones(n, dtype=bool)

    n_steps = int(MISSION_S / DT)
    violations: list[bool] = []
    abs_strains: list[float] = []
    max_strains: list[float] = []
    min_degrees: list[int] = []
    centroid_x: list[float] = []
    target_x: list[float] = []
    recovery_ticks = -1
    recovered = False

    initial_centroid_x = pos0[:, 0].mean()

    for tick in range(n_steps):
        # Spring-force toward target velocity (gentle acceleration ramp).
        vel_error = target_vel[None, :] - sim.vel
        # Only drive alive drones, gain ~ 0.5 / s.
        force = 0.5 * vel_error
        force[~alive] = 0.0

        sim.step(dt=DT, external_force=force, iters=8, alive=alive)

        if kill_event and tick == KILL_TICK:
            # Kill a central drone to maximize topology stress.
            centroid = sim.pos[alive].mean(axis=0)
            dists = np.linalg.norm(sim.pos - centroid, axis=1)
            dists[~alive] = np.inf
            victim = int(np.argmin(dists))
            alive[victim] = False

        # Topology check on alive subset.
        live_idx = np.where(alive)[0]
        live_pos = sim.pos[live_idx]
        live_neighbors_local = build_neighbor_graph(live_pos, comms_range)
        ncp = non_coplanar_neighbors(live_pos, live_neighbors_local)
        violators = int((ncp < 4).sum())
        violations.append(violators > 0)

        if (
            kill_event
            and not recovered
            and tick > KILL_TICK
            and violators == 0
        ):
            recovery_ticks = tick - KILL_TICK
            recovered = True

        strains = sim.edge_strain()
        abs_strains.append(float(np.mean(np.abs(strains))))
        max_strains.append(float(np.max(np.abs(strains))))
        min_degrees.append(int(min(len(ns) for ns in live_neighbors_local)))
        centroid_x.append(float(sim.pos[alive, 0].mean()))
        target_x.append(float(initial_centroid_x + velocity_mps * tick * DT))

    final_centroid_drift = abs(
        (centroid_x[-1] - initial_centroid_x) - velocity_mps * MISSION_S
    )

    return ScenarioResult(
        name=name,
        velocity_mps=velocity_mps,
        delta_ratio=delta_ratio,
        kill_event=kill_event,
        n_drones=n,
        n_edges=len(edges),
        centroid_drift_m=final_centroid_drift,
        centroid_target_m=velocity_mps * MISSION_S,
        mean_abs_strain=float(np.mean(abs_strains)),
        max_abs_strain=float(np.max(max_strains)),
        pct_ticks_topology_violated=100.0 * np.mean(violations),
        min_degree_observed=min(min_degrees),
        recovery_ticks=recovery_ticks,
        wall_time_s=time.perf_counter() - t_start,
    )


def main() -> None:
    scenarios: list[ScenarioResult] = []

    # (a) Topology vs velocity sweep
    for v in [0.0, 0.25, 0.5, 1.0, 2.0]:
        scenarios.append(
            run_scenario(name=f"hold_v={v}", velocity_mps=v, seed=0)
        )

    # (d) Sensitivity to comms-range slack
    for delta in [0.05, 0.10, 0.20, 0.30]:
        scenarios.append(
            run_scenario(
                name=f"slack_d={delta}",
                velocity_mps=1.0,
                delta_ratio=delta,
                seed=1,
            )
        )

    # (c) Recovery from loss at moderate velocity
    for s in range(3):
        scenarios.append(
            run_scenario(
                name=f"kill_v=1.0_seed={s}",
                velocity_mps=1.0,
                kill_event=True,
                seed=s + 100,
            )
        )

    out = [asdict(r) for r in scenarios]
    with open("bench_topology_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Console summary
    print(f"{'name':28s} {'v':>5s} {'delta':>6s} {'kill':>5s} "
          f"{'%viol':>7s} {'<str>':>7s} {'max str':>8s} {'recov':>6s}")
    for r in scenarios:
        recov = "-" if r.recovery_ticks < 0 else f"{r.recovery_ticks}"
        print(
            f"{r.name:28s} {r.velocity_mps:5.2f} {r.delta_ratio:6.2f} "
            f"{str(r.kill_event):>5s} {r.pct_ticks_topology_violated:7.2f} "
            f"{r.mean_abs_strain:7.4f} {r.max_abs_strain:8.4f} {recov:>6s}"
        )


if __name__ == "__main__":
    main()
