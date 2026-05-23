# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Debug: what's actually violating after a kill?"""

import numpy as np

from failures import select_random_drones
from lattice import build_neighbor_graph, hcp_positions, non_coplanar_neighbors
from mission import MissionConfig, MissionState, Phase, mission_force, update_mission
from pbd import PBDLattice, edges_from_neighbors

SPACING = 10.0
DT = 0.05


def main() -> None:
    rng = np.random.default_rng(2000)
    pos0 = hcp_positions(3, 3, spacing=SPACING)
    n = pos0.shape[0]
    pos0 = pos0 + rng.normal(scale=0.02 * SPACING, size=pos0.shape)
    comms_range = SPACING * 1.15
    base_neighbors = build_neighbor_graph(pos0, comms_range)
    edges = edges_from_neighbors(base_neighbors)

    cfg = MissionConfig(spacing=SPACING)
    initial_centroid = pos0.mean(axis=0)
    state = MissionState(leg_start_centroid=initial_centroid.copy(),
                         last_known_centroid=initial_centroid.copy())
    alive = np.ones(n, dtype=bool)
    sim = PBDLattice(pos0, edges, rest_length=SPACING, compliance=1e-5, damping=0.95)

    # Use the same kill schedule as bench seed=2000 loss_mid_move.
    victims_rng = np.random.default_rng(2000)
    victims = list(select_random_drones(n, 3, victims_rng))
    print(f"victims (will die at tick 1200): {victims}")

    for tick in range(2000):
        if tick == 1200:
            for v in victims:
                alive[v] = False
            print(f"\n*** KILL at tick {tick}: drones {victims} dead ***")

        is_violating = np.zeros(n, dtype=bool)
        current_degree = np.zeros(n, dtype=np.int64)
        ncp_full = np.zeros(n, dtype=np.int64)
        live_idx = np.where(alive)[0]
        live_pos = sim.pos[live_idx]
        ngh = build_neighbor_graph(live_pos, comms_range)
        ncp = non_coplanar_neighbors(live_pos, ngh)
        for k, did in enumerate(live_idx):
            current_degree[did] = len(ngh[k])
            ncp_full[did] = ncp[k]
            if ncp[k] < 4:
                is_violating[did] = True

        update_mission(state, cfg, tick, DT, sim.pos, sim.vel, alive, is_violating, current_degree)
        force = mission_force(state, cfg, sim.pos, sim.vel, alive, current_degree)
        sim.step(dt=DT, external_force=force, iters=8, alive=alive)

        if tick in (1199, 1200, 1201, 1210, 1250, 1300, 1500, 1800):
            viol_drones = np.where(is_violating)[0]
            ncp_viol = ncp_full[viol_drones]
            deg_viol = current_degree[viol_drones]
            print(
                f"tick {tick} phase={state.phase.value}: "
                f"n_violating={len(viol_drones)} of {alive.sum()} alive; "
                f"min_ncp={ncp_full[alive].min()} min_deg={current_degree[alive].min()}"
            )
            if len(viol_drones) > 0 and len(viol_drones) < 10:
                for v, n_c, d in zip(viol_drones, ncp_viol, deg_viol):
                    print(f"  drone {v}: ncp={n_c} degree={d}")


if __name__ == "__main__":
    main()
