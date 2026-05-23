# /// script
# dependencies = ["numpy<3"]
# ///
"""Drift baseline: lower bound when there is no protocol.

Drones get an initial waypoint at t=0 and then drift with their initial
velocity, applying only inter-drone collision avoidance. No gossip, no
leader, no re-assignment, no recovery on loss. The metric numbers from
this run define "what happens without the protocol" — a distributed
protocol must do better than this to be worth running.
"""

from __future__ import annotations

import numpy as np


def run_drift(
    n_drones: int,
    start_positions: np.ndarray,
    manifold: np.ndarray,
    n_ticks: int,
    physics_dt: float = 1.0,
    max_speed: float = 0.8,
    repulsion_radius_m: float = 3.5,
    alive_schedule: dict[int, list[int]] | None = None,
) -> dict:
    """Each drone is assigned to manifold[i % len(manifold)] at t=0 and
    moves straight toward it. No further re-assignment. If its target
    drone dies (in fancier setups), it does nothing about it.

    No gossip. No leader. Pure ballistic motion with collision avoidance.
    """
    n_targets = len(manifold)
    # Static assignment by index — the "stupid" no-protocol baseline.
    targets = np.array([manifold[i % n_targets] for i in range(n_drones)])
    positions = start_positions.copy()
    alive = np.ones(n_drones, dtype=bool)

    metrics = {
        "formation_error_mean": [],
        "formation_error_max": [],
        "fraction_alive": [],
        "n_collisions": [],
        "coverage_frac": [],
    }

    for tick in range(n_ticks):
        if alive_schedule is not None:
            for k in alive_schedule.get(tick, []):
                if 0 <= k < n_drones:
                    alive[k] = False

        new_pos = positions.copy()
        for i in range(n_drones):
            if not alive[i]:
                continue
            diff = targets[i] - positions[i]
            dist = float(np.linalg.norm(diff))
            if dist < 1e-9:
                continue
            attr = (diff / dist) * min(0.6, dist * 0.1)
            rep = np.zeros(3)
            for j in range(n_drones):
                if j == i or not alive[j]:
                    continue
                d = positions[i] - positions[j]
                r = float(np.linalg.norm(d))
                if 0 < r < repulsion_radius_m:
                    rep += (d / r) * ((repulsion_radius_m - r) / r) * 0.15
            v = attr + rep
            s = float(np.linalg.norm(v))
            if s > max_speed:
                v = (v / s) * max_speed
            new_pos[i] = positions[i] + v * physics_dt
        positions = new_pos

        # Metrics (same as oracle).
        if alive.any():
            live_pos = positions[alive]
            errs = []
            claimed = set()
            order = np.argsort(np.linalg.norm(live_pos[:, None, :] - manifold[None, :, :], axis=-1).min(axis=1))
            for k in order:
                d = np.linalg.norm(live_pos[k] - manifold, axis=1)
                for cand in np.argsort(d):
                    if int(cand) not in claimed:
                        claimed.add(int(cand))
                        errs.append(float(d[cand]))
                        break
            metrics["formation_error_mean"].append(float(np.mean(errs)))
            metrics["formation_error_max"].append(float(np.max(errs)))
        else:
            metrics["formation_error_mean"].append(0.0)
            metrics["formation_error_max"].append(0.0)
        metrics["fraction_alive"].append(float(alive.sum() / max(1, n_drones)))
        if alive.sum() >= 2:
            lp = positions[alive]
            d = np.linalg.norm(lp[:, None, :] - lp[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            metrics["n_collisions"].append(int((d < 1.0).sum() // 2))
        else:
            metrics["n_collisions"].append(0)

        if alive.any():
            live_pos = positions[alive]
            d_t = np.linalg.norm(manifold[:, None, :] - live_pos[None, :, :], axis=-1).min(axis=1)
            metrics["coverage_frac"].append(float((d_t < 3.0).sum() / len(manifold)))
        else:
            metrics["coverage_frac"].append(0.0)

    return {"metrics": metrics, "final_positions": positions, "final_alive": alive}
