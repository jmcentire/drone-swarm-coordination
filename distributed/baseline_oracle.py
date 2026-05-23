# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Oracle baseline: upper bound on what's achievable with global knowledge.

This is the ground-truth comparator. It implements the same per-drone
divide-and-conquer formation but feeds every drone the FULL global drone
set instead of its locally-gossipped subset. The oracle baseline still
suffers from physics (drones must move toward targets), but it never
suffers from comms loss, propagation delay, partition, or divergent
local views.

A distributed protocol that beats this is BUGGED. A distributed protocol
that matches this exactly is suspicious (likely has an oracle leak). A
distributed protocol that's worse but degrades sensibly is the expected
shape.
"""

from __future__ import annotations

import numpy as np

from manifold import ManifoldNode, compute_target


def run_oracle(
    n_drones: int,
    start_positions: np.ndarray,
    manifold: np.ndarray,
    n_ticks: int,
    physics_dt: float = 1.0,
    max_speed: float = 0.8,
    repulsion_radius_m: float = 3.5,
    approach_radius_m: float = 4.0,
    alive_schedule: dict[int, list[int]] | None = None,
) -> dict:
    """Returns metrics dict with same shape as World.metrics."""
    positions = start_positions.copy()
    alive = np.ones(n_drones, dtype=bool)
    locked = np.zeros(n_drones, dtype=bool)
    tree = ManifoldNode(manifold)

    metrics = {
        "formation_error_mean": [],
        "formation_error_max": [],
        "fraction_alive": [],
        "n_collisions": [],
        "coverage_frac": [],
    }

    for tick in range(n_ticks):
        # Process kill schedule.
        if alive_schedule is not None:
            kills = alive_schedule.get(tick, [])
            for k in kills:
                if 0 <= k < n_drones:
                    alive[k] = False

        # Build the global drone list (THIS is the oracle leak in the
        # baseline — every drone sees every alive drone).
        drones = [
            {"id": int(i), "pos": positions[i].copy()}
            for i in range(n_drones) if alive[i]
        ]

        # Pre-pass: unlock drones whose target has shifted (kill rebalanced
        # the bisection). Without this, surplus drones can't migrate.
        for i in range(n_drones):
            if not alive[i] or not locked[i]:
                continue
            t, _ = compute_target(i, drones, tree)
            if float(np.linalg.norm(positions[i] - t)) > 1.5 * approach_radius_m:
                locked[i] = False

        new_pos = positions.copy()
        for i in range(n_drones):
            if not alive[i] or locked[i]:
                continue
            target, is_primary = compute_target(i, drones, tree)
            diff = target - positions[i]
            dist = float(np.linalg.norm(diff))
            is_final = dist < approach_radius_m
            attr = (diff / dist) * min(0.6, dist * 0.1) if dist > 1e-9 else np.zeros(3)
            rep = np.zeros(3)
            eff_rep = repulsion_radius_m * (0.4 if is_final else 1.0)
            for j in range(n_drones):
                if j == i or not alive[j]:
                    continue
                d = positions[i] - positions[j]
                r = float(np.linalg.norm(d))
                if 0 < r < eff_rep:
                    rep += (d / r) * ((eff_rep - r) / r) * 0.15
            v = attr + rep
            eff_max = max_speed if not is_final else max(0.05, dist * 0.2)
            s = float(np.linalg.norm(v))
            if s > eff_max:
                v = (v / s) * eff_max
            new_pos[i] = positions[i] + v * physics_dt
            if is_final and dist < 0.3:
                new_pos[i] = target
                locked[i] = True
        positions = new_pos

        # Metrics.
        if alive.any():
            live_pos = positions[alive]
            mt = manifold
            errs = []
            claimed = set()
            order = np.argsort(np.linalg.norm(live_pos[:, None, :] - mt[None, :, :], axis=-1).min(axis=1))
            for k in order:
                d = np.linalg.norm(live_pos[k] - mt, axis=1)
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

        # Coverage: targets within 3m of any alive drone.
        if alive.any():
            live_pos = positions[alive]
            d_t = np.linalg.norm(manifold[:, None, :] - live_pos[None, :, :], axis=-1).min(axis=1)
            metrics["coverage_frac"].append(float((d_t < 3.0).sum() / len(manifold)))
        else:
            metrics["coverage_frac"].append(0.0)

    return {
        "metrics": metrics,
        "final_positions": positions,
        "final_alive": alive,
    }
