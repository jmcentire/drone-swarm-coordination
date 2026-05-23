# /// script
# dependencies = ["numpy<3"]
# ///
"""Per-tick convergence bench: how each drone's working model grows.

At round 0, each drone has only its own observations -- it sees its
immediate neighbors but knows nothing about anyone else.

At round t, gossip has propagated observations across t hops, so each
drone has accumulated everything originating within t hops. Each drone
can attempt to stitch a partial pose graph from what it has.

Crucially: at intermediate rounds, drones at different points in the
lattice have DIFFERENT working models. A center drone sees more of the
swarm than an edge drone at the same round, because more hops pass through
it. By round D (the gossip diameter), every drone has converged on the
same global picture.

Metrics per round:
  * coverage: per drone, what fraction of the swarm has it stitched?
  * mean position error: across all drones, mean |stitched - true|
  * agreement: across all drones' working models, how consistent are
    their estimates for drones they both know? (standard deviation of
    per-drone position estimates across observers)

Output: bench_convergence_results.json (per-round arrays) and console
summary.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from lattice import build_neighbor_graph, hcp_positions
from mapping import (
    gossip_snapshots,
    observe_neighbors_with_frame,
    random_rotations,
    stitch_global_rigid,
)

SPACING = 10.0
HEX_RADIUS = 3
N_LAYERS = 3
COMMS_DELTA = 0.15
NOISE_STD = 0.05            # 5 cm sensor noise
LOSS_RATE = 0.05            # 5% per-link message loss
MAX_ROUNDS = 12


@dataclass
class RoundStats:
    round_idx: int
    mean_coverage: float            # mean fraction of swarm each drone has stitched
    min_coverage: float
    max_coverage: float
    mean_pos_err_m: float           # only over drones each observer has stitched
    max_pos_err_m: float
    cross_observer_disagreement_m: float  # std of position estimates across observers


def true_positions_in_frame(
    true_pos: np.ndarray, local_frames: np.ndarray, anchor: int
) -> np.ndarray:
    """Convert world-frame true positions to anchor's body frame."""
    return ((local_frames[anchor].T @ (true_pos - true_pos[anchor]).T).T)


def main(seed: int = 0) -> list[RoundStats]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    true_pos = hcp_positions(HEX_RADIUS, N_LAYERS, spacing=SPACING)
    n = true_pos.shape[0]
    neighbors = build_neighbor_graph(true_pos, comms_range=SPACING * (1 + COMMS_DELTA))
    local_frames = random_rotations(n, rng)

    local_obs = observe_neighbors_with_frame(
        true_pos, neighbors, local_frames, noise_std=NOISE_STD, rng=rng
    )
    snapshots = gossip_snapshots(
        local_obs, neighbors, n_rounds=MAX_ROUNDS, loss_rate=LOSS_RATE, rng=rng
    )

    results: list[RoundStats] = []
    for round_idx, knowledge in enumerate(snapshots):
        per_observer_coverage: list[float] = []
        per_observer_errors: list[float] = []
        # For cross-observer disagreement: collect estimates per (observer, observed)
        # estimates[observed] = list of (observer, estimated_anchor_frame_position)
        estimates: dict[int, list[np.ndarray]] = {}
        # Use one global anchor (drone 0) for cross-observer comparison: each
        # observer stitches its own map in drone-0's frame by computing
        # transforms relative to drone-0 if it has any path. If drone 0 is
        # not reachable from observer's knowledge, skip this observer for
        # the agreement metric.
        truth_in_0 = true_positions_in_frame(true_pos, local_frames, anchor=0)

        for observer, obs in knowledge.items():
            # Build observer's local stitched map with observer as its own anchor.
            frames = stitch_global_rigid(obs, anchor=observer)
            coverage = len(frames) / n
            per_observer_coverage.append(coverage)

            # Position error: compare observer's stitched positions to true
            # positions in observer's own body frame.
            truth_in_observer = true_positions_in_frame(
                true_pos, local_frames, anchor=observer
            )
            for drone_id, (_, pos) in frames.items():
                err = np.linalg.norm(pos - truth_in_observer[drone_id])
                per_observer_errors.append(err)

            # For cross-observer agreement: re-stitch this observer's map
            # using drone 0 as the anchor if drone 0 is reachable from
            # observer's accumulated obs.
            if 0 in obs.keys().__iter__():
                pass  # placeholder
            try:
                frames_in_0 = stitch_global_rigid(obs, anchor=0)
                for drone_id, (_, pos) in frames_in_0.items():
                    estimates.setdefault(drone_id, []).append(pos)
            except Exception:
                pass

        # Cross-observer disagreement: per drone_id, std of estimates across
        # all observers that have an estimate. Average across drone_ids.
        disagreements: list[float] = []
        for drone_id, ests in estimates.items():
            if len(ests) >= 2:
                arr = np.array(ests)
                # Disagreement = mean pairwise distance / 2 (rough proxy).
                # Use std of the magnitude of (each estimate - mean).
                mean_est = arr.mean(axis=0)
                spread = np.mean(np.linalg.norm(arr - mean_est, axis=1))
                disagreements.append(spread)
        agreement_metric = float(np.mean(disagreements)) if disagreements else float("nan")

        cov_arr = np.array(per_observer_coverage)
        err_arr = (
            np.array(per_observer_errors) if per_observer_errors else np.array([0.0])
        )
        results.append(
            RoundStats(
                round_idx=round_idx,
                mean_coverage=float(cov_arr.mean()),
                min_coverage=float(cov_arr.min()),
                max_coverage=float(cov_arr.max()),
                mean_pos_err_m=float(err_arr.mean()),
                max_pos_err_m=float(err_arr.max()),
                cross_observer_disagreement_m=agreement_metric,
            )
        )

    print(
        f"{'round':>5s} {'cov mean':>10s} {'cov min':>9s} {'cov max':>9s} "
        f"{'err mean':>10s} {'err max':>9s} {'disagree':>10s}"
    )
    for r in results:
        print(
            f"{r.round_idx:5d} "
            f"{r.mean_coverage:10.3f} {r.min_coverage:9.3f} {r.max_coverage:9.3f} "
            f"{r.mean_pos_err_m:10.3f} {r.max_pos_err_m:9.3f} "
            f"{r.cross_observer_disagreement_m:10.3f}"
        )
    print(f"wall time: {time.perf_counter()-t0:.1f}s")

    with open("bench_convergence_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    return results


if __name__ == "__main__":
    main()
