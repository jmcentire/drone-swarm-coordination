# /// script
# dependencies = ["numpy<3"]
# ///
"""Distributed pose graph construction via local observation + gossip.

Each drone observes its immediate neighbors with noise. It broadcasts
those observations to its gossip neighbors. After D rounds, every drone
has accumulated the full observation set and can stitch a global pose
graph.

Two stitching modes:
  * Translation-only (stitch_global, stitch_averaged): assumes all drones
    share an orientation reference (compass + accelerometer + maybe
    gyrocompass). Simpler; baseline for comparison.
  * Full rigid frame consensus (stitch_global_rigid): each drone has its
    own arbitrary body frame. Relative rotations between frames are
    recovered via Procrustes/Kabsch on shared observed landmarks. After
    BFS propagation from a deterministic anchor, every reachable drone
    has both a position and a rotation in the anchor's frame.

The graph grows per gossip round. At round t each drone has accumulated
observations whose originator is within t hops. So at t=0 a drone knows
its immediate neighborhood; at t=D it has the full swarm. The benches
expose this "working model grows" property explicitly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalObservation:
    """One drone's noisy estimate of one neighbor's relative position."""
    observer: int
    observed: int
    relative_position: tuple[float, float, float]


def observe_neighbors(
    positions: np.ndarray,
    neighbors: list[list[int]],
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[int, list[LocalObservation]]:
    """For each drone, observe all neighbors with isotropic Gaussian noise.

    Returns: observer_id -> list of LocalObservation.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs: dict[int, list[LocalObservation]] = {}
    for i in range(positions.shape[0]):
        obs[i] = []
        for j in neighbors[i]:
            rel = positions[j] - positions[i]
            if noise_std > 0:
                rel = rel + rng.normal(scale=noise_std, size=3)
            obs[i].append(
                LocalObservation(
                    observer=i,
                    observed=j,
                    relative_position=(float(rel[0]), float(rel[1]), float(rel[2])),
                )
            )
    return obs


def gossip(
    local_obs: dict[int, list[LocalObservation]],
    neighbors: list[list[int]],
    n_rounds: int,
    loss_rate: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[int, dict[tuple[int, int], LocalObservation]]:
    """Run n_rounds of observation gossip.

    After each round, each drone has accumulated observations whose hop
    count from their originator is <= round_number.

    Returns: drone_id -> {(observer, observed) -> LocalObservation}.
    The map is indexed by (observer, observed) so duplicates dedupe naturally.
    """
    if rng is None:
        rng = np.random.default_rng(1)

    # Initial knowledge: each drone knows its own observations.
    knowledge: dict[int, dict[tuple[int, int], LocalObservation]] = {}
    for i, obs_list in local_obs.items():
        knowledge[i] = {(o.observer, o.observed): o for o in obs_list}

    for _ in range(n_rounds):
        # Snapshot so all exchanges in this round see the same "before" state.
        prev = {i: dict(k) for i, k in knowledge.items()}
        for i in range(len(neighbors)):
            for j in neighbors[i]:
                if loss_rate > 0 and rng.random() < loss_rate:
                    continue
                # j tells i everything j knows (deduplicated by key).
                for key, ob in prev[j].items():
                    if key not in knowledge[i]:
                        knowledge[i][key] = ob

    return knowledge


def stitch_global(
    observations: dict[tuple[int, int], LocalObservation],
    anchor: int,
    n_drones: int,
) -> dict[int, np.ndarray]:
    """BFS from anchor through observations to assign each drone a global
    position. Anchor is at origin; children are anchor's observed offset.

    Drones unreachable from the anchor are not included in the result.
    """
    # Build outgoing observation index: observer -> list of (observed, offset).
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    for (obs_i, obs_j), ob in observations.items():
        rel = np.array(ob.relative_position, dtype=np.float64)
        out.setdefault(obs_i, []).append((obs_j, rel))

    global_pos: dict[int, np.ndarray] = {anchor: np.zeros(3)}
    queue: deque[int] = deque([anchor])
    while queue:
        u = queue.popleft()
        for v, rel in out.get(u, []):
            if v not in global_pos:
                global_pos[v] = global_pos[u] + rel
                queue.append(v)
    return global_pos


def stitch_averaged(
    observations: dict[tuple[int, int], LocalObservation],
    anchor: int,
    n_drones: int,
) -> dict[int, np.ndarray]:
    """Better stitcher: average all paths from anchor to each drone.

    For each drone v, its position estimate is the average of:
      pos(u) + offset(u -> v)   for every u that observed v
    (recursively, with pos(anchor) := 0).

    This distributes noise across multiple paths rather than committing to
    one BFS branch. Converges by iterative relaxation.
    """
    incoming: dict[int, list[tuple[int, np.ndarray]]] = {}
    for (obs_i, obs_j), ob in observations.items():
        rel = np.array(ob.relative_position, dtype=np.float64)
        incoming.setdefault(obs_j, []).append((obs_i, rel))

    # Initialize with BFS estimate so we have a starting point for every
    # reachable drone.
    pos = stitch_global(observations, anchor, n_drones)

    # Relax: each drone's position = mean over (observer_pos + offset) for
    # all observers that saw it. Anchor stays pinned at origin.
    for _ in range(20):
        new_pos = dict(pos)
        for v, in_list in incoming.items():
            if v == anchor:
                continue
            ests = []
            for u, rel in in_list:
                if u in pos:
                    ests.append(pos[u] + rel)
            if ests:
                new_pos[v] = np.mean(ests, axis=0)
        pos = new_pos
    return pos


# ---------------------------------------------------------------------------
# Full frame consensus: each drone has its own arbitrary body frame.
# ---------------------------------------------------------------------------


def random_rotations(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n random rotation matrices uniform on SO(3) via quaternions."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def observe_neighbors_with_frame(
    positions: np.ndarray,
    neighbors: list[list[int]],
    local_frames: np.ndarray,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[int, list[LocalObservation]]:
    """Each drone observes neighbors in its own rotated body frame.

    A neighbor at world position q seen from drone i at world position p
    appears in i's local frame at: local_frames[i].T @ (q - p) + noise.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs: dict[int, list[LocalObservation]] = {}
    for i in range(positions.shape[0]):
        obs[i] = []
        R_i_T = local_frames[i].T  # world -> local
        for j in neighbors[i]:
            world_offset = positions[j] - positions[i]
            local_offset = R_i_T @ world_offset
            if noise_std > 0:
                local_offset = local_offset + rng.normal(scale=noise_std, size=3)
            obs[i].append(
                LocalObservation(
                    observer=i,
                    observed=j,
                    relative_position=(
                        float(local_offset[0]),
                        float(local_offset[1]),
                        float(local_offset[2]),
                    ),
                )
            )
    return obs


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray | None:
    """Find rotation R minimizing sum ||R @ P_i - Q_i||^2.

    P, Q: (M, 3) arrays of corresponding direction vectors (NOT centered;
    we want the pure rotation that aligns them).

    Returns R (3, 3) or None if underdetermined.
    """
    if P.shape[0] < 2:
        return None
    H = P.T @ Q
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def stitch_global_rigid(
    observations: dict[tuple[int, int], LocalObservation],
    anchor: int,
    min_shared_landmarks: int = 3,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Recover (rotation, position) in anchor's frame for every reachable drone.

    Returns: drone_id -> (R_anchor_from_drone, position_in_anchor_frame).
    The drone's own body-frame measurement of a point x can be transformed
    to anchor frame via: anchor_pt = R @ x + position.

    Anchor's frame is the reference: R[anchor] = I, position[anchor] = 0.
    """
    # Group observations by observer for quick lookup.
    by_observer: dict[int, dict[int, np.ndarray]] = {}
    for (oi, oj), ob in observations.items():
        by_observer.setdefault(oi, {})[oj] = np.array(
            ob.relative_position, dtype=np.float64
        )

    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {
        anchor: (np.eye(3), np.zeros(3))
    }

    # Repeatedly try to extend: for each unframed drone that we can fix
    # using a framed observer + enough shared landmarks, add it.
    changed = True
    while changed:
        changed = False
        for v in list(by_observer.keys()):
            if v in frames:
                continue
            # Find a framed neighbor u that observed v and shares landmarks
            # with v that we already have placed in the anchor frame.
            best: tuple[int, np.ndarray, np.ndarray] | None = None
            for u in frames:
                if u not in by_observer or v not in by_observer[u]:
                    continue
                # u observed v; we can derive v's position in anchor frame
                R_u, t_u = frames[u]
                v_pos = R_u @ by_observer[u][v] + t_u
                # Now find v's rotation from shared observed landmarks.
                # In v's local frame: by_observer[v][s] for each shared s.
                # In anchor frame: known position of s.
                v_obs = by_observer[v]
                shared_pts_v: list[np.ndarray] = []
                shared_pts_anchor: list[np.ndarray] = []
                # Direct: u itself is "observed" by v if v observed u.
                if u in v_obs:
                    shared_pts_v.append(v_obs[u])
                    shared_pts_anchor.append(t_u - v_pos)  # u relative to v in anchor frame
                # Other landmarks: drones both v and u observed AND that we
                # already have a position for in anchor frame.
                for s in v_obs:
                    if s == u or s == v:
                        continue
                    # Compute s's anchor-frame position via u's frame if u observed s.
                    if s in by_observer.get(u, {}):
                        s_anchor = R_u @ by_observer[u][s] + t_u
                        shared_pts_v.append(v_obs[s])
                        shared_pts_anchor.append(s_anchor - v_pos)
                    elif s in frames:
                        s_anchor = frames[s][1]
                        shared_pts_v.append(v_obs[s])
                        shared_pts_anchor.append(s_anchor - v_pos)
                if len(shared_pts_v) < min_shared_landmarks:
                    continue
                P = np.array(shared_pts_v)
                Q = np.array(shared_pts_anchor)
                R_v = _kabsch_rotation(P, Q)
                if R_v is None:
                    continue
                best = (v, R_v, v_pos)
                break
            if best is not None:
                _, R_v, v_pos = best
                frames[best[0]] = (R_v, v_pos)
                changed = True
    return frames


def stitch_per_round(
    knowledge_per_round: list[dict[int, dict[tuple[int, int], LocalObservation]]],
    anchor: int,
    rigid: bool,
) -> list[dict[int, dict[int, np.ndarray]]]:
    """For each gossip round and each drone, stitch what that drone knows
    so far. Returns list-indexed by round: drone_id -> {observed_id -> position_in_anchor_frame}.

    Anchor is used per drone -- each drone uses itself as anchor (so we see
    its locally-stitched map).
    """
    per_round: list[dict[int, dict[int, np.ndarray]]] = []
    for round_idx, knowledge in enumerate(knowledge_per_round):
        round_result: dict[int, dict[int, np.ndarray]] = {}
        for drone_id, obs in knowledge.items():
            if rigid:
                frames = stitch_global_rigid(obs, anchor=drone_id)
                round_result[drone_id] = {k: v[1] for k, v in frames.items()}
            else:
                pos_map = stitch_averaged(obs, anchor=drone_id, n_drones=0)
                round_result[drone_id] = pos_map
        per_round.append(round_result)
    return per_round


def gossip_snapshots(
    local_obs: dict[int, list[LocalObservation]],
    neighbors: list[list[int]],
    n_rounds: int,
    loss_rate: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[dict[int, dict[tuple[int, int], LocalObservation]]]:
    """Like gossip(), but returns a snapshot of every drone's knowledge
    after each round. snapshots[t][drone_id] is what drone_id knows after
    t rounds (snapshots[0] is the initial state -- each drone has only its
    own observations).
    """
    if rng is None:
        rng = np.random.default_rng(1)
    snapshots: list[dict[int, dict[tuple[int, int], LocalObservation]]] = []
    knowledge: dict[int, dict[tuple[int, int], LocalObservation]] = {
        i: {(o.observer, o.observed): o for o in obs_list}
        for i, obs_list in local_obs.items()
    }
    snapshots.append({i: dict(k) for i, k in knowledge.items()})
    for _ in range(n_rounds):
        prev = {i: dict(k) for i, k in knowledge.items()}
        for i in range(len(neighbors)):
            for j in neighbors[i]:
                if loss_rate > 0 and rng.random() < loss_rate:
                    continue
                for key, ob in prev[j].items():
                    if key not in knowledge[i]:
                        knowledge[i][key] = ob
        snapshots.append({i: dict(k) for i, k in knowledge.items()})
    return snapshots


if __name__ == "__main__":
    # Smoke test: observe a noisy lattice, gossip, stitch, measure error.
    from lattice import build_neighbor_graph, hcp_positions

    spacing = 10.0
    true_pos = hcp_positions(hex_radius=3, n_layers=3, spacing=spacing)
    n = true_pos.shape[0]
    neighbors = build_neighbor_graph(true_pos, comms_range=spacing * 1.15)

    rng = np.random.default_rng(42)
    for noise in [0.0, 0.05, 0.20, 0.50]:
        local = observe_neighbors(true_pos, neighbors, noise_std=noise, rng=rng)
        knowledge = gossip(local, neighbors, n_rounds=10)
        # Stitch from the perspective of drone 0.
        anchor_view = knowledge[0]
        bfs = stitch_global(anchor_view, anchor=0, n_drones=n)
        avg = stitch_averaged(anchor_view, anchor=0, n_drones=n)

        # Compare to true positions (translated so drone 0 is at origin).
        true_centered = true_pos - true_pos[0]

        bfs_err = np.array(
            [np.linalg.norm(bfs[i] - true_centered[i]) for i in range(n) if i in bfs]
        )
        avg_err = np.array(
            [np.linalg.norm(avg[i] - true_centered[i]) for i in range(n) if i in avg]
        )
        print(
            f"[translation-only] noise={noise:.2f}m  reached={len(bfs)}/{n}  "
            f"bfs mean={bfs_err.mean():.3f}m max={bfs_err.max():.3f}m  "
            f"averaged mean={avg_err.mean():.3f}m max={avg_err.max():.3f}m"
        )

    # Frame consensus test: each drone has its own random rotation.
    print()
    rng = np.random.default_rng(7)
    local_frames = random_rotations(n, rng)
    for noise in [0.0, 0.05, 0.20]:
        obs_local = observe_neighbors_with_frame(
            true_pos, neighbors, local_frames, noise_std=noise, rng=rng
        )
        knowledge = gossip(obs_local, neighbors, n_rounds=10)
        anchor_view = knowledge[0]
        frames = stitch_global_rigid(anchor_view, anchor=0)
        # Compare recovered positions to true positions in drone-0's frame.
        # True positions in drone-0 anchor frame:
        true_centered = (
            (local_frames[0].T @ (true_pos - true_pos[0]).T).T
        )
        err_list = []
        rot_err_list = []
        for i in range(n):
            if i in frames:
                R_i, t_i = frames[i]
                err_list.append(np.linalg.norm(t_i - true_centered[i]))
                # Rotation error: rotation that maps recovered to true should be near identity.
                # True R: anchor_frame -> drone i's frame is local_frames[0].T @ local_frames[i]
                true_R = local_frames[0].T @ local_frames[i]
                R_err = R_i.T @ true_R  # should be identity
                # Angle of rotation from identity: arccos((trace - 1) / 2)
                cos_a = (np.trace(R_err) - 1) / 2
                cos_a = max(-1.0, min(1.0, cos_a))
                rot_err_list.append(np.degrees(np.arccos(cos_a)))
        err_arr = np.array(err_list)
        rot_arr = np.array(rot_err_list)
        print(
            f"[rigid frame] noise={noise:.2f}m  reached={len(frames)}/{n}  "
            f"pos mean={err_arr.mean():.3f}m max={err_arr.max():.3f}m  "
            f"rot mean={rot_arr.mean():.2f}deg max={rot_arr.max():.2f}deg"
        )
