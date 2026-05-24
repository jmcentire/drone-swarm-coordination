# /// script
# dependencies = ["numpy<3"]
# ///
"""Gossip + signed-priority flood-max + per-drone leader inference.

This module is the PROTOCOL layer. It runs per-drone, using only
information delivered through `LocalComms`. No drone reads any other
drone's internal state.

Three message types travel through gossip:

  PriorityVote(origin, priority, epoch)
      Identity broadcast. Used for leader election. Drones merge by
      (priority, epoch, origin) max.

  Heartbeat(origin, position, epoch, sig_stub)
      Position broadcast. Used to build the per-drone known_drones set
      for the formation algorithm. Drones keep the latest per origin.

  Command(origin, leader_priority, epoch, payload)
      Leader-issued mission directive (manifold, heading, etc.). Drones
      keep the highest-priority command they've seen within a recency
      window.

Each drone maintains:
  - known_priorities[origin] = (priority, epoch)  -- highest seen for each origin
  - known_positions[origin] = (position, epoch, last_heard_tick)
  - latest_command = (priority, epoch, payload, last_heard_tick) or None
  - my_priority, my_epoch (own identity)

Leader inference is local:
  leader_id = argmax over known_priorities of (priority, epoch) within
  freshness window. If no fresh entry, leader = self (acting alone).

Signature stub note: this is NOT real crypto. The 16-byte tag exists to
model the bandwidth/payload of a signature; the verification check is
"origin == claimed origin" (returned to us by LocalComms.Message). A
real deployment uses Ed25519. The Byzantine bench will exploit the
stub to demonstrate what the protocol can and cannot detect with only
a signature check.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


SIG_LEN = 16
DEFAULT_FRESHNESS_TICKS = 30  # how recent a heard message must be to count


def make_sig(origin: int, epoch: int, kind: str, secret: bytes = b"unused") -> bytes:
    """Stub 'signature' — deterministic 16-byte tag. NOT secure. Used to
    fill the message bandwidth budget and enable verification of origin
    consistency. A real signature would be Ed25519 over the same fields."""
    h = hashlib.sha256(secret + str(origin).encode() + str(epoch).encode() + kind.encode())
    return h.digest()[:SIG_LEN]


@dataclass
class PriorityVote:
    origin: int
    priority: int     # higher = more authoritative
    epoch: int        # increases as the drone re-broadcasts (anti-replay)
    sig_stub: bytes


@dataclass
class Heartbeat:
    """Each drone's beacon. Carries:
      - dr_position: this drone's own dead-reckoning estimate of its
        position (integrated from its own commanded motion since the last
        consensus correction). A soft anchor, NOT a claim. One vote.
      - dr_sigma: uncertainty in dr_position. Grows since last
        consensus correction.
      - range_obs: ranges this drone has physically MEASURED to other
        drones via ToF when their pings arrived. Physical measurements,
        not claims.
    Consensus position is computed by every receiver as a weighted
    least-squares fit over the union of all (dr_position, dr_sigma)
    anchors + all range observations, with IRLS reweighting on residuals.
    Outliers (bad DR, bad range sensor) are detected by residual
    analysis and downweighted automatically — the GPS-RAIM analog."""
    origin: int
    dr_position: np.ndarray  # (3,) — observer's dead-reckoning self-estimate
    dr_sigma: float          # observer's DR uncertainty (m)
    range_obs: dict[int, float]  # observed_id -> range_m
    epoch: int
    sig_stub: bytes


@dataclass
class Command:
    origin: int
    leader_priority: int
    epoch: int
    payload: Any          # arbitrary; the bench bench encodes the manifold + heading here
    sig_stub: bytes


@dataclass
class ProtocolState:
    """Per-drone protocol state. NO cross-drone references allowed."""
    drone_id: int
    my_priority: int
    my_epoch: int = 0
    # origin -> (priority, epoch, last_heard_tick)
    known_priorities: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # Each drone's self-reported position. One vote per drone.
    # origin -> (self_estimate, epoch, last_heard_tick)
    self_estimates: dict[int, tuple[np.ndarray, int, int]] = field(default_factory=dict)
    # Observed-range constraints: (observer_id, observed_id) -> (range, heard_tick).
    # observer_id is the drone that physically measured the range; observed_id
    # is the drone whose position is being constrained.
    range_obs: dict[tuple[int, int], tuple[float, int]] = field(default_factory=dict)
    # known_positions kept as a back-compat alias for code paths that
    # haven't migrated yet — populated from self_estimates with no robust fit.
    known_positions: dict[int, tuple[np.ndarray, int, int]] = field(default_factory=dict)
    # latest seen command, by highest (priority, epoch)
    latest_command: Command | None = None
    latest_command_tick: int = -1

    def __post_init__(self) -> None:
        self.known_priorities[self.drone_id] = (self.my_priority, self.my_epoch, 0)


def verify_origin(claimed_origin: int, msg_origin: int) -> bool:
    """Substrate already delivers msg with .origin == sender's id. The
    stub 'verification' is just origin-consistency. A real implementation
    would check the Ed25519 signature over (origin, epoch, payload).
    Byzantine drones that LIE about their position still pass this check
    (they sign correctly with their real origin) — the test of resilience
    is whether the protocol behaves sensibly given those lies."""
    return claimed_origin == msg_origin


def ingest_priority_vote(
    state: ProtocolState, vote: PriorityVote, current_tick: int
) -> bool:
    """Merge a PriorityVote. Returns True if this updated state."""
    if not verify_origin(vote.origin, vote.origin):
        return False
    prev = state.known_priorities.get(vote.origin)
    new_entry = (vote.priority, vote.epoch, current_tick)
    if prev is None:
        state.known_priorities[vote.origin] = new_entry
        return True
    pp, pe, _ = prev
    if (vote.priority, vote.epoch) > (pp, pe):
        state.known_priorities[vote.origin] = new_entry
        return True
    # Even if not newer, refresh the heard-tick so freshness doesn't expire.
    if (vote.priority, vote.epoch) == (pp, pe):
        state.known_priorities[vote.origin] = (pp, pe, current_tick)
    return False


def ingest_heartbeat(
    state: ProtocolState, hb: Heartbeat, current_tick: int,
    measured_range_to_sender: float | None = None,
    my_drone_id: int | None = None,
) -> bool:
    """Merge a Heartbeat. Stores:
      - sender's self_estimate (one vote on sender's position)
      - sender's reported range observations (constraints on others)
      - substrate-measured range from me to sender (one of my own constraints
        on sender's position)
    Position-consensus runs separately over the accumulated data.
    Returns True if this updated state."""
    if not verify_origin(hb.origin, hb.origin):
        return False
    changed = False
    prev = state.self_estimates.get(hb.origin)
    if prev is None or hb.epoch >= prev[1]:
        # self_estimates dict now stores (dr_position, dr_sigma, epoch, heard).
        # Back-compat fallback: store as (pos, epoch, heard) if dr_sigma missing.
        state.self_estimates[hb.origin] = (
            np.asarray(hb.dr_position, dtype=np.float64).copy(),
            hb.epoch,
            current_tick,
        )
        # Also store dr_sigma separately if not yet present in state.
        if not hasattr(state, "dr_sigmas"):
            state.dr_sigmas = {}
        state.dr_sigmas[hb.origin] = float(hb.dr_sigma)
        changed = True
    for observed_id, r in hb.range_obs.items():
        state.range_obs[(hb.origin, int(observed_id))] = (float(r), current_tick)
    if measured_range_to_sender is not None and my_drone_id is not None:
        state.range_obs[(my_drone_id, hb.origin)] = (
            float(measured_range_to_sender), current_tick
        )
    return changed


def ingest_command(
    state: ProtocolState, cmd: Command, current_tick: int
) -> bool:
    """Merge a Command. Returns True if this updated state."""
    if not verify_origin(cmd.origin, cmd.origin):
        return False
    if state.latest_command is None:
        state.latest_command = cmd
        state.latest_command_tick = current_tick
        return True
    cur = state.latest_command
    if (cmd.leader_priority, cmd.epoch) > (cur.leader_priority, cur.epoch):
        state.latest_command = cmd
        state.latest_command_tick = current_tick
        return True
    return False


def infer_leader(
    state: ProtocolState, current_tick: int, freshness: int = DEFAULT_FRESHNESS_TICKS
) -> int:
    """Return the drone_id currently believed to be the leader.

    Leader = highest (priority, epoch) among fresh known_priorities. Tie
    break: highest drone_id. Falls back to self if no peer is fresh.
    """
    best_origin = state.drone_id
    best = (state.my_priority, state.my_epoch, state.drone_id)
    for origin, (pri, epoch, heard) in state.known_priorities.items():
        if current_tick - heard > freshness:
            continue
        key = (pri, epoch, origin)
        if key > best:
            best = key
            best_origin = origin
    return best_origin


def fresh_known_drones(
    state: ProtocolState, current_tick: int, freshness: int = DEFAULT_FRESHNESS_TICKS
) -> list[dict]:
    """Return the drone set this drone currently considers alive+known.
    Reads from `known_positions` for back-compat with code that hasn't
    migrated to consensus_position lookups."""
    drones = []
    for origin, (pos, epoch, heard) in state.known_positions.items():
        if origin == state.drone_id:
            continue
        if current_tick - heard > freshness:
            continue
        drones.append({"id": origin, "pos": pos.copy()})
    return drones


def compute_consensus_positions(
    state: ProtocolState,
    current_tick: int,
    freshness: int = DEFAULT_FRESHNESS_TICKS,
    huber_scale_m: float = 2.0,
    n_irls_iters: int = 3,
    default_range_sigma_m: float = 0.5,
) -> dict[int, np.ndarray]:
    """DR-anchored Huber least-squares consensus position fit.
    Validated standalone in test_gpa_consensus.py Scenario G.

    Minimizes:
        Σ_i (1/σ_DR_i²) ρ(||x_i - DR_i||) +
        Σ_(i,j) (1/σ_R_ij²) ρ(||x_i - x_j|| - r_ij)

    where ρ is the Huber loss (handled internally by least_squares). The DR
    anchors break the rigid-body / reflection ambiguity that pure-range
    fits suffer; the Huber-loss IRLS inside least_squares down-weights
    outliers (bad sensors) automatically.

    Hot-path optimization (2026-05-23): vectorized residuals + analytic
    Jacobian collapse a 333 ms/call (n=20, 292 edges, finite-diff
    Jacobian, 4 outer IRLS) implementation down to a single least_squares
    call with O(1) Python overhead. n_irls_iters default dropped to 1
    because Huber loss does the reweighting internally.

    Returns: drone_id -> (3,) consensus position.
    """
    from scipy.optimize import least_squares

    fresh_self_ids = [
        origin for origin, (_, _, heard) in state.self_estimates.items()
        if current_tick - heard <= freshness
    ]
    drone_ids = sorted(fresh_self_ids)
    if not drone_ids:
        return {}
    id_to_idx = {did: k for k, did in enumerate(drone_ids)}
    n = len(drone_ids)
    DR = np.array([state.self_estimates[did][0] for did in drone_ids], dtype=np.float64)
    dr_sigmas = getattr(state, "dr_sigmas", {})
    dr_sig = np.array([dr_sigmas.get(did, 1.0) for did in drone_ids], dtype=np.float64)
    aw_sqrt = 1.0 / dr_sig  # sqrt(1/sigma^2) = 1/sigma

    # Edge arrays: parallel vectors of (i_idx, j_idx, distance, weight_sqrt).
    e_i, e_j, e_d = [], [], []
    for (obs, observed), (r, heard) in state.range_obs.items():
        if current_tick - heard > freshness:
            continue
        if obs not in id_to_idx or observed not in id_to_idx:
            continue
        e_i.append(id_to_idx[obs]); e_j.append(id_to_idx[observed]); e_d.append(float(r))
    edges_i = np.array(e_i, dtype=np.int64)
    edges_j = np.array(e_j, dtype=np.int64)
    edges_d = np.array(e_d, dtype=np.float64)
    n_edges = edges_i.size
    ew_sqrt = np.full(n_edges, 1.0 / default_range_sigma_m, dtype=np.float64)

    M_anchor = 3 * n  # 3 residual entries per anchor
    M_total = M_anchor + n_edges
    # Mutable weights that the outer IRLS reweights between solves (residual-
    # based: w = 1 / (sigma^2 + r^2) — Geman-McClure-like, more aggressive than
    # plain Huber). aw_sqrt_rep and ew_sqrt are bound into the closures and
    # mutated in-place between iterations.
    aw_sqrt_rep = np.repeat(aw_sqrt, 3).copy()  # (3n,)
    ew_sqrt = ew_sqrt.copy()

    def residuals(x_flat: np.ndarray) -> np.ndarray:
        X = x_flat.reshape((n, 3))
        anchor_res = ((X - DR).ravel()) * aw_sqrt_rep
        if n_edges:
            diffs = X[edges_i] - X[edges_j]              # (E, 3)
            dists = np.linalg.norm(diffs, axis=1)        # (E,)
            edge_res = (dists - edges_d) * ew_sqrt
            return np.concatenate([anchor_res, edge_res])
        return anchor_res

    def jacobian(x_flat: np.ndarray) -> np.ndarray:
        X = x_flat.reshape((n, 3))
        J = np.zeros((M_total, M_anchor), dtype=np.float64)
        idx = np.arange(M_anchor)
        J[idx, idx] = aw_sqrt_rep
        if n_edges:
            diffs = X[edges_i] - X[edges_j]
            dists = np.linalg.norm(diffs, axis=1)
            dists = np.maximum(dists, 1e-12)
            units = (diffs / dists[:, None]) * ew_sqrt[:, None]
            rows = np.arange(n_edges) + M_anchor
            for k in range(3):
                J[rows, edges_i * 3 + k] = units[:, k]
                J[rows, edges_j * 3 + k] = -units[:, k]
        return J

    x = DR.ravel().copy()
    edge_sig = default_range_sigma_m
    for _ in range(max(1, n_irls_iters)):
        try:
            result = least_squares(
                residuals, x0=x, jac=jacobian, method="trf",
                loss="huber", f_scale=huber_scale_m, max_nfev=30,
            )
            x = result.x
        except Exception:
            break
        # Residual-based reweight: w_i = 1 / (sigma_i^2 + r_i^2), sqrt weights
        # are what residuals() multiplies in. Done in-place so the closures
        # see the new values on the next iteration.
        X = x.reshape((n, 3))
        anchor_raw = np.linalg.norm(X - DR, axis=1)                    # (n,)
        new_aw_sqrt = 1.0 / np.sqrt(dr_sig ** 2 + anchor_raw ** 2)     # (n,)
        aw_sqrt_rep[:] = np.repeat(new_aw_sqrt, 3)
        if n_edges:
            diffs = X[edges_i] - X[edges_j]
            edge_raw = np.linalg.norm(diffs, axis=1) - edges_d
            ew_sqrt[:] = 1.0 / np.sqrt(edge_sig ** 2 + edge_raw ** 2)

    positions = x.reshape((n, 3))
    return {did: positions[id_to_idx[did]] for did in drone_ids}


# ---------------------------------------------------------------------------
# Falsifiability tests for the protocol primitives.
# ---------------------------------------------------------------------------


def _tests() -> int:
    failed = 0

    # T1: priority-vote merge picks max.
    s = ProtocolState(drone_id=0, my_priority=10, my_epoch=0)
    v1 = PriorityVote(origin=1, priority=5, epoch=0, sig_stub=make_sig(1, 0, "P"))
    v2 = PriorityVote(origin=1, priority=5, epoch=1, sig_stub=make_sig(1, 1, "P"))
    v3 = PriorityVote(origin=1, priority=4, epoch=99, sig_stub=make_sig(1, 99, "P"))
    ingest_priority_vote(s, v1, current_tick=0)
    ingest_priority_vote(s, v2, current_tick=1)
    ingest_priority_vote(s, v3, current_tick=2)  # priority-lower should NOT override
    p, e, _ = s.known_priorities[1]
    if (p, e) != (5, 1):
        print(f"FAIL T1: priority vote merge wrong: got ({p}, {e})")
        failed += 1

    # T2: leader inference picks highest fresh priority.
    s = ProtocolState(drone_id=0, my_priority=2, my_epoch=0)
    ingest_priority_vote(s, PriorityVote(1, 5, 0, b""), current_tick=0)
    ingest_priority_vote(s, PriorityVote(2, 7, 0, b""), current_tick=0)
    if infer_leader(s, current_tick=0) != 2:
        print(f"FAIL T2a: leader should be 2, got {infer_leader(s, 0)}")
        failed += 1
    # T2b: stale entries drop out of consideration.
    later = infer_leader(s, current_tick=100, freshness=10)
    if later != 0:
        print(f"FAIL T2b: stale peers should fall away, expected self(0), got {later}")
        failed += 1

    # T3: heartbeat keeps latest dr_position by epoch.
    s = ProtocolState(drone_id=0, my_priority=1, my_epoch=0)
    h1 = Heartbeat(origin=2, dr_position=np.array([1.0, 0, 0]), dr_sigma=1.0, range_obs={}, epoch=0, sig_stub=b"")
    h2 = Heartbeat(origin=2, dr_position=np.array([2.0, 0, 0]), dr_sigma=1.0, range_obs={}, epoch=1, sig_stub=b"")
    h3 = Heartbeat(origin=2, dr_position=np.array([99.0, 0, 0]), dr_sigma=1.0, range_obs={}, epoch=0, sig_stub=b"")  # stale
    ingest_heartbeat(s, h1, 0)
    ingest_heartbeat(s, h2, 1)
    ingest_heartbeat(s, h3, 2)
    pos, ep, _ = s.self_estimates[2]
    if not (abs(pos[0] - 2.0) < 1e-9 and ep == 1):
        print(f"FAIL T3: dr_position merge wrong, got pos={pos[0]} ep={ep}")
        failed += 1

    # T3b: bad-self-estimate gets corrected by neighbor range measurements.
    # Drone 5 reports self_estimate at (50, 0, 0) but its k neighbors all
    # measure it via ToF at ranges consistent with TRUE position (10, 0, 0).
    s = ProtocolState(drone_id=99, my_priority=1, my_epoch=0)
    # Anchors at known positions:
    for nid, pos in [(0, [0, 0, 0]), (1, [20, 0, 0]), (2, [10, 10, 0]), (3, [10, 0, 10])]:
        s.self_estimates[nid] = (np.array(pos, dtype=float), 0, 0)
    # Bad-sensor drone 5 SAYS it's at (50, 0, 0), but is actually at (10, 0, 0):
    s.self_estimates[5] = (np.array([50.0, 0, 0]), 0, 0)
    true_p5 = np.array([10.0, 0, 0])
    for nid, npos in [(0, [0,0,0]), (1, [20,0,0]), (2, [10,10,0]), (3, [10,0,10])]:
        r = float(np.linalg.norm(true_p5 - np.array(npos)))
        s.range_obs[(nid, 5)] = (r, 0)
    consensus = compute_consensus_positions(s, current_tick=0)
    p5_consensus = consensus[5]
    err_from_truth = float(np.linalg.norm(p5_consensus - true_p5))
    err_from_lie = float(np.linalg.norm(p5_consensus - np.array([50.0, 0, 0])))
    if err_from_truth > 1.0:
        print(f"FAIL T3b: consensus position for bad-sensor drone {err_from_truth:.3f}m from truth (>1m)")
        failed += 1
    if err_from_lie < 30.0:
        print(f"FAIL T3b: consensus didn't reject the outlier (only {err_from_lie:.3f}m from lie)")
        failed += 1

    # T4: command takes highest (priority, epoch).
    s = ProtocolState(drone_id=0, my_priority=1, my_epoch=0)
    c1 = Command(origin=5, leader_priority=10, epoch=0, payload="A", sig_stub=b"")
    c2 = Command(origin=6, leader_priority=10, epoch=1, payload="B", sig_stub=b"")
    c3 = Command(origin=7, leader_priority=20, epoch=0, payload="C", sig_stub=b"")
    c4 = Command(origin=8, leader_priority=5, epoch=999, payload="D", sig_stub=b"")  # lower priority despite epoch
    ingest_command(s, c1, 0)
    ingest_command(s, c2, 1)
    ingest_command(s, c3, 2)
    ingest_command(s, c4, 3)
    if state := s.latest_command:
        if state.payload != "C":
            print(f"FAIL T4: command merge wrong, got {state.payload}")
            failed += 1
    else:
        print("FAIL T4: latest_command is None")
        failed += 1

    return failed


if __name__ == "__main__":
    n_failed = _tests()
    print("protocol: all tests passed" if n_failed == 0 else f"protocol: {n_failed} tests failed")
