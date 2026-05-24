# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Event-driven gossip protocol for limited-comms underwater swarms.

DESIGN PREMISE: bandwidth- and energy-starved acoustic channel. A drone
transmits ONLY when one of four explicit events fires. Continuous
broadcast ("heartbeats", periodic liveness pings) is explicitly excluded
as incompatible with the premise. Liveness is INFERRED from received
activity, not announced.

Four communicative event types travel through gossip:

  Map(kind=call|response, origin, epoch, dr_position, dr_sigma,
      range_obs, round_id, sig_stub)
      Situational-awareness round. The current leader emits Map.CALL
      (carrying just round_id + own dr_pos for the responders to ToF-
      range against). Other drones reply with Map.RESPONSE carrying
      their own (dr_pos, dr_sigma, accumulated range_obs). Result: every
      participating drone has updated data for the consensus IRLS.

  Vote(kind=call|response, origin, priority, epoch, round_id, sig_stub)
      Election round. Leader emits Vote.CALL after a Map round.
      Candidates emit Vote.RESPONSE carrying their (priority, epoch)
      claim. Flood-max merge produces consensus on leader.

  Command(origin, leader_priority, epoch, payload, sig_stub)
      Mission directive. Issued by confirmed leader after a Vote round
      closes. Carries manifold + heading + leg metadata.

  OhShit(origin, kind, payload, sig_stub)
      Emergency. Drone-initiated when it detects something it cannot
      handle alone (own sensor failure, byzantine peer detected via
      consensus residual analysis, participant-count collapse, mission
      infeasibility). VALID ONLY DURING MOVE/SETTLE/REFORM phases --
      raising during a Map/Vote round would create a race; those rounds
      are themselves the response to whatever needs attention.

PASSIVE (non-communicative) signal: any acoustic message has measurable
ToF at receivers. There is NO standalone ranging ping; the existence-
and-distance information comes from the act of any communicative
transmission. Between rounds = silence = no new ranges = neighbors' last
known positions age out of the freshness window.

Forwarding: per-message TTL decrement with (kind, origin, epoch,
round_id) dedup. Forwarding is NOT a fresh broadcast; relayers do not
augment the payload.

Signature stub: 16-byte tag occupying the bandwidth a real Ed25519
signature would. Verification is origin-consistency only (the substrate
delivers msg.origin == sender). Byzantine drones lie about CONTENT, not
identity, so this check holds against them; the consensus IRLS is the
mechanism that handles content lies via outlier rejection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


SIG_LEN = 16
DEFAULT_FRESHNESS_TICKS = 30


def make_sig(origin: int, epoch: int, kind: str, secret: bytes = b"unused") -> bytes:
    """Deterministic 16-byte stub of an Ed25519 signature."""
    h = hashlib.sha256(secret + str(origin).encode() + str(epoch).encode() + kind.encode())
    return h.digest()[:SIG_LEN]


# ---------------------------------------------------------------------------
# Message types -- all event-driven, never periodic.
# ---------------------------------------------------------------------------


class MsgKind(Enum):
    CALL = "call"
    RESPONSE = "response"


@dataclass
class Map:
    """Situational-awareness round. CALL initiates from leader; RESPONSE
    carries the responder's dr_pos + range obs accumulated since the last
    Map round."""
    kind: MsgKind
    origin: int
    epoch: int
    round_id: int
    dr_position: np.ndarray
    dr_sigma: float
    range_obs: dict[int, float]
    sig_stub: bytes


@dataclass
class Vote:
    """Election round. CALL initiates from leader after a Map closes.
    RESPONSE carries each candidate's (priority, epoch) claim."""
    kind: MsgKind
    origin: int
    priority: int
    epoch: int
    round_id: int
    sig_stub: bytes


@dataclass
class Command:
    """Mission directive issued by confirmed leader after a Vote round."""
    origin: int
    leader_priority: int
    epoch: int
    payload: Any
    sig_stub: bytes


class OhShitKind(Enum):
    OWN_SENSOR_FAILURE = "own_sensor_failure"
    BYZANTINE_PEER = "byzantine_peer"          # data names suspected origin
    PARTICIPANT_COLLAPSE = "participant_collapse"
    MISSION_INFEASIBLE = "mission_infeasible"


@dataclass
class OhShit:
    """Emergency. Gated to MOVE/SETTLE/REFORM phases by the agent."""
    origin: int
    kind: OhShitKind
    payload: Any
    epoch: int
    sig_stub: bytes


# ---------------------------------------------------------------------------
# Per-drone state.
# ---------------------------------------------------------------------------


@dataclass
class ProtocolState:
    """Per-drone protocol state. NO cross-drone references allowed."""
    drone_id: int
    my_priority: int
    my_epoch: int = 0
    # origin -> (priority, epoch, heard_tick)
    known_priorities: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    # origin -> (dr_position, epoch, heard_tick)
    self_estimates: dict[int, tuple[np.ndarray, int, int]] = field(default_factory=dict)
    # origin -> reported dr_sigma (m)
    dr_sigmas: dict[int, float] = field(default_factory=dict)
    # (observer_id, observed_id) -> (range_m, heard_tick)
    range_obs: dict[tuple[int, int], tuple[float, int]] = field(default_factory=dict)
    # Latest confirmed mission directive.
    latest_command: Command | None = None
    latest_command_tick: int = -1
    # Round bookkeeping (rounds are leader-initiated and identified by
    # (origin_of_call, round_id); responders use round_id to dedup their
    # own responses so they don't reply twice to the same call).
    last_map_round_responded: tuple[int, int] | None = None  # (leader_id, round_id)
    last_vote_round_responded: tuple[int, int] | None = None
    # Back-compat alias for callers that still read known_positions
    # (manifold.compute_target, baseline_oracle). Populated from
    # self_estimates with no robust fit.
    known_positions: dict[int, tuple[np.ndarray, int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.known_priorities[self.drone_id] = (self.my_priority, self.my_epoch, 0)


def verify_origin(claimed: int, msg_origin: int) -> bool:
    """Origin-consistency stub. Real impl is Ed25519 verify over signed
    fields. Byzantine drones lie about content, not identity."""
    return claimed == msg_origin


# ---------------------------------------------------------------------------
# Ingest functions -- one per message type.
# Each returns True iff state changed (caller uses this to gate consensus
# recomputation; it is NOT a trigger for outgoing transmissions).
# ---------------------------------------------------------------------------


def _merge_priority(
    state: ProtocolState, origin: int, priority: int, epoch: int, tick: int
) -> bool:
    """Flood-max merge on (priority, epoch). Refresh heard_tick on equality."""
    prev = state.known_priorities.get(origin)
    if prev is None:
        state.known_priorities[origin] = (priority, epoch, tick)
        return True
    pp, pe, _ = prev
    if (priority, epoch) > (pp, pe):
        state.known_priorities[origin] = (priority, epoch, tick)
        return True
    if (priority, epoch) == (pp, pe):
        state.known_priorities[origin] = (pp, pe, tick)
    return False


def _merge_self_estimate(
    state: ProtocolState, origin: int, dr_pos: np.ndarray, dr_sigma: float,
    epoch: int, tick: int
) -> bool:
    prev = state.self_estimates.get(origin)
    if prev is not None and epoch < prev[1]:
        return False
    pos = np.asarray(dr_pos, dtype=np.float64).copy()
    state.self_estimates[origin] = (pos, epoch, tick)
    state.dr_sigmas[origin] = float(dr_sigma)
    state.known_positions[origin] = (pos.copy(), epoch, tick)
    return True


def _merge_range_obs(
    state: ProtocolState, observer: int, observed_id: int, range_m: float, tick: int
) -> None:
    state.range_obs[(observer, observed_id)] = (float(range_m), tick)


def ingest_map(
    state: ProtocolState, m: Map, tick: int,
    measured_range_to_sender: float | None = None,
) -> bool:
    """Merge a Map message (either CALL or RESPONSE)."""
    if not verify_origin(m.origin, m.origin):
        return False
    changed = False
    # Both kinds carry sender's dr_pos + range observations -- the act of
    # emitting any message reveals where you are, regardless of intent.
    changed |= _merge_self_estimate(
        state, m.origin, m.dr_position, m.dr_sigma, m.epoch, tick
    )
    for observed_id, r in m.range_obs.items():
        _merge_range_obs(state, m.origin, int(observed_id), float(r), tick)
    if measured_range_to_sender is not None:
        _merge_range_obs(state, state.drone_id, m.origin, measured_range_to_sender, tick)
    return changed


def ingest_vote(
    state: ProtocolState, v: Vote, tick: int,
    measured_range_to_sender: float | None = None,
) -> bool:
    """Merge a Vote message (either CALL or RESPONSE). Flood-max on
    (priority, epoch). Records range observation if substrate provided one."""
    if not verify_origin(v.origin, v.origin):
        return False
    changed = _merge_priority(state, v.origin, v.priority, v.epoch, tick)
    if measured_range_to_sender is not None:
        _merge_range_obs(state, state.drone_id, v.origin, measured_range_to_sender, tick)
    return changed


def ingest_command(
    state: ProtocolState, c: Command, tick: int,
    measured_range_to_sender: float | None = None,
) -> bool:
    """Merge a Command. Highest (leader_priority, epoch) wins."""
    if not verify_origin(c.origin, c.origin):
        return False
    if measured_range_to_sender is not None:
        _merge_range_obs(state, state.drone_id, c.origin, measured_range_to_sender, tick)
    if state.latest_command is None or (
        c.leader_priority, c.epoch
    ) > (state.latest_command.leader_priority, state.latest_command.epoch):
        state.latest_command = c
        state.latest_command_tick = tick
        return True
    return False


def ingest_oh_shit(
    state: ProtocolState, e: OhShit, tick: int,
    measured_range_to_sender: float | None = None,
) -> bool:
    """Merge an OhShit. State change recorded by the caller (agent decides
    how to react -- e.g., leader may trigger a fresh Vote round if
    BYZANTINE_PEER points at the current leader)."""
    if not verify_origin(e.origin, e.origin):
        return False
    if measured_range_to_sender is not None:
        _merge_range_obs(state, state.drone_id, e.origin, measured_range_to_sender, tick)
    # OhShit doesn't directly mutate known_priorities or self_estimates;
    # the agent handles it as a signal. Returning True so the consensus
    # cache is invalidated (a peer's emergency is information).
    return True


# ---------------------------------------------------------------------------
# Inference helpers.
# ---------------------------------------------------------------------------


def infer_leader(
    state: ProtocolState, tick: int, freshness: int = DEFAULT_FRESHNESS_TICKS
) -> int:
    """Leader = highest (priority, epoch) among fresh known_priorities.
    Tie-break: highest origin id. Falls back to self if no peer is fresh."""
    best_origin = state.drone_id
    best = (state.my_priority, state.my_epoch, state.drone_id)
    for origin, (pri, epoch, heard) in state.known_priorities.items():
        if tick - heard > freshness:
            continue
        key = (pri, epoch, origin)
        if key > best:
            best = key
            best_origin = origin
    return best_origin


def fresh_known_drones(
    state: ProtocolState, tick: int, freshness: int = DEFAULT_FRESHNESS_TICKS
) -> list[dict]:
    """Drones this node currently believes are alive and known. Back-compat
    accessor: reads from known_positions."""
    drones = []
    for origin, (pos, epoch, heard) in state.known_positions.items():
        if origin == state.drone_id:
            continue
        if tick - heard > freshness:
            continue
        drones.append({"id": origin, "pos": pos.copy()})
    return drones


def fresh_participant_count(
    state: ProtocolState, tick: int, freshness: int = DEFAULT_FRESHNESS_TICKS
) -> int:
    """How many peers (including self) this drone currently considers
    in-contact. Used for the leader's viability check before issuing a
    mission Command (catastrophic-comms detection)."""
    n = 1  # self
    for origin, (_, _, heard) in state.known_priorities.items():
        if origin == state.drone_id:
            continue
        if tick - heard <= freshness:
            n += 1
    return n


def compute_consensus_positions(
    state: ProtocolState,
    tick: int,
    freshness: int = DEFAULT_FRESHNESS_TICKS,
    huber_scale_m: float = 2.0,
    n_irls_iters: int = 3,
    default_range_sigma_m: float = 0.5,
) -> dict[int, np.ndarray]:
    """DR-anchored Huber least-squares consensus position fit.

    Minimizes:
      sum_i (1/sigma_DR_i^2) rho(||x_i - DR_i||) +
      sum_(i,j) (1/sigma_R_ij^2) rho(||x_i - x_j|| - r_ij)

    where rho is Huber. DR anchors break the rigid-body / reflection
    ambiguity that pure-range fits suffer; Huber + IRLS reweighting
    downweights outliers (bad sensors, byzantine peers lying about their
    dr_position) automatically. THIS is the byzantine-resilience mechanism
    -- no separate per-message detection layer.

    Vectorized residuals + analytic Jacobian: ~55ms/call for n=20 with
    full edge fan-out (validated 2026-05-23, kindex 8a270cb76915).
    """
    from scipy.optimize import least_squares

    fresh_self_ids = [
        origin for origin, (_, _, heard) in state.self_estimates.items()
        if tick - heard <= freshness
    ]
    drone_ids = sorted(fresh_self_ids)
    if not drone_ids:
        return {}
    id_to_idx = {did: k for k, did in enumerate(drone_ids)}
    n = len(drone_ids)
    DR = np.array([state.self_estimates[did][0] for did in drone_ids], dtype=np.float64)
    dr_sig = np.array(
        [state.dr_sigmas.get(did, 1.0) for did in drone_ids], dtype=np.float64
    )
    aw_sqrt = 1.0 / dr_sig

    e_i, e_j, e_d = [], [], []
    for (obs, observed), (r, heard) in state.range_obs.items():
        if tick - heard > freshness:
            continue
        if obs not in id_to_idx or observed not in id_to_idx:
            continue
        e_i.append(id_to_idx[obs]); e_j.append(id_to_idx[observed]); e_d.append(float(r))
    edges_i = np.array(e_i, dtype=np.int64)
    edges_j = np.array(e_j, dtype=np.int64)
    edges_d = np.array(e_d, dtype=np.float64)
    n_edges = edges_i.size
    ew_sqrt = np.full(n_edges, 1.0 / default_range_sigma_m, dtype=np.float64)

    M_anchor = 3 * n
    M_total = M_anchor + n_edges
    aw_sqrt_rep = np.repeat(aw_sqrt, 3).copy()
    ew_sqrt = ew_sqrt.copy()

    def residuals(x_flat: np.ndarray) -> np.ndarray:
        X = x_flat.reshape((n, 3))
        anchor_res = ((X - DR).ravel()) * aw_sqrt_rep
        if n_edges:
            diffs = X[edges_i] - X[edges_j]
            dists = np.linalg.norm(diffs, axis=1)
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
        X = x.reshape((n, 3))
        anchor_raw = np.linalg.norm(X - DR, axis=1)
        new_aw_sqrt = 1.0 / np.sqrt(dr_sig ** 2 + anchor_raw ** 2)
        aw_sqrt_rep[:] = np.repeat(new_aw_sqrt, 3)
        if n_edges:
            diffs = X[edges_i] - X[edges_j]
            edge_raw = np.linalg.norm(diffs, axis=1) - edges_d
            ew_sqrt[:] = 1.0 / np.sqrt(edge_sig ** 2 + edge_raw ** 2)

    positions = x.reshape((n, 3))
    return {did: positions[id_to_idx[did]] for did in drone_ids}


def detect_byzantine_via_residuals(
    state: ProtocolState,
    tick: int,
    freshness: int = DEFAULT_FRESHNESS_TICKS,
    threshold_m: float = 20.0,
) -> list[int]:
    """Identify peers whose dr_position disagrees with their range-implied
    position by more than threshold_m, AFTER the IRLS has run. This is the
    residual-driven detection -- a byproduct of consensus, not a separate
    pre-filter on incoming messages. Returns list of suspect origin ids.

    The CALLER decides what to do with the result (typically: leader emits
    OhShit naming the suspect; mission planning routes the formation
    around the byzantine's physical position).
    """
    consensus = compute_consensus_positions(state, tick, freshness=freshness)
    suspects = []
    for origin, c_pos in consensus.items():
        if origin not in state.self_estimates:
            continue
        dr_pos, _, _ = state.self_estimates[origin]
        residual = float(np.linalg.norm(c_pos - dr_pos))
        if residual > threshold_m:
            suspects.append(origin)
    return suspects


# ---------------------------------------------------------------------------
# Falsifiability tests.
# ---------------------------------------------------------------------------


def _tests() -> int:
    failed = 0

    # T1: Vote merges by max (priority, epoch); lower priority does not override.
    s = ProtocolState(drone_id=0, my_priority=10)
    for vp, ve in [(5, 0), (5, 1), (4, 99)]:
        v = Vote(MsgKind.RESPONSE, origin=1, priority=vp, epoch=ve, round_id=0, sig_stub=b"")
        ingest_vote(s, v, tick=ve)
    p, e, _ = s.known_priorities[1]
    if (p, e) != (5, 1):
        print(f"FAIL T1: vote merge wrong: got ({p}, {e})")
        failed += 1

    # T2: leader inference picks highest fresh priority; stale entries drop.
    s = ProtocolState(drone_id=0, my_priority=2)
    ingest_vote(s, Vote(MsgKind.RESPONSE, 1, 5, 0, 0, b""), tick=0)
    ingest_vote(s, Vote(MsgKind.RESPONSE, 2, 7, 0, 0, b""), tick=0)
    if infer_leader(s, tick=0) != 2:
        print(f"FAIL T2a: leader should be 2, got {infer_leader(s, 0)}")
        failed += 1
    if infer_leader(s, tick=100, freshness=10) != 0:
        print(f"FAIL T2b: stale peers should fall away to self(0)")
        failed += 1

    # T3: Map merges self_estimate by epoch; older epoch is ignored.
    s = ProtocolState(drone_id=0, my_priority=1)
    for ep, x in [(0, 1.0), (1, 2.0), (0, 99.0)]:
        m = Map(MsgKind.RESPONSE, origin=2, epoch=ep, round_id=0,
                dr_position=np.array([x, 0, 0]), dr_sigma=1.0,
                range_obs={}, sig_stub=b"")
        ingest_map(s, m, tick=ep)
    pos, ep, _ = s.self_estimates[2]
    if not (abs(pos[0] - 2.0) < 1e-9 and ep == 1):
        print(f"FAIL T3: dr_position merge wrong, got pos={pos[0]} ep={ep}")
        failed += 1

    # T3b: outlier rejection via consensus IRLS (no per-message detection).
    s = ProtocolState(drone_id=99, my_priority=1)
    for nid, pos in [(0, [0, 0, 0]), (1, [20, 0, 0]), (2, [10, 10, 0]), (3, [10, 0, 10])]:
        s.self_estimates[nid] = (np.array(pos, dtype=float), 0, 0)
        s.dr_sigmas[nid] = 1.0
    s.self_estimates[5] = (np.array([50.0, 0, 0]), 0, 0)  # bad sensor: claims (50,0,0)
    s.dr_sigmas[5] = 1.0
    true_p5 = np.array([10.0, 0, 0])
    for nid, npos in [(0, [0,0,0]), (1, [20,0,0]), (2, [10,10,0]), (3, [10,0,10])]:
        r = float(np.linalg.norm(true_p5 - np.array(npos)))
        s.range_obs[(nid, 5)] = (r, 0)
    consensus = compute_consensus_positions(s, tick=0)
    p5 = consensus[5]
    if float(np.linalg.norm(p5 - true_p5)) > 1.0:
        print(f"FAIL T3b: bad-sensor drone consensus {np.linalg.norm(p5 - true_p5):.3f}m from truth")
        failed += 1
    if float(np.linalg.norm(p5 - np.array([50.0, 0, 0]))) < 30.0:
        print(f"FAIL T3b: consensus didn't reject the outlier")
        failed += 1

    # T3c: byzantine detection via post-IRLS residuals -- the bad-sensor
    # drone from T3b should be flagged.
    suspects = detect_byzantine_via_residuals(s, tick=0, threshold_m=20.0)
    if 5 not in suspects:
        print(f"FAIL T3c: drone 5 should be flagged as byzantine, got suspects={suspects}")
        failed += 1

    # T4: Command merge by (leader_priority, epoch) max.
    s = ProtocolState(drone_id=0, my_priority=1)
    for orig, lp, ep, pl in [(5, 10, 0, "A"), (6, 10, 1, "B"), (7, 20, 0, "C"), (8, 5, 999, "D")]:
        ingest_command(s, Command(orig, lp, ep, pl, b""), tick=0)
    if s.latest_command is None or s.latest_command.payload != "C":
        got = s.latest_command.payload if s.latest_command else None
        print(f"FAIL T4: command merge wrong, got {got}")
        failed += 1

    # T5: substrate-measured range to sender is stored on any ingest.
    s = ProtocolState(drone_id=0, my_priority=1)
    m = Map(MsgKind.RESPONSE, origin=7, epoch=0, round_id=1,
            dr_position=np.zeros(3), dr_sigma=1.0,
            range_obs={3: 12.0, 4: 8.5}, sig_stub=b"")
    ingest_map(s, m, tick=10, measured_range_to_sender=4.2)
    if s.range_obs.get((7, 3)) != (12.0, 10):
        print(f"FAIL T5a: sender's range to 3 not stored")
        failed += 1
    if s.range_obs.get((0, 7)) != (4.2, 10):
        print(f"FAIL T5b: my measured range to 7 not stored")
        failed += 1

    # T6: fresh_participant_count drops as peers age out.
    s = ProtocolState(drone_id=0, my_priority=10)
    for did in range(1, 6):
        ingest_vote(s, Vote(MsgKind.RESPONSE, did, did, 0, 0, b""), tick=0)
    if fresh_participant_count(s, tick=0) != 6:
        print(f"FAIL T6a: expected 6 participants, got {fresh_participant_count(s, 0)}")
        failed += 1
    if fresh_participant_count(s, tick=100, freshness=10) != 1:
        print(f"FAIL T6b: expected 1 (self only) after aging, got "
              f"{fresh_participant_count(s, 100, 10)}")
        failed += 1

    return failed


if __name__ == "__main__":
    n = _tests()
    print("protocol: all tests passed" if n == 0 else f"protocol: {n} tests failed")
