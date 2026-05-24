# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Single-message-type, event-driven gossip protocol for limited-comms.

DESIGN PREMISE: the underlying channel is bandwidth- and energy-starved
(acoustic, range-limited, with propagation delay). Continuous broadcast
("heartbeats", periodic vote pings) defeats the entire purpose of the
research and is explicitly excluded. A drone transmits IFF it has new
information worth propagating. Liveness is INFERRED from received
activity, not announced.

One message type travels through gossip:

  Update(origin, priority, epoch, dr_position, dr_sigma, range_obs,
         command, sig_stub)
      Carries everything the sender currently has worth sharing:
        - signed (priority, epoch) -- this IS the vote; flood-max merge
          on the receiving side
        - dr_position + dr_sigma   -- sender's self-estimate, soft anchor
                                       for the DR-anchored IRLS consensus
        - range_obs                -- ranges the sender has physically
                                       MEASURED since its last transmit
                                       (origin->neighbor); not claims
        - command                  -- attached if the sender is currently
                                       acting as leader and has a new
                                       directive; otherwise None

  Forwarding is per-message: receivers relay the original Update with
  TTL decrement and (origin, epoch) dedup. Forwarding is NOT a fresh
  broadcast; the relayer does not augment the payload.

Each drone maintains:
  - known_priorities[origin] = (priority, epoch, heard_tick)
  - self_estimates[origin]   = (dr_position, epoch, heard_tick)
  - dr_sigmas[origin]        = sender's reported dr_sigma
  - range_obs[(obs, observed)] = (range_m, heard_tick)
  - latest_command           = highest-(priority,epoch) Command in fresh
                               window, or None

Leader inference is local: highest (priority, epoch) among fresh
known_priorities; tie-break on origin id; falls back to self if no peer
is fresh.

Signature stub is NOT real crypto. The 16-byte tag exists to model the
bandwidth of an Ed25519 signature; verification is "origin == claimed
origin" (returned to us by LocalComms.Message). Byzantine drones still
pass this check because they lie about content, not identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


SIG_LEN = 16
DEFAULT_FRESHNESS_TICKS = 30  # how recent a heard message must be to count


def make_sig(origin: int, epoch: int, kind: str, secret: bytes = b"unused") -> bytes:
    """Stub 'signature' -- deterministic 16-byte tag. NOT secure. Used to
    fill the bandwidth budget for a real Ed25519 signature."""
    h = hashlib.sha256(secret + str(origin).encode() + str(epoch).encode() + kind.encode())
    return h.digest()[:SIG_LEN]


@dataclass
class Command:
    """Mission directive payload. Carried INSIDE an Update when the
    sender is acting as leader and has new content to issue."""
    origin: int
    leader_priority: int
    epoch: int
    payload: Any
    sig_stub: bytes


@dataclass
class Update:
    """The single event-driven gossip message. Transmitted by a drone
    ONLY when it has something genuinely new to say (see Agent.step for
    the trigger conditions). Forwarded by neighbors via TTL+dedup gossip.

    Fields:
      origin       : sender drone id
      priority     : sender's signed priority (this is the vote)
      epoch        : sender's update counter; monotonic per origin.
                     Receivers accept the (origin, epoch) tuple at most
                     once and merge by max (priority, epoch).
      dr_position  : sender's current dead-reckoning self-estimate (3,)
      dr_sigma     : sender's DR uncertainty (m), grows since last fix
      range_obs    : {observed_id: range_m} -- physical ToF measurements
                     the sender accumulated since its last transmit
      command      : optional Command if the sender is leader and has a
                     new directive; otherwise None
      sig_stub     : 16-byte stub of an Ed25519 signature over the
                     packed fields
    """
    origin: int
    priority: int
    epoch: int
    dr_position: np.ndarray
    dr_sigma: float
    range_obs: dict[int, float]
    command: Command | None
    sig_stub: bytes


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
    # origin -> dr_sigma (m), reported by sender; updated when newer epoch arrives
    dr_sigmas: dict[int, float] = field(default_factory=dict)
    # (observer_id, observed_id) -> (range_m, heard_tick)
    range_obs: dict[tuple[int, int], tuple[float, int]] = field(default_factory=dict)
    # Latest known mission directive by (priority, epoch).
    latest_command: Command | None = None
    latest_command_tick: int = -1
    # Back-compat alias used by callers that haven't migrated to consensus
    # position lookups. Populated from self_estimates with no robust fit.
    known_positions: dict[int, tuple[np.ndarray, int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.known_priorities[self.drone_id] = (self.my_priority, self.my_epoch, 0)


def verify_origin(claimed_origin: int, msg_origin: int) -> bool:
    """Origin-consistency stub. Real impl is Ed25519 verify over the
    Update's signed fields. Byzantine drones lie about content, not
    identity, so this check holds against them."""
    return claimed_origin == msg_origin


def ingest_update(
    state: ProtocolState,
    u: Update,
    current_tick: int,
    measured_range_to_sender: float | None = None,
) -> bool:
    """Merge an Update into local state.

    Side effects:
      - known_priorities: max-merge on (priority, epoch); refreshes
        heard_tick when the seen entry equals the stored one
      - self_estimates / dr_sigmas: replace when epoch >= stored epoch
      - range_obs: store sender's reported ranges (origin -> observed_id)
        AND, if the substrate told us the measured range to the sender,
        store our own (self -> origin) observation
      - latest_command: if Update carries a Command, fold it in by
        (leader_priority, epoch) max
      - known_positions: maintained as a back-compat alias to self_estimates

    Returns True iff state changed (caller uses this to decide whether
    consensus needs to be re-computed -- pure throttle, not a trigger
    for outgoing transmits).
    """
    if not verify_origin(u.origin, u.origin):
        return False
    changed = False

    prev_pri = state.known_priorities.get(u.origin)
    if prev_pri is None:
        state.known_priorities[u.origin] = (u.priority, u.epoch, current_tick)
        changed = True
    else:
        pp, pe, _ = prev_pri
        if (u.priority, u.epoch) > (pp, pe):
            state.known_priorities[u.origin] = (u.priority, u.epoch, current_tick)
            changed = True
        elif (u.priority, u.epoch) == (pp, pe):
            # Refresh heard-tick so freshness window doesn't expire.
            state.known_priorities[u.origin] = (pp, pe, current_tick)

    prev_self = state.self_estimates.get(u.origin)
    if prev_self is None or u.epoch >= prev_self[1]:
        state.self_estimates[u.origin] = (
            np.asarray(u.dr_position, dtype=np.float64).copy(),
            u.epoch,
            current_tick,
        )
        state.dr_sigmas[u.origin] = float(u.dr_sigma)
        state.known_positions[u.origin] = (
            np.asarray(u.dr_position, dtype=np.float64).copy(),
            u.epoch,
            current_tick,
        )
        changed = True

    for observed_id, r in u.range_obs.items():
        state.range_obs[(u.origin, int(observed_id))] = (float(r), current_tick)
    if measured_range_to_sender is not None:
        state.range_obs[(state.drone_id, u.origin)] = (
            float(measured_range_to_sender), current_tick
        )

    if u.command is not None:
        if state.latest_command is None or (
            u.command.leader_priority, u.command.epoch
        ) > (state.latest_command.leader_priority, state.latest_command.epoch):
            state.latest_command = u.command
            state.latest_command_tick = current_tick
            changed = True

    return changed


def infer_leader(
    state: ProtocolState, current_tick: int, freshness: int = DEFAULT_FRESHNESS_TICKS
) -> int:
    """Leader = highest (priority, epoch) among fresh known_priorities.
    Tie-break: highest origin id. Falls back to self if no peer is fresh."""
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
    """Drones this node currently believes are alive and known. Reads
    from known_positions (back-compat) for callers that haven't migrated
    to consensus_position lookups."""
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

    Minimizes:
        sum_i (1/sigma_DR_i^2) rho(||x_i - DR_i||) +
        sum_(i,j) (1/sigma_R_ij^2) rho(||x_i - x_j|| - r_ij)

    where rho is Huber. DR anchors break the rigid-body / reflection
    ambiguity that pure-range fits suffer; Huber + IRLS reweighting
    downweights outliers (bad sensors) automatically.

    Hot-path optimization (2026-05-23): vectorized residuals + analytic
    Jacobian collapse a 333ms/call implementation to ~55ms/call.

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
    dr_sig = np.array(
        [state.dr_sigmas.get(did, 1.0) for did in drone_ids], dtype=np.float64
    )
    aw_sqrt = 1.0 / dr_sig

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


# ---------------------------------------------------------------------------
# Falsifiability tests.
# ---------------------------------------------------------------------------


def _tests() -> int:
    failed = 0

    # T1: update merges max (priority, epoch); lower priority does not override.
    s = ProtocolState(drone_id=0, my_priority=10, my_epoch=0)
    u1 = Update(origin=1, priority=5, epoch=0, dr_position=np.zeros(3),
                dr_sigma=1.0, range_obs={}, command=None, sig_stub=b"")
    u2 = Update(origin=1, priority=5, epoch=1, dr_position=np.zeros(3),
                dr_sigma=1.0, range_obs={}, command=None, sig_stub=b"")
    u3 = Update(origin=1, priority=4, epoch=99, dr_position=np.zeros(3),
                dr_sigma=1.0, range_obs={}, command=None, sig_stub=b"")
    ingest_update(s, u1, current_tick=0)
    ingest_update(s, u2, current_tick=1)
    ingest_update(s, u3, current_tick=2)
    p, e, _ = s.known_priorities[1]
    if (p, e) != (5, 1):
        print(f"FAIL T1: priority merge wrong: got ({p}, {e})")
        failed += 1

    # T2: leader inference picks highest fresh priority; stale entries drop out.
    s = ProtocolState(drone_id=0, my_priority=2, my_epoch=0)
    ingest_update(s, Update(1, 5, 0, np.zeros(3), 1.0, {}, None, b""), current_tick=0)
    ingest_update(s, Update(2, 7, 0, np.zeros(3), 1.0, {}, None, b""), current_tick=0)
    if infer_leader(s, current_tick=0) != 2:
        print(f"FAIL T2a: leader should be 2, got {infer_leader(s, 0)}")
        failed += 1
    if infer_leader(s, current_tick=100, freshness=10) != 0:
        print(f"FAIL T2b: stale peers should fall away to self(0)")
        failed += 1

    # T3: self-estimate keeps latest dr_position by epoch.
    s = ProtocolState(drone_id=0, my_priority=1, my_epoch=0)
    ingest_update(s, Update(2, 1, 0, np.array([1.0, 0, 0]), 1.0, {}, None, b""), 0)
    ingest_update(s, Update(2, 1, 1, np.array([2.0, 0, 0]), 1.0, {}, None, b""), 1)
    ingest_update(s, Update(2, 1, 0, np.array([99.0, 0, 0]), 1.0, {}, None, b""), 2)  # stale
    pos, ep, _ = s.self_estimates[2]
    if not (abs(pos[0] - 2.0) < 1e-9 and ep == 1):
        print(f"FAIL T3: dr_position merge wrong, got pos={pos[0]} ep={ep}")
        failed += 1

    # T3b: outlier detection -- a bad self-estimate gets corrected by neighbor
    # range observations through the consensus IRLS.
    s = ProtocolState(drone_id=99, my_priority=1, my_epoch=0)
    for nid, pos in [(0, [0, 0, 0]), (1, [20, 0, 0]), (2, [10, 10, 0]), (3, [10, 0, 10])]:
        s.self_estimates[nid] = (np.array(pos, dtype=float), 0, 0)
        s.dr_sigmas[nid] = 1.0
    s.self_estimates[5] = (np.array([50.0, 0, 0]), 0, 0)  # bad sensor: claims 50,0,0
    s.dr_sigmas[5] = 1.0
    true_p5 = np.array([10.0, 0, 0])
    for nid, npos in [(0, [0,0,0]), (1, [20,0,0]), (2, [10,10,0]), (3, [10,0,10])]:
        r = float(np.linalg.norm(true_p5 - np.array(npos)))
        s.range_obs[(nid, 5)] = (r, 0)
    consensus = compute_consensus_positions(s, current_tick=0)
    p5 = consensus[5]
    err_from_truth = float(np.linalg.norm(p5 - true_p5))
    err_from_lie = float(np.linalg.norm(p5 - np.array([50.0, 0, 0])))
    if err_from_truth > 1.0:
        print(f"FAIL T3b: consensus for bad-sensor drone {err_from_truth:.3f}m from truth")
        failed += 1
    if err_from_lie < 30.0:
        print(f"FAIL T3b: consensus didn't reject the outlier (only {err_from_lie:.3f}m from lie)")
        failed += 1

    # T4: command attached to an Update is folded in by (priority, epoch) max.
    s = ProtocolState(drone_id=0, my_priority=1, my_epoch=0)
    c1 = Command(origin=5, leader_priority=10, epoch=0, payload="A", sig_stub=b"")
    c2 = Command(origin=6, leader_priority=10, epoch=1, payload="B", sig_stub=b"")
    c3 = Command(origin=7, leader_priority=20, epoch=0, payload="C", sig_stub=b"")
    c4 = Command(origin=8, leader_priority=5, epoch=999, payload="D", sig_stub=b"")
    for ci in [c1, c2, c3, c4]:
        u = Update(ci.origin, ci.leader_priority, ci.epoch, np.zeros(3), 1.0,
                   {}, ci, b"")
        ingest_update(s, u, current_tick=0)
    if s.latest_command is None or s.latest_command.payload != "C":
        got = s.latest_command.payload if s.latest_command else None
        print(f"FAIL T4: command merge wrong, got {got}")
        failed += 1

    # T5: range_obs from the Update payload are stored under sender's origin.
    s = ProtocolState(drone_id=0, my_priority=1, my_epoch=0)
    u = Update(origin=7, priority=1, epoch=0, dr_position=np.zeros(3),
               dr_sigma=1.0, range_obs={3: 12.0, 4: 8.5}, command=None,
               sig_stub=b"")
    ingest_update(s, u, current_tick=10, measured_range_to_sender=4.2)
    if s.range_obs.get((7, 3)) != (12.0, 10):
        print(f"FAIL T5a: sender's range to 3 not stored, got {s.range_obs.get((7,3))}")
        failed += 1
    if s.range_obs.get((7, 4)) != (8.5, 10):
        print(f"FAIL T5b: sender's range to 4 not stored, got {s.range_obs.get((7,4))}")
        failed += 1
    if s.range_obs.get((0, 7)) != (4.2, 10):
        print(f"FAIL T5c: my measured range to 7 not stored, got {s.range_obs.get((0,7))}")
        failed += 1

    return failed


if __name__ == "__main__":
    n_failed = _tests()
    print("protocol: all tests passed" if n_failed == 0 else f"protocol: {n_failed} tests failed")
