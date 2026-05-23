# /// script
# dependencies = ["numpy<3"]
# ///
"""Per-drone agent: state + decision step.

The agent has its own state (position, velocity, lock flag, protocol
state, known drone set) and a single decision function `step()` that
takes its current inbox and emits its next action and outgoing
messages. The agent NEVER reads another drone's state directly — all
knowledge of other drones must arrive through `inbox` (which the
`LocalComms` substrate produced from past message exchanges).

This is the unit that simulator.py's DroneAgent thread became when we
removed the global Broadcast assumption.

The decision pipeline each tick:
  1. ingest inbox: PriorityVotes, Heartbeats, Commands.
  2. infer current leader from known_priorities (per protocol.py).
  3. if I am the leader: bump my command epoch, set a fresh Command.
  4. build my known-drone set (self + fresh known_positions).
  5. build a manifold tree from my known target (from latest_command).
  6. run compute_target against my own known set.
  7. compute velocity toward target (with PBD-style repulsion via
     known_positions to avoid collisions with peers I can see).
  8. emit outgoing PriorityVote + Heartbeat + (if leader) Command.

The bench layer is responsible for: physics integration (position +=
vel * dt), failure injection (kill, byzantine), and comms scheduling
(call comms.send for each emitted message).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from manifold import ManifoldNode, compute_target
from protocol import (
    Command,
    DEFAULT_FRESHNESS_TICKS,
    Heartbeat,
    PriorityVote,
    ProtocolState,
    fresh_known_drones,
    infer_leader,
    ingest_command,
    ingest_heartbeat,
    ingest_priority_vote,
    make_sig,
)


@dataclass
class AgentStepLog:
    """Per-step audit record for falsifiability post-mortems."""
    tick: int
    drone_id: int
    leader: int
    n_known: int
    target: np.ndarray
    is_primary_at_leaf: bool
    speed: float
    locked: bool


MAX_HOPS = 6  # multi-hop relay TTL (gossip diameter for our typical setups)


def _fibonacci_manifold(n_targets: int) -> np.ndarray:
    """Build a Fibonacci-sphere manifold with `n_targets` leaves. Matches
    the algorithm used in bench_distributed.py so that smaller leader-
    issued manifolds during reformation are geometrically similar to the
    original (just fewer points)."""
    pts = []
    phi = (1 + 5 ** 0.5) / 2
    R = 8.0 * (n_targets ** 0.5) / (20 ** 0.5)
    for i in range(n_targets):
        z = 1 - 2 * (i + 0.5) / n_targets
        r = (1 - z * z) ** 0.5
        theta = 2 * np.pi * i / phi
        pts.append([R * r * np.cos(theta), R * r * np.sin(theta), R * z + R * 1.5])
    return np.array(pts)


@dataclass
class Agent:
    drone_id: int
    priority: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    locked: bool = False
    lie_about_position: Any = None
    refuse_to_forward: bool = False
    spam_priority: int | None = None
    enable_byzantine_detection: bool = False
    byzantine_tolerance_m: float = 5.0
    # Reformation: when loss exceeds threshold, the leader issues a smaller
    # manifold and every drone re-runs compute_target ONCE against it. This
    # avoids the continuous-bisection demotion problem.
    enable_reformation: bool = False
    initial_n_drones: int = 0       # set by bench; baseline for threshold
    loss_threshold_pct: float = 0.15  # 15% loss triggers reform

    def __post_init__(self) -> None:
        eff_priority = self.spam_priority if self.spam_priority is not None else self.priority
        self.proto = ProtocolState(
            drone_id=self.drone_id,
            my_priority=eff_priority,
            my_epoch=0,
        )
        self.proto.known_positions[self.drone_id] = (self.position.copy(), 0, 0)
        self.command_epoch_counter = 0
        self.heartbeat_epoch_counter = 0
        self.priority_epoch_counter = 0
        self._tree: ManifoldNode | None = None
        self._tree_signature: int = -1
        self.seen_messages: set[tuple[str, int, int]] = set()
        self.byzantine_flags: dict[int, int] = {}
        # Cache for consensus_positions: only recompute when new heartbeats
        # arrived OR every K ticks. Avoids running per-tick IRLS in the
        # common case where the swarm is in steady state.
        self._consensus_cache: dict[int, np.ndarray] = {}
        self._consensus_last_tick: int = -1
        self._heartbeats_since_last_consensus: int = 0
        # Emit-on-change throttling: only re-broadcast Heartbeats/PriorityVotes
        # when there's new content, or every K ticks for liveness.
        self._last_hb_tick: int = -1
        self._last_hb_position: np.ndarray | None = None
        self._last_pv_tick: int = -1

    def _compute_my_proposal(self, current_tick: int) -> "Command | None":
        """Decentralized: I compute the directive I would propose, given my
        local state alone. Return a Command if I have something new to say
        that would beat my current `latest_command`; otherwise return None
        (don't emit, save bandwidth, let other drones' proposals dominate).
        """
        if not self.enable_reformation or self.initial_n_drones == 0:
            return None
        from protocol import DEFAULT_FRESHNESS_TICKS as FW
        fresh_count = 1 + sum(
            1 for origin, (_, _, heard) in self.proto.known_positions.items()
            if origin != self.drone_id and current_tick - heard <= FW
        )
        expected = int(self.initial_n_drones * (1 - self.loss_threshold_pct))
        if fresh_count >= expected:
            return None  # no reform needed, nothing new to say
        new_manifold = _fibonacci_manifold(max(4, fresh_count))

        current = self.proto.latest_command
        # My proposal's epoch: if I've already signed at this priority,
        # bump my own counter; otherwise start fresh.
        if current is not None and current.origin == self.drone_id:
            new_epoch = current.epoch + 1
        else:
            self.command_epoch_counter += 1
            new_epoch = self.command_epoch_counter

        # Would my proposal beat the current latest_command via flood-max?
        # Comparison key: (priority, epoch). If not strictly greater,
        # stay silent — my proposal would be dominated.
        if current is not None:
            if (self.proto.my_priority, new_epoch) <= (
                current.leader_priority, current.epoch
            ):
                return None
            cur_m = current.payload.get("manifold_targets")
            if (
                cur_m is not None
                and cur_m.shape == new_manifold.shape
                and np.allclose(cur_m, new_manifold)
            ):
                # No actual change in content; don't churn the epoch.
                return None

        return Command(
            origin=self.drone_id,
            leader_priority=self.proto.my_priority,
            epoch=new_epoch,
            payload={
                "manifold_targets": new_manifold,
                "heading": np.array([1.0, 0.0, 0.0]),
                "leg": 0,
            },
            sig_stub=make_sig(self.drone_id, new_epoch, "C"),
        )

    def _reported_position(self, current_tick: int) -> np.ndarray:
        if self.lie_about_position is not None:
            return np.asarray(self.lie_about_position(current_tick), dtype=np.float64)
        return self.position.copy()

    def step(
        self,
        current_tick: int,
        inbox: list[Any],
        physics_dt: float = 1.0,
        max_speed: float = 0.8,
        repulsion_radius_m: float = 3.5,
        approach_radius_m: float = 4.0,
    ) -> tuple[np.ndarray, list[Any], AgentStepLog]:
        """Returns (new_velocity, outgoing_messages, audit_log)."""
        # --- 1. Ingest inbox, dedup, run Byzantine detection, queue forwards.
        forwards: list[tuple[Any, int, int]] = []  # (payload, next_hop, origin)
        for msg in inbox:
            payload = msg.payload
            origin = msg.origin
            epoch = getattr(payload, "epoch", 0)
            kind = type(payload).__name__
            key = (kind, origin, epoch)
            if key in self.seen_messages:
                continue
            self.seen_messages.add(key)

            # Byzantine detection: cross-check claimed position against the
            # time-of-flight-implied distance. Direct (hop==1) Heartbeats
            # are verifiable; if claimed_pos disagrees with the substrate's
            # measured range, flag the origin and drop the message (do NOT
            # ingest, do NOT forward — this keeps lies from propagating
            # through relays).
            rejected_as_byzantine = False
            if (
                self.enable_byzantine_detection
                and isinstance(payload, Heartbeat)
                and msg.hop == 1
            ):
                claimed_pos = np.asarray(payload.dr_position, dtype=np.float64)
                claimed_dist = float(np.linalg.norm(claimed_pos - self.position))
                physical_dist = msg.range_at_send
                if abs(claimed_dist - physical_dist) > self.byzantine_tolerance_m:
                    self.byzantine_flags[origin] = self.byzantine_flags.get(origin, 0) + 1
                    if self.byzantine_flags[origin] >= 3:
                        rejected_as_byzantine = True
            if rejected_as_byzantine:
                continue  # don't ingest and don't forward

            if isinstance(payload, PriorityVote):
                ingest_priority_vote(self.proto, payload, current_tick)
            elif isinstance(payload, Heartbeat):
                measured = getattr(msg, "range_at_send", None)
                if ingest_heartbeat(
                    self.proto, payload, current_tick,
                    measured_range_to_sender=measured,
                    my_drone_id=self.drone_id,
                ):
                    self._heartbeats_since_last_consensus += 1
            elif isinstance(payload, Command):
                ingest_command(self.proto, payload, current_tick)

            if not self.refuse_to_forward and msg.hop < MAX_HOPS:
                forwards.append((payload, msg.hop + 1, origin))

        # --- 2. Refresh own position in known_positions.
        self.proto.known_positions[self.drone_id] = (
            self.position.copy(), self.heartbeat_epoch_counter, current_tick
        )

        # --- 3. Infer leader.
        leader_id = infer_leader(self.proto, current_tick)

        # --- 4. DECENTRALIZED proposal. Every drone, every round, computes
        # its own proposal for the directive. If the proposal beats its
        # current latest_command, the drone adopts the proposal locally
        # AND broadcasts it. If not (its priority is dominated, or there
        # is nothing new to say), the drone does NOT emit a Command this
        # tick. That's the "minimal messaging" half — Commands only fly
        # when someone has a new winning proposal.
        outgoing: list[Any] = []
        my_proposal = self._compute_my_proposal(current_tick)
        if my_proposal is not None:
            # Adopt locally and broadcast.
            self.proto.latest_command = my_proposal
            self.proto.latest_command_tick = current_tick
            self.locked = False
            self._tree = None
            self._tree_signature = -1
            if not self.refuse_to_forward:
                outgoing.append(my_proposal)

        # --- 5. Build known-drone set from DR-anchored IRLS consensus.
        # Throttled: recompute every REFRESH_TICKS ticks. With 30+ drones
        # all running consensus per tick, the IRLS optimization dominates
        # runtime; throttling to every 5 ticks gives ~5x speedup with
        # minimal staleness (drones move ≤0.8 m/tick so consensus drift
        # over 5 ticks is bounded ~4 m, comparable to the consensus's
        # own clean baseline).
        from protocol import compute_consensus_positions
        REFRESH_TICKS = 5
        if (
            self._consensus_last_tick < 0
            or (current_tick - self._consensus_last_tick) >= REFRESH_TICKS
        ):
            self._consensus_cache = compute_consensus_positions(
                self.proto, current_tick
            )
            self._consensus_last_tick = current_tick
            self._heartbeats_since_last_consensus = 0
        consensus = self._consensus_cache
        my_entry = {"id": self.drone_id, "pos": self.position.copy()}
        peers = [
            {"id": did, "pos": pos.copy()}
            for did, pos in consensus.items() if did != self.drone_id
        ]
        known = [my_entry] + peers

        # --- 6. Compute target using bench-supplied manifold or fall back to
        #        rallying near the known-set centroid.
        manifold_targets = None
        if self.proto.latest_command is not None:
            mt = self.proto.latest_command.payload.get("manifold_targets")
            if mt is not None:
                manifold_targets = mt
        if manifold_targets is not None:
            sig = hash(manifold_targets.tobytes())
            if sig != self._tree_signature:
                self._tree = ManifoldNode(manifold_targets)
                self._tree_signature = sig
            target, is_primary = compute_target(self.drone_id, known, self._tree)
        else:
            # No command yet: hold position; primary=False.
            target = self.position.copy()
            is_primary = False

        # --- 7. Steering: simple attractor + repulsion from known peers.
        diff = target - self.position
        dist = float(np.linalg.norm(diff))
        is_final = dist < approach_radius_m

        attr = (diff / dist) * min(0.6, dist * 0.1) if dist > 1e-9 else np.zeros(3)

        effective_repulsion = repulsion_radius_m * (0.4 if is_final else 1.0)
        rep = np.zeros(3)
        for peer in peers:
            d = self.position - peer["pos"]
            r = float(np.linalg.norm(d))
            if 0 < r < effective_repulsion:
                unit = d / r
                force = ((effective_repulsion - r) / r)
                rep += unit * force * 0.15

        v = attr + rep
        s = float(np.linalg.norm(v))
        eff_max = max_speed if not is_final else max(0.05, dist * 0.2)
        if s > eff_max:
            v = (v / s) * eff_max

        # Unlock if the assigned target has shifted away from the locked
        # position by more than ~one approach radius — the assignment has
        # changed (e.g. due to a death rebalancing the bisection) and the
        # drone needs to move to its new slot. Without this, locked
        # surplus drones never migrate to fill vacated leaves.
        if self.locked and dist > 1.5 * approach_radius_m:
            self.locked = False
        if self.locked:
            v = np.zeros(3)
        elif is_final and dist < 0.3:
            v = np.zeros(3)
            self.locked = True

        # --- 8. Emit outgoing messages — ONLY when there's something new
        # to say or a liveness interval has elapsed. Minimal messaging.
        HB_POSITION_DELTA_M = 0.5      # emit HB if moved this far since last
        HB_LIVENESS_INTERVAL = 30       # ...or this many ticks have passed
        PV_LIVENESS_INTERVAL = 30       # PriorityVote: pure liveness ping

        # PriorityVote: priority doesn't change in our model, so only emit
        # for liveness on the interval (so peers don't time us out of their
        # known_priorities). Saves N×(K-1)/K of message traffic.
        if (self._last_pv_tick < 0
                or current_tick - self._last_pv_tick >= PV_LIVENESS_INTERVAL):
            self.priority_epoch_counter += 1
            pv = PriorityVote(
                origin=self.drone_id,
                priority=self.proto.my_priority,
                epoch=self.priority_epoch_counter,
                sig_stub=make_sig(self.drone_id, self.priority_epoch_counter, "P"),
            )
            outgoing.append(pv)
            self._last_pv_tick = current_tick

        # Heartbeat: emit if position changed meaningfully OR liveness fires.
        my_dr_pos = self._reported_position(current_tick)
        position_changed = (
            self._last_hb_position is None
            or float(np.linalg.norm(my_dr_pos - self._last_hb_position))
                >= HB_POSITION_DELTA_M
        )
        liveness_fired = (
            self._last_hb_tick < 0
            or current_tick - self._last_hb_tick >= HB_LIVENESS_INTERVAL
        )
        if position_changed or liveness_fired:
            my_dr_sigma = 1.0
            my_range_obs: dict[int, float] = {}
            for (observer, observed), (r, heard) in self.proto.range_obs.items():
                if observer == self.drone_id and (current_tick - heard) <= 30:
                    my_range_obs[observed] = float(r)
            self.heartbeat_epoch_counter += 1
            hb = Heartbeat(
                origin=self.drone_id,
                dr_position=my_dr_pos,
                dr_sigma=my_dr_sigma,
                range_obs=my_range_obs,
                epoch=self.heartbeat_epoch_counter,
                sig_stub=make_sig(self.drone_id, self.heartbeat_epoch_counter, "H"),
            )
            outgoing.append(hb)
            self._last_hb_tick = current_tick
            self._last_hb_position = my_dr_pos.copy()

        if self.refuse_to_forward:
            outgoing = [m for m in outgoing if isinstance(m, (PriorityVote, Heartbeat))]

        # Append multi-hop forwards. Each forward is annotated with its
        # origin and next-hop so the World knows to call comms.send with
        # the right header values.
        outgoing.extend(forwards)

        log = AgentStepLog(
            tick=current_tick,
            drone_id=self.drone_id,
            leader=leader_id,
            n_known=len(self.proto.known_positions),
            target=target.copy(),
            is_primary_at_leaf=is_primary,
            speed=float(np.linalg.norm(v)),
            locked=self.locked,
        )
        return v, outgoing, log


# ---------------------------------------------------------------------------
# Falsifiability tests of the agent in isolation.
# ---------------------------------------------------------------------------


def _tests() -> int:
    failed = 0

    # T1: an agent with no inbox holds position and reports itself as
    # leader by inference. Under the LEADERLESS Command model, the drone
    # has nothing new to propose so it doesn't emit a Command (minimal
    # messaging). Heartbeat + PriorityVote still emit every tick for
    # liveness.
    a = Agent(drone_id=0, priority=10, position=np.array([5.0, 0, 0]))
    v, out, log = a.step(current_tick=0, inbox=[])
    if log.leader != 0:
        print(f"FAIL T1: alone-leader should be self, got {log.leader}")
        failed += 1
    if log.n_known != 1:
        print(f"FAIL T1b: known-count alone should be 1, got {log.n_known}")
        failed += 1
    kinds = sorted(type(m).__name__ for m in out)
    expected = sorted(["Heartbeat", "PriorityVote"])
    if kinds != expected:
        print(f"FAIL T1c: outgoing kinds {kinds} != {expected}")
        failed += 1

    # T2: an agent receiving a higher-priority vote infers that peer as leader.
    a = Agent(drone_id=0, priority=2, position=np.array([0.0, 0, 0]))
    higher = PriorityVote(origin=5, priority=9, epoch=1, sig_stub=b"")
    class M:
        def __init__(self, p, origin=None, hop=1, range_at_send=0.0):
            self.payload = p
            self.origin = origin if origin is not None else getattr(p, "origin", 0)
            self.hop = hop
            self.range_at_send = range_at_send
    a.step(current_tick=0, inbox=[M(higher)])
    if infer_leader(a.proto, current_tick=0) != 5:
        print(f"FAIL T2: should infer leader=5, got {infer_leader(a.proto, 0)}")
        failed += 1

    # T3: the SAME drone with two different known sets computes different
    # targets. This is the falsifying behavior that proves the agent
    # actually depends on locally-known state, not on hidden global info.
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(8, 3)) * 5
    cmd = Command(origin=99, leader_priority=999, epoch=1, payload={
        "manifold_targets": targets,
        "heading": np.array([1.0, 0, 0]),
        "leg": 0,
    }, sig_stub=b"")
    # Run A: drone 1 knows peers at +x positions only.
    aA = Agent(drone_id=1, priority=10, position=np.array([0.0, 0, 0]))
    aA.proto.self_estimates[1] = (np.array([0.0, 0, 0]), 0, 0)
    if not hasattr(aA.proto, "dr_sigmas"):
        aA.proto.dr_sigmas = {}
    aA.proto.dr_sigmas[1] = 1.0
    for did, pos in [(2, [1,0,0]), (3, [2,0,0]), (4, [3,0,0]), (5, [4,0,0]),
                     (6, [5,0,0]), (7, [6,0,0]), (8, [7,0,0])]:
        aA.proto.self_estimates[did] = (np.asarray(pos, dtype=float), 0, 0)
        aA.proto.dr_sigmas[did] = 1.0
    aA.proto.latest_command = cmd; aA.proto.latest_command_tick = 0
    _, _, logA = aA.step(current_tick=0, inbox=[])
    # Run B: SAME drone (id=1), same position, but knows peers in -x.
    aB = Agent(drone_id=1, priority=10, position=np.array([0.0, 0, 0]))
    aB.proto.self_estimates[1] = (np.array([0.0, 0, 0]), 0, 0)
    if not hasattr(aB.proto, "dr_sigmas"):
        aB.proto.dr_sigmas = {}
    aB.proto.dr_sigmas[1] = 1.0
    for did, pos in [(2, [-1,0,0]), (3, [-2,0,0]), (4, [-3,0,0]), (5, [-4,0,0]),
                     (6, [-5,0,0]), (7, [-6,0,0]), (8, [-7,0,0])]:
        aB.proto.self_estimates[did] = (np.asarray(pos, dtype=float), 0, 0)
        aB.proto.dr_sigmas[did] = 1.0
    aB.proto.latest_command = cmd; aB.proto.latest_command_tick = 0
    _, _, logB = aB.step(current_tick=0, inbox=[])
    if np.allclose(logA.target, logB.target, atol=1e-6):
        print(f"FAIL T3: same drone, opposite-flipped known sets -> same target {logA.target}")
        failed += 1

    # T4: oracle-leak audit. Agent.step signature should not have any
    # parameter that would let it observe global state.
    import inspect
    sig = inspect.signature(Agent.step)
    forbidden_params = {
        "global_positions", "all_positions", "all_drones",
        "positions", "alive", "broadcast",
    }
    leaked = forbidden_params & set(sig.parameters.keys())
    if leaked:
        print(f"FAIL T4: Agent.step takes oracle-leaking params: {leaked}")
        failed += 1

    return failed


if __name__ == "__main__":
    n = _tests()
    print("agent: all tests passed" if n == 0 else f"agent: {n} tests failed")
