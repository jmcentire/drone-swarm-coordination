# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Per-drone agent: phase FSM + leader-initiated rounds.

Each agent runs the same code. State and decisions are local. No drone
reads another drone's internal state.

PHASE FSM (per-drone, ported from underwater/mission.py):
  WAITING -> got a Command -> MOVE
  MOVE -> reached approach radius -> SETTLE
  SETTLE -> own velocity below threshold for N ticks -> REFORM
  REFORM -> at target AND velocity low -> locked=True -> READY
  READY -> hold position; if leader, run round orchestration

LEADER-INITIATED ROUNDS (only emitted by the drone currently inferring
itself as leader):
  - MAP_CALL when the plan's expected completion tick arrives OR own
    READY makes the leg clearly complete -- gathers updated dr_pos +
    range obs from peers
  - VOTE_CALL after a MAP round closes -- collects candidate plan ranks
    (not a democratic vote; a flood-max arrangement step)
  - COMMAND after a VOTE round -- issues the selected directive

EVERY OTHER DRONE responds reactively:
  - On MAP_CALL: emit MAP_RESPONSE with own dr_pos + accumulated range obs
  - On VOTE_CALL: emit VOTE_RESPONSE with (priority, epoch)
  - On Command: adopt by (priority, epoch) max, forward via TTL gossip

OH_SHIT is gated to MOVE/SETTLE/REFORM phases. Triggered by:
  - leader-side byzantine detection (post-IRLS residual > threshold)
  - participant count collapse (fresh peers < threshold)
  - own dr_pos disagreeing with consensus by > threshold (own sensor bad)
Any accepted OhShit pushes the receiver into a bounded alert mode. During
alert it steers to the current plan's rally point if one exists, otherwise
to the safest known stationary target it has. The alert is intentionally
not a new periodic mode; the OhShit packet is the event.

PASSIVE RANGING: any received message yields a ToF measurement via the
substrate's msg.range_at_send. There is no standalone ping. Between
rounds = silence = no fresh ranges = aged consensus.

NO PERIODIC BROADCAST. NO PER-MESSAGE BYZANTINE FILTER. The IRLS
handles byzantine outliers via Huber + residual reweighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from manifold import compute_target
from protocol import (
    Command,
    DEFAULT_FRESHNESS_TICKS,
    Map,
    MsgKind,
    OhShit,
    OhShitKind,
    ProtocolState,
    Vote,
    compute_consensus_positions,
    fresh_participant_count,
    infer_leader,
    ingest_command,
    ingest_map,
    ingest_oh_shit,
    ingest_vote,
    make_sig,
)


MAX_HOPS = 6


class Phase(Enum):
    WAITING = "waiting"   # no Command yet
    MOVE = "move"         # heading to target
    SETTLE = "settle"     # decelerating near target
    REFORM = "reform"     # snapping into lock
    READY = "ready"       # locked, swarm-ready


@dataclass
class AgentStepLog:
    tick: int
    drone_id: int
    phase: str
    leader: int
    n_known: int
    n_participants: int
    target: np.ndarray
    is_primary_at_slot: bool
    speed: float
    locked: bool


def _fibonacci_manifold(n_targets: int) -> np.ndarray:
    """Fibonacci-sphere manifold. Matches the bench's _make_manifold."""
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
    # Bench-injected fault modes:
    lie_about_position: Any = None     # callable(tick) -> np.ndarray offset
    refuse_to_forward: bool = False
    spam_priority: int | None = None
    # Reformation / recovery knobs:
    enable_reformation: bool = False
    initial_n_drones: int = 0
    loss_threshold_pct: float = 0.15

    # --- Phase FSM thresholds ---
    target_arrival_m: float = 0.6     # within this distance == "at target"
    lock_velocity_mps: float = 0.15   # below this speed == "slow enough to lock"
    settle_speed_threshold: float = 0.20
    settle_quiescent_ticks: int = 8   # consecutive slow ticks to enter REFORM
    reform_locked_ticks: int = 5      # consecutive locked ticks to enter READY

    # --- Leader-side round scheduling ---
    map_round_interval_ticks: int = 30   # de-dup throttle for plan-timeout/READY maps
    map_response_window_ticks: int = 8   # how long leader waits for responses
    vote_response_window_ticks: int = 8
    expected_completion_margin_ticks: int = 20

    # --- Byzantine residual detection thresholds ---
    byzantine_residual_threshold_m: float = 15.0
    own_sensor_residual_threshold_m: float = 25.0
    participant_collapse_threshold: float = 0.5  # frac of initial_n_drones
    consensus_irls_iters: int = 2
    consensus_refresh_ticks: int = 15
    # Throttle for the leader-side byzantine residual check. Byzantine
    # state doesn't change every tick; running this every tick doubles
    # consensus cost. Once per 15 ticks gives plenty of detection
    # latency while halving compute on the leader.
    byzantine_check_interval_ticks: int = 15
    alert_duration_ticks: int = 80

    def __post_init__(self) -> None:
        eff_priority = self.spam_priority if self.spam_priority is not None else self.priority
        self.proto = ProtocolState(
            drone_id=self.drone_id,
            my_priority=eff_priority,
            my_epoch=0,
        )
        # Bootstrap: seed my own dr_pos so consensus sees me.
        self.proto.self_estimates[self.drone_id] = (self.position.copy(), 0, 0)
        self.proto.dr_sigmas[self.drone_id] = 1.0
        self.proto.known_positions[self.drone_id] = (self.position.copy(), 0, 0)

        # FSM state
        self.phase: Phase = Phase.WAITING
        self.phase_start_tick: int = 0
        self._speed_history: list[float] = []
        self._locked_streak: int = 0

        # Round bookkeeping
        self.map_round_counter: int = 0    # leader's local counter
        self.vote_round_counter: int = 0
        self._last_map_call_tick: int = -10_000
        self._last_vote_call_tick: int = -10_000
        self._last_vote_for_map_round: int = -1
        self._map_responses_received: dict[int, int] = {}   # round_id -> n responses
        self._vote_responses_received: dict[int, int] = {}
        # Per-round queued response (key: (round_kind, round_id) so each call
        # is responded to once even if duplicates are forwarded to us)
        self._pending_map_response: tuple[int, int] | None = None  # (leader_id, round_id)
        self._pending_vote_response: tuple[int, int] | None = None

        # Range obs accumulated since I last emitted a MAP_RESPONSE
        self._range_obs_buffer: dict[int, float] = {}
        self._signal_strength_buffer: dict[int, float] = {}

        # Epoch counters per emission kind
        self.my_dr_epoch: int = 0
        self.my_command_epoch: int = 0
        self.my_vote_epoch: int = 0
        self.my_ohshit_epoch: int = 0

        # Gossip-forwarding dedup
        self.seen_messages: set[tuple[str, int, int, int]] = set()  # (kind, origin, epoch, round_id)

        # Consensus cache
        self._consensus_cache: dict[int, np.ndarray] = {}
        self._consensus_last_tick: int = -1
        self._dirty_since_consensus: bool = False

        # OhShit cooldown so a stuck condition doesn't flood
        self._last_ohshit_tick: dict[OhShitKind, int] = {}
        self._ohshit_cooldown_ticks: int = 30
        self._last_byzantine_check_tick: int = -10_000
        self._alert_until_tick: int = -1
        self._alert_reason: OhShitKind | None = None

    def _current_command_payload(self) -> dict[str, Any]:
        if self.proto.latest_command is None:
            return {}
        payload = self.proto.latest_command.payload
        return payload if isinstance(payload, dict) else {}

    def _expected_completion_tick(self, fallback_tick: int) -> int:
        payload = self._current_command_payload()
        explicit = payload.get("expected_completion_tick")
        if explicit is not None:
            return int(explicit)
        targets = payload.get("manifold_targets")
        if targets is None:
            return fallback_tick
        try:
            target, _ = compute_target(
                self.drone_id,
                [{"id": self.drone_id, "pos": self.position.copy()}],
                targets,
            )
            dist = float(np.linalg.norm(target - self.position))
        except Exception:
            dist = 0.0
        # max_speed defaults to 0.8 in step(); keep this conservative so
        # a stuck peer does not block the next map/arrangement round.
        return self.proto.latest_command_tick + int(np.ceil(dist / 0.8)) + self.expected_completion_margin_ticks

    def _rally_target(self) -> np.ndarray | None:
        payload = self._current_command_payload()
        rally = payload.get("rally_points")
        if rally is None:
            rally = payload.get("rally_point")
        if rally is None:
            targets = payload.get("manifold_targets")
            if targets is None:
                return None
            pts = np.asarray(targets, dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0:
                return np.mean(pts, axis=0)
            return None
        pts = np.asarray(rally, dtype=np.float64)
        if pts.ndim == 1 and pts.shape[0] == 3:
            return pts.copy()
        if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0:
            d = np.linalg.norm(pts - self.position[None, :], axis=1)
            return pts[int(np.argmin(d))].copy()
        return None

    def _enter_alert(self, kind: OhShitKind, tick: int) -> None:
        self._alert_reason = kind
        self._alert_until_tick = max(self._alert_until_tick, tick + self.alert_duration_ticks)
        self.locked = False
        if self.phase == Phase.READY:
            self.phase = Phase.MOVE
            self.phase_start_tick = tick

    # ------------------------------------------------------------------
    # Local self-estimate update (called each tick before emission decisions)
    # ------------------------------------------------------------------

    def _refresh_my_self_estimate(self, tick: int) -> None:
        """Update my own entry in self_estimates with current dr_pos.
        For honest drones, dr_pos == self.position. For byzantines, it's
        the lie."""
        pos = self._reported_position(tick)
        self.proto.self_estimates[self.drone_id] = (pos.copy(), self.my_dr_epoch, tick)
        self.proto.known_positions[self.drone_id] = (pos.copy(), self.my_dr_epoch, tick)
        self.proto.dr_sigmas[self.drone_id] = 1.0

    def _reported_position(self, tick: int) -> np.ndarray:
        if self.lie_about_position is not None:
            return np.asarray(self.lie_about_position(tick), dtype=np.float64)
        return self.position.copy()

    # ------------------------------------------------------------------
    # Phase FSM
    # ------------------------------------------------------------------

    def _update_phase(self, tick: int, current_speed: float, dist_to_target: float) -> None:
        # WAITING: leave when a Command has been adopted.
        if self.phase == Phase.WAITING:
            if self.proto.latest_command is not None:
                self.phase = Phase.MOVE
                self.phase_start_tick = tick
                self._speed_history.clear()
                self._locked_streak = 0
            return

        # MOVE: enter SETTLE once we approach the target slot.
        if self.phase == Phase.MOVE:
            if dist_to_target < self.target_arrival_m * 2.5:
                self.phase = Phase.SETTLE
                self.phase_start_tick = tick
                self._speed_history.clear()
            return

        # SETTLE: enter REFORM after N consecutive slow ticks.
        if self.phase == Phase.SETTLE:
            self._speed_history.append(current_speed)
            if len(self._speed_history) > self.settle_quiescent_ticks:
                self._speed_history.pop(0)
            if (
                len(self._speed_history) == self.settle_quiescent_ticks
                and max(self._speed_history) < self.settle_speed_threshold
            ):
                self.phase = Phase.REFORM
                self.phase_start_tick = tick
                self._locked_streak = 0
            # If target moved out from under me (re-assignment), back to MOVE.
            if dist_to_target > self.target_arrival_m * 4.0:
                self.phase = Phase.MOVE
                self.phase_start_tick = tick
                self._speed_history.clear()
            return

        # REFORM: try to lock to target.
        if self.phase == Phase.REFORM:
            am_locked_now = (
                dist_to_target < self.target_arrival_m
                and current_speed < self.lock_velocity_mps
            )
            if am_locked_now:
                self.locked = True
                self._locked_streak += 1
            else:
                self.locked = False
                self._locked_streak = 0
            if self._locked_streak >= self.reform_locked_ticks:
                self.phase = Phase.READY
                self.phase_start_tick = tick
            # If target shifted (re-assignment): unlock and back to MOVE.
            if dist_to_target > self.target_arrival_m * 4.0:
                self.locked = False
                self.phase = Phase.MOVE
                self.phase_start_tick = tick
                self._speed_history.clear()
            return

        # READY: hold; transition back to MOVE if target shifted.
        if self.phase == Phase.READY:
            if dist_to_target > self.target_arrival_m * 2.0:
                self.locked = False
                self.phase = Phase.MOVE
                self.phase_start_tick = tick
                self._speed_history.clear()
                self._locked_streak = 0
            return

    # ------------------------------------------------------------------
    # OhShit triggers (only emitted during MOVE/SETTLE/REFORM)
    # ------------------------------------------------------------------

    def _check_oh_shit(self, tick: int) -> list[OhShit]:
        if self.phase not in (Phase.MOVE, Phase.SETTLE, Phase.REFORM):
            return []
        events: list[OhShit] = []

        # Cooldown helper
        def can_fire(kind: OhShitKind) -> bool:
            last = self._last_ohshit_tick.get(kind, -10_000)
            return tick - last >= self._ohshit_cooldown_ticks

        # 1. own sensor failure: my own dr_pos disagrees with consensus
        if self.drone_id in self._consensus_cache:
            c_pos = self._consensus_cache[self.drone_id]
            my_pos = self._reported_position(tick)
            if float(np.linalg.norm(c_pos - my_pos)) > self.own_sensor_residual_threshold_m \
                    and can_fire(OhShitKind.OWN_SENSOR_FAILURE):
                events.append(self._make_ohshit(
                    OhShitKind.OWN_SENSOR_FAILURE,
                    {"my_dr": my_pos.tolist(), "consensus": c_pos.tolist()},
                ))
                self._last_ohshit_tick[OhShitKind.OWN_SENSOR_FAILURE] = tick
                self._enter_alert(OhShitKind.OWN_SENSOR_FAILURE, tick)

        # 2. participant collapse
        if self.initial_n_drones > 0:
            np_count = fresh_participant_count(self.proto, tick)
            threshold = max(2, int(self.initial_n_drones * self.participant_collapse_threshold))
            if np_count < threshold and can_fire(OhShitKind.PARTICIPANT_COLLAPSE):
                events.append(self._make_ohshit(
                    OhShitKind.PARTICIPANT_COLLAPSE,
                    {"n_visible": np_count, "expected_min": threshold},
                ))
                self._last_ohshit_tick[OhShitKind.PARTICIPANT_COLLAPSE] = tick
                self._enter_alert(OhShitKind.PARTICIPANT_COLLAPSE, tick)

        # 3. byzantine peer (leader-side only -- everyone could detect but
        # only the leader broadcasts to avoid duplication storms). Use the
        # already-computed self._consensus_cache and run the residual check
        # only every byzantine_check_interval_ticks; rerunning IRLS per tick
        # here would double consensus cost on the leader.
        leader_id = infer_leader(self.proto, tick)
        do_check = (
            leader_id == self.drone_id
            and can_fire(OhShitKind.BYZANTINE_PEER)
            and (tick - self._last_byzantine_check_tick) >= self.byzantine_check_interval_ticks
        )
        if do_check:
            self._last_byzantine_check_tick = tick
            suspects: list[int] = []
            for origin, c_pos in self._consensus_cache.items():
                if origin == self.drone_id:
                    continue
                if origin not in self.proto.self_estimates:
                    continue
                dr_pos, _, _ = self.proto.self_estimates[origin]
                if float(np.linalg.norm(c_pos - dr_pos)) > self.byzantine_residual_threshold_m:
                    suspects.append(origin)
            if suspects:
                events.append(self._make_ohshit(
                    OhShitKind.BYZANTINE_PEER,
                    {"suspects": suspects},
                ))
                self._last_ohshit_tick[OhShitKind.BYZANTINE_PEER] = tick
                self._enter_alert(OhShitKind.BYZANTINE_PEER, tick)

        return events

    def _make_ohshit(self, kind: OhShitKind, payload: Any) -> OhShit:
        self.my_ohshit_epoch += 1
        return OhShit(
            origin=self.drone_id,
            kind=kind,
            payload=payload,
            epoch=self.my_ohshit_epoch,
            sig_stub=make_sig(self.drone_id, self.my_ohshit_epoch, "O"),
        )

    # ------------------------------------------------------------------
    # Leader round orchestration
    # ------------------------------------------------------------------

    def _maybe_initiate_map(self, tick: int, am_leader: bool) -> Map | None:
        """Leader emits MAP_CALL when the current plan should have finished
        or this drone has locally reached READY. Gated to
        post-WAITING phases -- a drone with no Command yet has no
        situational-awareness round to run."""
        if not am_leader or self.phase == Phase.WAITING:
            return None
        since_last_map = tick - self._last_map_call_tick
        locally_ready = (
            self.phase == Phase.READY
            and tick - self.phase_start_tick >= self.reform_locked_ticks
        )
        expected_done = tick >= self._expected_completion_tick(tick)
        # map_round_interval_ticks is now a de-duplication throttle, not
        # a liveness poll. Silence before expected completion is normal.
        if not ((locally_ready or expected_done) and since_last_map >= self.map_round_interval_ticks):
            return None
        self.map_round_counter += 1
        self._last_map_call_tick = tick
        self.my_dr_epoch += 1
        return Map(
            kind=MsgKind.CALL,
            origin=self.drone_id,
            epoch=self.my_dr_epoch,
            round_id=self.map_round_counter,
            dr_position=self._reported_position(tick),
            dr_sigma=1.0,
            range_obs={},  # CALL doesn't ship range_obs; responders do
            sig_stub=make_sig(self.drone_id, self.map_round_counter, "MC"),
        )

    def _build_map_response(self, leader_id: int, round_id: int, tick: int) -> Map:
        """RESPONSE carries my dr_pos + accumulated range observations
        I've collected since my last response."""
        self.my_dr_epoch += 1
        # Flush the buffer of range obs accumulated since last response.
        # Filter to only include observations within freshness.
        my_obs: dict[int, float] = {}
        for obs_id, r in self._range_obs_buffer.items():
            my_obs[int(obs_id)] = float(r)
        self._range_obs_buffer.clear()
        strength_obs: dict[int, float] = {}
        for obs_id, strength in self._signal_strength_buffer.items():
            strength_obs[int(obs_id)] = float(strength)
        self._signal_strength_buffer.clear()
        return Map(
            kind=MsgKind.RESPONSE,
            origin=self.drone_id,
            epoch=self.my_dr_epoch,
            round_id=round_id,
            dr_position=self._reported_position(tick),
            dr_sigma=1.0,
            range_obs=my_obs,
            sig_stub=make_sig(self.drone_id, self.my_dr_epoch, "MR"),
            signal_strength_obs=strength_obs,
        )

    def _build_vote_response(self, leader_id: int, round_id: int, tick: int) -> Vote:
        self.my_vote_epoch += 1
        return Vote(
            kind=MsgKind.RESPONSE,
            origin=self.drone_id,
            priority=self.proto.my_priority,
            epoch=self.my_vote_epoch,
            round_id=round_id,
            sig_stub=make_sig(self.drone_id, self.my_vote_epoch, "VR"),
        )

    def _maybe_initiate_vote(self, tick: int, am_leader: bool) -> Vote | None:
        """Leader starts the arrangement round after its MAP response window.

        This is not a periodic liveness vote. It is the second half of the
        plan-timeout/READY event: after MAP gathers situational awareness,
        Vote flood-max propagates the highest-ranked plan/authority visible
        to this component.
        """
        if not am_leader or self.map_round_counter <= 0:
            return None
        if self._last_vote_for_map_round == self.map_round_counter:
            return None
        if tick - self._last_map_call_tick < self.map_response_window_ticks:
            return None
        self.vote_round_counter += 1
        self._last_vote_call_tick = tick
        self._last_vote_for_map_round = self.map_round_counter
        self.my_vote_epoch += 1
        return Vote(
            kind=MsgKind.CALL,
            origin=self.drone_id,
            priority=self.proto.my_priority,
            epoch=self.my_vote_epoch,
            round_id=self.vote_round_counter,
            sig_stub=make_sig(self.drone_id, self.my_vote_epoch, "VC"),
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

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
        forwards: list[tuple[Any, int, int]] = []  # (payload, next_hop, origin)
        new_response_pending: tuple[str, int, int] | None = None  # debug

        # ----- 1. Ingest inbox -----
        for msg in inbox:
            payload = msg.payload
            origin = msg.origin
            measured_range = getattr(msg, "range_at_send", None)
            signal_strength = getattr(msg, "signal_strength_at_receive", None)
            # Dedup key: (kind, origin, epoch, round_id_or_0)
            kind_name = type(payload).__name__
            epoch = getattr(payload, "epoch", 0)
            round_id = getattr(payload, "round_id", 0)
            key = (kind_name, origin, epoch, round_id)
            if key in self.seen_messages:
                continue
            self.seen_messages.add(key)

            if isinstance(payload, Map):
                changed = ingest_map(
                    self.proto, payload, current_tick,
                    measured_range_to_sender=measured_range,
                )
                if changed:
                    self._dirty_since_consensus = True
                # If CALL: queue a response (one per leader+round_id).
                if payload.kind == MsgKind.CALL:
                    key2 = (payload.origin, payload.round_id)
                    if self.proto.last_map_round_responded != key2:
                        self._pending_map_response = key2
                # If RESPONSE: leader-side tally for the round
                elif payload.kind == MsgKind.RESPONSE:
                    self._map_responses_received[payload.round_id] = (
                        self._map_responses_received.get(payload.round_id, 0) + 1
                    )
                # Record my own range to the sender (substrate-measured)
                if measured_range is not None:
                    self._range_obs_buffer[origin] = float(measured_range)
                if signal_strength is not None:
                    self._signal_strength_buffer[origin] = float(signal_strength)

            elif isinstance(payload, Vote):
                changed = ingest_vote(
                    self.proto, payload, current_tick,
                    measured_range_to_sender=measured_range,
                )
                if changed:
                    self._dirty_since_consensus = True
                if payload.kind == MsgKind.CALL:
                    key2 = (payload.origin, payload.round_id)
                    if self.proto.last_vote_round_responded != key2:
                        self._pending_vote_response = key2
                elif payload.kind == MsgKind.RESPONSE:
                    self._vote_responses_received[payload.round_id] = (
                        self._vote_responses_received.get(payload.round_id, 0) + 1
                    )
                if measured_range is not None:
                    self._range_obs_buffer[origin] = float(measured_range)
                if signal_strength is not None:
                    self._signal_strength_buffer[origin] = float(signal_strength)

            elif isinstance(payload, Command):
                ingest_command(
                    self.proto, payload, current_tick,
                    measured_range_to_sender=measured_range,
                )
                if measured_range is not None:
                    self._range_obs_buffer[origin] = float(measured_range)
                if signal_strength is not None:
                    self._signal_strength_buffer[origin] = float(signal_strength)

            elif isinstance(payload, OhShit):
                ingest_oh_shit(
                    self.proto, payload, current_tick,
                    measured_range_to_sender=measured_range,
                )
                self._enter_alert(payload.kind, current_tick)
                if measured_range is not None:
                    self._range_obs_buffer[origin] = float(measured_range)
                if signal_strength is not None:
                    self._signal_strength_buffer[origin] = float(signal_strength)

            # Forward all live (non-stale) messages.
            if not self.refuse_to_forward and msg.hop < MAX_HOPS:
                forwards.append((payload, msg.hop + 1, origin))

        # ----- 2. Update my own self_estimate (so consensus sees me) -----
        self._refresh_my_self_estimate(current_tick)

        # ----- 3. Maybe recompute consensus (throttled) -----
        if (
            self._dirty_since_consensus
            and (current_tick - self._consensus_last_tick) >= self.consensus_refresh_ticks
        ) or self._consensus_last_tick < 0:
            self._consensus_cache = compute_consensus_positions(
                self.proto,
                current_tick,
                n_irls_iters=self.consensus_irls_iters,
            )
            self._consensus_last_tick = current_tick
            self._dirty_since_consensus = False

        consensus = self._consensus_cache
        leader_id = infer_leader(self.proto, current_tick)
        am_leader = leader_id == self.drone_id

        # ----- 4. Build target via Hungarian (compute_target) -----
        my_entry = {"id": self.drone_id, "pos": self.position.copy()}
        peers = [
            {"id": did, "pos": pos.copy()}
            for did, pos in consensus.items() if did != self.drone_id
        ]
        known = [my_entry] + peers

        manifold_targets = None
        if self.proto.latest_command is not None:
            mt = self.proto.latest_command.payload.get("manifold_targets")
            if mt is not None:
                manifold_targets = mt
        rally_target = self._rally_target() if current_tick <= self._alert_until_tick else None
        if rally_target is not None:
            target = rally_target
            is_primary = True
        elif manifold_targets is not None:
            target, is_primary = compute_target(self.drone_id, known, manifold_targets)
        else:
            target = self.position.copy()
            is_primary = False

        # ----- 5. Steering -----
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

        # Lock dynamics: snap to zero when locked; release if assignment drifted.
        if self.locked and dist > 1.5 * approach_radius_m:
            self.locked = False
        if self.locked:
            v = np.zeros(3)

        # ----- 6. Phase FSM update (uses current speed AFTER capping) -----
        current_speed = float(np.linalg.norm(v))
        self._update_phase(current_tick, current_speed, dist)

        # ----- 7. Emission decisions -----
        outgoing: list[Any] = []

        # 7a. Pending MAP/VOTE responses (always allowed regardless of phase).
        if self._pending_map_response is not None:
            leader_of_call, round_of_call = self._pending_map_response
            outgoing.append(self._build_map_response(leader_of_call, round_of_call, current_tick))
            self.proto.last_map_round_responded = self._pending_map_response
            self._pending_map_response = None
        if self._pending_vote_response is not None:
            leader_of_call, round_of_call = self._pending_vote_response
            outgoing.append(self._build_vote_response(leader_of_call, round_of_call, current_tick))
            self.proto.last_vote_round_responded = self._pending_vote_response
            self._pending_vote_response = None

        # 7b. Leader-side: initiate next round if timing dictates.
        map_call = self._maybe_initiate_map(current_tick, am_leader)
        if map_call is not None:
            outgoing.append(map_call)
        vote_call = self._maybe_initiate_vote(current_tick, am_leader)
        if vote_call is not None:
            outgoing.append(vote_call)

        # 7c. OhShit (gated to MOVE/SETTLE/REFORM).
        for ev in self._check_oh_shit(current_tick):
            outgoing.append(ev)

        # 7d. Bootstrap: if I am the leader AND there is a fresh Command
        # I haven't been the origin of, re-issue/forward it so it propagates.
        # (Forwarding already does this via the forwards list; nothing extra
        # to do at the broadcast layer here.)

        # ----- 8. Append forwards -----
        outgoing.extend(forwards)

        # ----- 9. Build log -----
        n_part = fresh_participant_count(self.proto, current_tick)
        log = AgentStepLog(
            tick=current_tick,
            drone_id=self.drone_id,
            phase=self.phase.value,
            leader=leader_id,
            n_known=len(self.proto.known_positions),
            n_participants=n_part,
            target=target.copy(),
            is_primary_at_slot=is_primary,
            speed=current_speed,
            locked=self.locked,
        )
        return v, outgoing, log


# ---------------------------------------------------------------------------
# Self-tests.
# ---------------------------------------------------------------------------


def _tests() -> int:
    failed = 0

    # T1: alone -- no inbox, no Command. Drone stays in WAITING; no broadcast.
    a = Agent(drone_id=0, priority=10, position=np.array([5.0, 0, 0]))
    v, out, log = a.step(current_tick=0, inbox=[])
    if log.phase != "waiting":
        print(f"FAIL T1a: expected waiting, got {log.phase}")
        failed += 1
    if out:
        print(f"FAIL T1b: expected zero emissions when alone+waiting, got {len(out)}")
        failed += 1
    if log.leader != 0:
        print(f"FAIL T1c: alone-leader should be self, got {log.leader}")
        failed += 1

    # T2: after receiving a Command, agent transitions WAITING -> MOVE on next step.
    a = Agent(drone_id=0, priority=10, position=np.array([0, 0, 0]))
    manifold = np.array([[10.0, 0, 0]])
    cmd = Command(
        origin=99, leader_priority=100, epoch=0,
        payload={"manifold_targets": manifold, "heading": np.array([1, 0, 0])},
        sig_stub=b"",
    )
    class _Msg:
        def __init__(self, payload, origin, hop=0, range_at_send=None):
            self.payload = payload; self.origin = origin; self.hop = hop
            self.range_at_send = range_at_send
    a.step(current_tick=0, inbox=[_Msg(cmd, origin=99, range_at_send=5.0)])
    v, out, log = a.step(current_tick=1, inbox=[])
    if log.phase != "move":
        print(f"FAIL T2: after Command, expected move, got {log.phase}")
        failed += 1

    # T3: leader periodically initiates MAP_CALL (>=1 MAP within map_round_interval_ticks).
    a = Agent(drone_id=0, priority=10, position=np.array([0, 0, 0]))
    a.proto.latest_command = cmd
    a.phase = Phase.READY
    a.phase_start_tick = 0
    emitted_map_calls = 0
    for t in range(50):
        _, out, _ = a.step(current_tick=t, inbox=[])
        emitted_map_calls += sum(
            1 for o in out if isinstance(o, Map) and o.kind == MsgKind.CALL
        )
    if emitted_map_calls < 1:
        print(f"FAIL T3: leader in READY should emit at least one MAP_CALL in 50 ticks, got {emitted_map_calls}")
        failed += 1
    # And it should NOT be many -- the cap is map_round_interval_ticks.
    if emitted_map_calls > 5:
        print(f"FAIL T3: leader emitted too many MAP_CALLS in 50 ticks: {emitted_map_calls}")
        failed += 1
    emitted_vote_calls = 0
    for t in range(50, 90):
        _, out, _ = a.step(current_tick=t, inbox=[])
        emitted_vote_calls += sum(
            1 for o in out if isinstance(o, Vote) and o.kind == MsgKind.CALL
        )
    if emitted_vote_calls < 1:
        print(f"FAIL T3c: leader should emit VOTE_CALL after MAP window, got {emitted_vote_calls}")
        failed += 1

    # T4: non-leader receives MAP_CALL, responds once.
    a = Agent(drone_id=5, priority=5, position=np.array([0, 0, 0]))
    leader_call = Map(
        kind=MsgKind.CALL, origin=99, epoch=0, round_id=1,
        dr_position=np.array([3, 0, 0]), dr_sigma=1.0, range_obs={},
        sig_stub=b"",
    )
    # Make 99 visible as a peer so it can be inferred as leader
    a.proto.known_priorities[99] = (100, 0, 0)
    inbox = [_Msg(leader_call, origin=99, range_at_send=3.0)]
    _, out, _ = a.step(current_tick=0, inbox=inbox)
    responses = [o for o in out if isinstance(o, Map) and o.kind == MsgKind.RESPONSE]
    if len(responses) != 1:
        print(f"FAIL T4: expected 1 MAP_RESPONSE, got {len(responses)}")
        failed += 1
    # Duplicate of same call -- should not re-respond.
    inbox2 = [_Msg(leader_call, origin=99, range_at_send=3.0)]
    _, out2, _ = a.step(current_tick=1, inbox=inbox2)
    dups = [o for o in out2 if isinstance(o, Map) and o.kind == MsgKind.RESPONSE]
    if dups:
        print(f"FAIL T4: duplicate call should not produce a second response, got {len(dups)}")
        failed += 1

    # T5: OhShit moves receiver into alert mode and targets rally point.
    a = Agent(drone_id=2, priority=2, position=np.array([10.0, 0.0, 0.0]))
    rally = np.array([1.0, 2.0, 3.0])
    a.proto.latest_command = Command(
        origin=9,
        leader_priority=9,
        epoch=0,
        payload={
            "manifold_targets": np.array([[10.0, 0.0, 0.0]]),
            "rally_point": rally,
            "expected_completion_tick": 100,
        },
        sig_stub=b"",
    )
    a.phase = Phase.READY
    alert = OhShit(
        origin=1,
        kind=OhShitKind.PARTICIPANT_COLLAPSE,
        payload={"n_visible": 1},
        epoch=1,
        sig_stub=b"",
    )
    _, _, log = a.step(current_tick=10, inbox=[_Msg(alert, origin=1, range_at_send=4.0)])
    if log.phase != "move":
        print(f"FAIL T5a: alert should leave READY for MOVE, got {log.phase}")
        failed += 1
    if float(np.linalg.norm(log.target - rally)) > 1e-9:
        print(f"FAIL T5b: alert target should be rally point, got {log.target}")
        failed += 1

    return failed


if __name__ == "__main__":
    n = _tests()
    print("agent: all tests passed" if n == 0 else f"agent: {n} tests failed")
