# /// script
# dependencies = ["numpy<3"]
# ///
"""Range-limited acoustic comms substrate with propagation delay.

Replaces simulator.py's Broadcast (everyone hears everyone instantly).

Design principles, all motivated by Sim/Advocate critique of the
prior plan:

  - Propagation delay: receive_tick = send_tick + ceil(d/c) where c is
    sound speed in water (~1500 m/s). Messages are NOT instant.
  - Signal strength degrades with range. Receivers get a coarse
    strength scalar on every packet, giving them a passive bearing cue
    when they vary their own position and observe whether strength rises
    or falls. This is a physical side-effect of communication, not a
    separate ranging ping.
  - Range check at RECEIVE time, not send time. A message sent in-range
    can still be lost if the receiver moves out of range before it
    arrives. Messages also dropped if sender or receiver dies between.
  - Per-tick per-drone random processing order. Drones do NOT all merge
    state before all acting — each drone, in random order, processes its
    inbox AND computes its action atomically per tick.
  - Per-drone event log captures (tick, drone_id, kind, info). Used by
    benches to audit that the substrate is not a synchronous broadcast
    in disguise: real gossip MUST show some drones acting on stale info.
  - Per-message loss draw at receive time (uniform iid loss rate per
    edge per tick).

The unit of "tick" is the agent decision cycle (e.g., 100 ms). The
"sound speed" parameter is in units of meters per tick. At a 100 ms tick
and 1500 m/s sound speed, that's 150 m/tick — so most short-range
messages arrive within one tick, but long-range ones can be delayed
multiple ticks. For an HCP lattice with 10 m spacing, near-neighbors
are 1-tick, far edges of the lattice may be 2-3 ticks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


SOUND_SPEED_M_PER_TICK_DEFAULT = 150.0  # ~1500 m/s at 100 ms tick


@dataclass
class Message:
    """An in-flight acoustic packet."""
    origin: int               # drone_id that originally produced the payload
    sender: int               # most-recent forwarder (for hop tracking)
    sent_tick: int            # tick the message was launched
    deliver_tick: int         # tick it arrives at the destination
    receiver: int             # intended destination drone_id
    sender_pos_at_send: np.ndarray  # snapshot for receive-time range check
    payload: Any              # opaque; gossip layer interprets
    sig_stub: bytes           # 16-byte tag; protocol verifies origin == claimed
    hop: int = 0              # propagation hop count
    # Bookkeeping for the audit log:
    range_at_send: float = 0.0
    signal_strength_at_send: float = 0.0
    signal_strength_at_receive: float = 0.0


@dataclass
class CommsEvent:
    """Per-drone-per-tick log entry for substrate audit."""
    tick: int
    drone_id: int
    kind: str    # 'SEND' | 'RECV' | 'DROP_RANGE' | 'DROP_LOSS' | 'DROP_DEAD' | 'ACT'
    info: dict = field(default_factory=dict)


class LocalComms:
    """Range-limited message-passing with propagation delay.

    Public surface:
        send(sender, payload, sig_stub)         -> enqueue to all neighbors in range
        deliver(receiver, current_tick)          -> list of Messages now arriving
        agent_step_order(rng)                    -> shuffled drone_id order
        events                                    -> list of CommsEvent (audit log)
    """

    def __init__(
        self,
        n: int,
        comms_range_m: float,
        loss_rate: float = 0.0,
        sound_speed: float = SOUND_SPEED_M_PER_TICK_DEFAULT,
        log_events: bool = True,
    ) -> None:
        self.n = n
        self.comms_range_m = float(comms_range_m)
        self.loss_rate = float(loss_rate)
        self.sound_speed = float(sound_speed)
        # In-flight messages indexed by deliver_tick for fast lookup.
        self._inflight: dict[int, list[Message]] = {}
        self.events: list[CommsEvent] = []
        self.log_events = log_events
        # Audit: cumulative counters.
        self.n_sent = 0
        self.n_delivered = 0
        self.n_dropped_loss = 0
        self.n_dropped_range = 0
        self.n_dropped_dead = 0
        self._seen_packet_receivers: set[tuple[int, str, int, int, int]] = set()

    def _packet_receiver_key(
        self, receiver: int, origin: int, payload: Any
    ) -> tuple[int, str, int, int, int | str]:
        kind = type(payload).__name__
        epoch = int(getattr(payload, "epoch", 0))
        round_id = int(getattr(payload, "round_id", 0))
        if not hasattr(payload, "origin") and not hasattr(payload, "epoch"):
            round_key: int | str = repr(payload)
        else:
            round_key = round_id
        return (int(receiver), kind, int(origin), epoch, round_key)

    def signal_strength(self, distance_m: float) -> float:
        """Coarse acoustic strength model.

        This intentionally avoids pretending to be a full sonar channel:
        spreading loss is enough for the protocol to ask directional
        questions like "did my small motion make this peer louder or
        quieter?" Strength is normalized to 1.0 at zero range and decays
        monotonically with distance.
        """
        d = max(0.0, float(distance_m))
        return 1.0 / (1.0 + d * d)

    def send(
        self,
        sender_id: int,
        sender_pos: np.ndarray,
        payload: Any,
        sig_stub: bytes,
        positions: np.ndarray,
        alive: np.ndarray,
        current_tick: int,
        rng: np.random.Generator,
        origin: int | None = None,
        starting_hop: int = 1,
    ) -> None:
        """Broadcast `payload` from sender_id to all *currently* in-range drones.

        The range check at SEND time gates which destinations to enqueue.
        A separate range check fires at receive time (in deliver()) so a
        receiver that has moved out of range will not actually receive.
        Loss is rolled at receive time too.
        """
        if not alive[sender_id]:
            return
        # `origin` is the ORIGINAL sender (the drone whose Map / Vote /
        # Command / OhShit this carries). When `origin` is None,
        # this is a fresh broadcast from sender_id itself. When set, this
        # is a multi-hop relay: sender_id is forwarding info originally
        # sourced from `origin`. `starting_hop` lets relayers increment
        # the hop count (1 = original sender, 2+ = forwarded).
        effective_origin = sender_id if origin is None else origin
        dvec = positions - sender_pos[None, :]
        dists = np.linalg.norm(dvec, axis=1)
        for j in range(self.n):
            if j == sender_id:
                continue
            if origin is not None and j == effective_origin:
                continue
            if not alive[j]:
                continue
            d_send = float(dists[j])
            if d_send > self.comms_range_m:
                continue
            packet_key = self._packet_receiver_key(j, effective_origin, payload)
            if packet_key in self._seen_packet_receivers:
                continue
            self._seen_packet_receivers.add(packet_key)
            delay_ticks = max(1, int(math.ceil(d_send / self.sound_speed)))
            deliver_tick = current_tick + delay_ticks
            msg = Message(
                origin=effective_origin,
                sender=sender_id,
                sent_tick=current_tick,
                deliver_tick=deliver_tick,
                receiver=j,
                sender_pos_at_send=sender_pos.copy(),
                payload=payload,
                sig_stub=sig_stub,
                hop=starting_hop,
                range_at_send=d_send,
                signal_strength_at_send=self.signal_strength(d_send),
            )
            self._inflight.setdefault(deliver_tick, []).append(msg)
            self.n_sent += 1
            if self.log_events:
                self.events.append(
                    CommsEvent(
                        tick=current_tick,
                        drone_id=sender_id,
                        kind="SEND",
                        info={
                            "to": j,
                            "deliver_tick": deliver_tick,
                            "d": d_send,
                            "strength": msg.signal_strength_at_send,
                        },
                    )
                )

    def deliver(
        self,
        current_tick: int,
        positions: np.ndarray,
        alive: np.ndarray,
        rng: np.random.Generator,
    ) -> dict[int, list[Message]]:
        """Return per-drone inbox of messages arriving this tick.

        Each message survives only if:
          - receiver is alive
          - sender is alive at delivery (we don't model speed-of-sound
            persistence after sender death — corpses don't transmit)
          - receiver is still within comms_range_m of the sender position
            *now* (not at send time)
          - loss roll fails (uniform iid per-message)
        """
        arrivals = self._inflight.pop(current_tick, [])
        inbox: dict[int, list[Message]] = {i: [] for i in range(self.n)}
        for msg in arrivals:
            # Sender alive at delivery?
            if not alive[msg.origin]:
                self.n_dropped_dead += 1
                if self.log_events:
                    self.events.append(
                        CommsEvent(
                            tick=current_tick,
                            drone_id=msg.receiver,
                            kind="DROP_DEAD",
                            info={"from": msg.origin},
                        )
                    )
                continue
            # Receiver alive?
            if not alive[msg.receiver]:
                continue
            # Receive-time range check — receiver may have moved.
            d_recv = float(np.linalg.norm(positions[msg.receiver] - positions[msg.origin]))
            rx_strength = self.signal_strength(d_recv)
            if d_recv > self.comms_range_m:
                self.n_dropped_range += 1
                if self.log_events:
                    self.events.append(
                        CommsEvent(
                            tick=current_tick,
                            drone_id=msg.receiver,
                            kind="DROP_RANGE",
                            info={
                                "from": msg.origin,
                                "d_send": msg.range_at_send,
                                "d_recv": d_recv,
                                "strength": rx_strength,
                            },
                        )
                    )
                continue
            # Loss roll.
            if self.loss_rate > 0 and rng.random() < self.loss_rate:
                self.n_dropped_loss += 1
                if self.log_events:
                    self.events.append(
                        CommsEvent(
                            tick=current_tick,
                            drone_id=msg.receiver,
                            kind="DROP_LOSS",
                            info={"from": msg.origin},
                        )
                    )
                continue
            msg.signal_strength_at_receive = rx_strength
            inbox[msg.receiver].append(msg)
            self.n_delivered += 1
            if self.log_events:
                self.events.append(
                    CommsEvent(
                        tick=current_tick,
                        drone_id=msg.receiver,
                        kind="RECV",
                        info={
                            "from": msg.origin,
                            "d_recv": d_recv,
                            "hop": msg.hop,
                            "strength": rx_strength,
                        },
                    )
                )
        return inbox

    def agent_step_order(self, rng: np.random.Generator) -> np.ndarray:
        """Per-tick random drone-processing order. Caller calls each
        drone's step in this order. This is the antidote to the
        'synchronous broadcast in disguise' failure mode — different
        drones process their inboxes (and act) in different orders each
        tick, so a drone that acts early in the tick uses stale info
        relative to drones that act later.
        """
        return rng.permutation(self.n)

    def summary(self) -> dict:
        return {
            "n_sent": self.n_sent,
            "n_delivered": self.n_delivered,
            "n_dropped_loss": self.n_dropped_loss,
            "n_dropped_range": self.n_dropped_range,
            "n_dropped_dead": self.n_dropped_dead,
            "delivery_rate": (
                self.n_delivered / max(1, self.n_sent)
            ),
        }


# ---------------------------------------------------------------------------
# Falsifiability tests: prove the substrate can surface real failures.
# These tests would FAIL if the substrate were a synchronous broadcast.
# ---------------------------------------------------------------------------


def _falsifiability_tests() -> int:
    """Returns number of failed tests. 0 = pass."""
    failed = 0
    rng = np.random.default_rng(42)

    # Test 1: messages exceeding range at receive time are dropped.
    n = 3
    positions = np.array([[0.0, 0, 0], [50.0, 0, 0], [200.0, 0, 0]])
    alive = np.ones(n, dtype=bool)
    comms = LocalComms(n=n, comms_range_m=100.0, sound_speed=150.0, loss_rate=0.0)
    comms.send(0, positions[0], "hello", b"\x00" * 16, positions, alive, current_tick=0, rng=rng)
    # Drone 1 is in range at d=50 → should receive. Drone 2 is at d=200 → not even queued.
    inbox = comms.deliver(current_tick=1, positions=positions, alive=alive, rng=rng)
    if len(inbox[1]) != 1:
        print(f"FAIL test1a: drone 1 received {len(inbox[1])} msgs, expected 1")
        failed += 1
    if len(inbox[2]) != 0:
        print(f"FAIL test1b: drone 2 received {len(inbox[2])} msgs, expected 0")
        failed += 1

    # Test 2: receiver moves out of range between send and receive — message dropped.
    n = 2
    positions = np.array([[0.0, 0, 0], [50.0, 0, 0]])
    alive = np.ones(n, dtype=bool)
    comms = LocalComms(n=n, comms_range_m=100.0, sound_speed=10.0, loss_rate=0.0)
    # Tick 0: send. With sound_speed=10 m/tick and d=50m, delivery is tick 5.
    comms.send(0, positions[0], "hello", b"\x00" * 16, positions, alive, current_tick=0, rng=rng)
    # Tick 5: receiver has moved out of range.
    positions[1] = np.array([200.0, 0, 0])
    inbox = comms.deliver(current_tick=5, positions=positions, alive=alive, rng=rng)
    if len(inbox[1]) != 0:
        print(f"FAIL test2: receiver moved out of range but got {len(inbox[1])} msgs")
        failed += 1
    if comms.n_dropped_range != 1:
        print(f"FAIL test2b: expected 1 range-drop, got {comms.n_dropped_range}")
        failed += 1

    # Test 3: propagation delay scales with distance.
    n = 3
    positions = np.array([[0.0, 0, 0], [10.0, 0, 0], [90.0, 0, 0]])
    alive = np.ones(n, dtype=bool)
    comms = LocalComms(n=n, comms_range_m=100.0, sound_speed=10.0, loss_rate=0.0)
    comms.send(0, positions[0], "hello", b"\x00" * 16, positions, alive, current_tick=0, rng=rng)
    # d=10 -> delay 1 (delivery tick 1). d=90 -> delay 9 (delivery tick 9).
    inbox_t1 = comms.deliver(1, positions, alive, rng)
    inbox_t9 = comms.deliver(9, positions, alive, rng)
    if len(inbox_t1[1]) != 1:
        print(f"FAIL test3a: near drone should receive at t=1, got {len(inbox_t1[1])}")
        failed += 1
    if len(inbox_t9[2]) != 1:
        print(f"FAIL test3b: far drone should receive at t=9, got {len(inbox_t9[2])}")
        failed += 1

    # Test 4: dead sender → message dropped at delivery.
    n = 2
    positions = np.array([[0.0, 0, 0], [50.0, 0, 0]])
    alive = np.ones(n, dtype=bool)
    comms = LocalComms(n=n, comms_range_m=100.0, sound_speed=10.0, loss_rate=0.0)
    comms.send(0, positions[0], "hello", b"\x00" * 16, positions, alive, current_tick=0, rng=rng)
    alive[0] = False
    inbox = comms.deliver(5, positions, alive, rng)
    if comms.n_dropped_dead != 1:
        print(f"FAIL test4: dead sender drop count {comms.n_dropped_dead} expected 1")
        failed += 1

    # Test 5: signal strength degrades monotonically with range.
    comms = LocalComms(n=3, comms_range_m=100.0)
    if not (comms.signal_strength(10.0) > comms.signal_strength(50.0) > comms.signal_strength(90.0)):
        print("FAIL test5: signal strength should decrease as range increases")
        failed += 1

    # Test 6: delivered messages carry receive-time signal strength.
    positions = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    alive = np.ones(2, dtype=bool)
    comms = LocalComms(n=2, comms_range_m=100.0, sound_speed=100.0, loss_rate=0.0)
    comms.send(0, positions[0], "hello", b"\x00" * 16, positions, alive, current_tick=0, rng=rng)
    inbox = comms.deliver(1, positions, alive, rng)
    if not inbox[1] or inbox[1][0].signal_strength_at_receive <= 0:
        print("FAIL test6: delivered message missing receive signal strength")
        failed += 1

    # Test 7: loss rate actually drops some messages.
    n = 2
    positions = np.array([[0.0, 0, 0], [50.0, 0, 0]])
    alive = np.ones(n, dtype=bool)
    comms = LocalComms(n=n, comms_range_m=100.0, sound_speed=10.0, loss_rate=0.5)
    rng2 = np.random.default_rng(42)
    n_trials = 1000
    for i in range(n_trials):
        comms.send(0, positions[0], f"hello-{i}", b"\x00" * 16, positions, alive, current_tick=0, rng=rng2)
    inbox = comms.deliver(5, positions, alive, rng2)
    dropped_fraction = comms.n_dropped_loss / n_trials
    if not (0.4 < dropped_fraction < 0.6):
        print(f"FAIL test7: loss fraction {dropped_fraction:.3f} not within [0.4, 0.6]")
        failed += 1

    # Test 8: shuffle order is actually random (different across calls).
    rng3 = np.random.default_rng(0)
    comms = LocalComms(n=10, comms_range_m=100.0)
    o1 = comms.agent_step_order(rng3)
    o2 = comms.agent_step_order(rng3)
    if list(o1) == list(o2):
        print("FAIL test8: agent_step_order returned same permutation twice")
        failed += 1

    return failed


if __name__ == "__main__":
    failed = _falsifiability_tests()
    if failed == 0:
        print("local_comms: all falsifiability tests passed")
    else:
        print(f"local_comms: {failed} tests failed")
