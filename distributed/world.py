# /// script
# dependencies = ["numpy<3"]
# ///
"""Simulation world: physics, comms, agents, failure injection.

This is the substrate. It owns the ground-truth positions (only for
physics and comms range computation), drives the per-tick loop with
random drone ordering, and exposes a per-tick callback for failure
injection. Agents see ONLY their inbox; the world is responsible for
NOT leaking ground truth into the agent decision pipeline.

The bench writes scenarios as World configurations + callback hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from agent import Agent, AgentStepLog
from local_comms import LocalComms, Message
from protocol import Command


@dataclass
class WorldConfig:
    n_drones: int
    comms_range_m: float
    sound_speed_m_per_tick: float = 150.0
    loss_rate: float = 0.0
    physics_dt: float = 1.0   # ticks are arbitrary units; world.tick = 1 step
    max_ticks: int = 1000
    log_events: bool = True


@dataclass
class WorldMetrics:
    # Time-series, sampled per tick or periodically.
    leader_consensus_frac: list[float] = field(default_factory=list)
    leader_modal_id: list[int] = field(default_factory=list)  # most-common leader id per tick
    leader_id_distribution: list[dict] = field(default_factory=list)  # per-tick {leader_id: count}
    formation_error_mean: list[float] = field(default_factory=list)
    formation_error_max: list[float] = field(default_factory=list)
    n_known_mean: list[float] = field(default_factory=list)
    n_collisions: list[int] = field(default_factory=list)  # drones <1m apart
    fraction_alive: list[float] = field(default_factory=list)
    # Coverage: fraction of manifold points with at least one drone within R.
    coverage_frac: list[float] = field(default_factory=list)
    # Coverage of the CURRENT elected manifold (may differ from
    # true_manifold after a leader-issued reformation).
    coverage_current_manifold: list[float] = field(default_factory=list)
    current_manifold_size: list[int] = field(default_factory=list)
    # Summary at end.
    final_summary: dict = field(default_factory=dict)


# Per-tick hook signature.
TickHook = Callable[["World", int], None]


class World:
    def __init__(self, cfg: WorldConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.n = cfg.n_drones
        self.positions = np.zeros((cfg.n_drones, 3))
        self.velocities = np.zeros((cfg.n_drones, 3))
        self.alive = np.ones(cfg.n_drones, dtype=bool)
        self.agents: list[Agent | None] = [None] * cfg.n_drones
        self.comms = LocalComms(
            n=cfg.n_drones,
            comms_range_m=cfg.comms_range_m,
            loss_rate=cfg.loss_rate,
            sound_speed=cfg.sound_speed_m_per_tick,
            log_events=cfg.log_events,
        )
        self.metrics = WorldMetrics()
        self.agent_logs: list[AgentStepLog] = []
        self.current_tick = 0
        self.tick_hooks: list[TickHook] = []
        # Optional ground-truth target manifold (the BENCH knows; agents
        # only know via the leader's command).
        self.true_manifold: np.ndarray | None = None

    def attach_agent(self, drone_id: int, agent: Agent) -> None:
        self.agents[drone_id] = agent
        self.positions[drone_id] = agent.position.copy()

    def add_tick_hook(self, hook: TickHook) -> None:
        self.tick_hooks.append(hook)

    def issue_global_command(self, manifold: np.ndarray, heading: np.ndarray) -> None:
        """Bench-side helper: write a single Command into every agent's
        protocol state as the initial mission. After this, the protocol
        is on its own — drones gossip the command around, leader may
        re-issue, etc. This is the analogue of an at-port pre-deployment
        briefing; the underwater swarm does not get free oracle updates
        afterward. Subsequent commands must propagate via gossip.

        Calling this AFTER tick 0 is a smell — the test should rely on
        gossip from the leader, not on the bench injecting commands.
        """
        if self.current_tick > 0:
            print(
                f"WARN: issue_global_command called after tick 0 "
                f"(tick={self.current_tick}). This is an oracle leak unless "
                f"you intend to model a fresh radio briefing."
            )
        self.true_manifold = manifold.copy()
        # The initial command's priority MUST match the highest drone
        # priority (not exceed it), otherwise the leader's subsequent
        # commands (priority = its drone_id <= N-1) can never override
        # the operator's initial command via the (priority, epoch) merge.
        # Setting equal means the leader wins by bumping the epoch.
        op_priority = max(0, self.n - 1)
        live_positions = self.positions[self.alive]
        rally_point = (
            np.mean(live_positions, axis=0)
            if len(live_positions) > 0
            else np.zeros(3, dtype=np.float64)
        )
        if len(live_positions) > 0 and len(manifold) > 0:
            nearest = np.linalg.norm(
                live_positions[:, None, :] - manifold[None, :, :],
                axis=-1,
            ).min(axis=1)
            max_leg_distance = float(np.max(nearest))
        else:
            max_leg_distance = 0.0
        expected_completion_tick = self.current_tick + int(np.ceil(max_leg_distance / 0.8)) + 20
        for i, agent in enumerate(self.agents):
            if agent is None:
                continue
            cmd = Command(
                origin=-1,
                leader_priority=op_priority,
                epoch=0,
                payload={
                    "manifold_targets": manifold.copy(),
                    "heading": heading.copy(),
                    "leg": 0,
                    "rally_point": rally_point.copy(),
                    "expected_completion_tick": expected_completion_tick,
                },
                sig_stub=b"\x00" * 16,
            )
            agent.proto.latest_command = cmd
            agent.proto.latest_command_tick = self.current_tick

    def step(self) -> None:
        # 1. Pre-step hooks (failures, byzantine swaps, obstacles, etc.).
        for hook in self.tick_hooks:
            hook(self, self.current_tick)

        # 2. Deliver inboxes for this tick. We materialize the inbox dict
        #    BEFORE per-drone action so all messages timed for this tick
        #    are "available" but agents still see only their own inbox.
        inbox_dict = self.comms.deliver(
            current_tick=self.current_tick,
            positions=self.positions,
            alive=self.alive,
            rng=self.rng,
        )

        # 3. Per-drone action in random order. The random order means
        #    a drone acting early in the tick uses pre-step state for
        #    its known_positions of late-acting drones; drones acting
        #    later only learn about early-acting drones after an actual
        #    event-triggered acoustic message is delivered. Within this
        #    tick no drone sees another's current-tick decision.
        order = self.comms.agent_step_order(self.rng)
        new_velocities = np.zeros_like(self.velocities)
        new_positions = self.positions.copy()
        outgoing_per_drone: list[tuple[int, list[Any]]] = []
        for did in order:
            if not self.alive[did]:
                continue
            agent = self.agents[did]
            if agent is None:
                continue
            inbox = inbox_dict.get(int(did), [])
            v, out_msgs, log = agent.step(
                current_tick=self.current_tick,
                inbox=inbox,
                physics_dt=self.cfg.physics_dt,
            )
            new_velocities[did] = v
            # Integrate position.
            new_positions[did] = self.positions[did] + v * self.cfg.physics_dt
            agent.position = new_positions[did]
            agent.velocity = v
            outgoing_per_drone.append((int(did), out_msgs))
            self.agent_logs.append(log)

        self.positions = new_positions
        self.velocities = new_velocities

        # 4. Send outgoing messages NOW (queue for delivery at
        #    sent_tick + propagation_delay).
        #
        #    Each entry is either a bare payload (own broadcast) or a
        #    (payload, next_hop, origin) tuple (multi-hop forward).
        for did, out_msgs in outgoing_per_drone:
            for item in out_msgs:
                if isinstance(item, tuple) and len(item) == 3:
                    payload, next_hop, origin = item
                else:
                    payload, next_hop, origin = item, 1, None
                self.comms.send(
                    sender_id=did,
                    sender_pos=self.positions[did],
                    payload=payload,
                    sig_stub=getattr(payload, "sig_stub", b""),
                    positions=self.positions,
                    alive=self.alive,
                    current_tick=self.current_tick,
                    rng=self.rng,
                    origin=origin,
                    starting_hop=next_hop,
                )

        # 5. Collect metrics for this tick.
        self._record_metrics()

        self.current_tick += 1

    def _record_metrics(self) -> None:
        # Leader-consensus: track modal leader-id and full distribution.
        from collections import Counter
        from protocol import infer_leader
        if self.alive.any():
            leaders = []
            for i, agent in enumerate(self.agents):
                if agent is None or not self.alive[i]:
                    continue
                leaders.append(infer_leader(agent.proto, self.current_tick))
            if leaders:
                c = Counter(leaders)
                top_id, top_count = c.most_common(1)[0]
                self.metrics.leader_consensus_frac.append(top_count / len(leaders))
                self.metrics.leader_modal_id.append(int(top_id))
                self.metrics.leader_id_distribution.append(dict(c))
            else:
                self.metrics.leader_consensus_frac.append(0.0)
                self.metrics.leader_modal_id.append(-1)
                self.metrics.leader_id_distribution.append({})
        else:
            self.metrics.leader_consensus_frac.append(0.0)
            self.metrics.leader_modal_id.append(-1)
            self.metrics.leader_id_distribution.append({})

        # Formation error vs ground-truth manifold (BENCH'S view; this is
        # the audience metric — what an outside observer measures, NOT
        # what drones perceive). Each alive drone is matched to its nearest
        # ground-truth target; report mean and max distance.
        if self.true_manifold is not None and self.alive.any():
            live_idx = np.where(self.alive)[0]
            live_pos = self.positions[live_idx]
            # Greedy nearest assignment (each drone -> nearest unclaimed leaf).
            mt = self.true_manifold
            errors = []
            claimed = set()
            order_by_dist = np.argsort(np.linalg.norm(
                live_pos[:, None, :] - mt[None, :, :], axis=-1
            ).min(axis=1))
            for k in order_by_dist:
                d = np.linalg.norm(live_pos[k] - mt, axis=1)
                # Pick nearest unclaimed.
                idx_sorted = np.argsort(d)
                for cand in idx_sorted:
                    if int(cand) not in claimed:
                        claimed.add(int(cand))
                        errors.append(float(d[cand]))
                        break
            if errors:
                self.metrics.formation_error_mean.append(float(np.mean(errors)))
                self.metrics.formation_error_max.append(float(np.max(errors)))
            else:
                self.metrics.formation_error_mean.append(0.0)
                self.metrics.formation_error_max.append(0.0)
        else:
            self.metrics.formation_error_mean.append(0.0)
            self.metrics.formation_error_max.append(0.0)

        # Mean known-drones count across alive agents.
        n_knowns = []
        for i, agent in enumerate(self.agents):
            if agent is None or not self.alive[i]:
                continue
            n_knowns.append(len(agent.proto.known_positions))
        self.metrics.n_known_mean.append(float(np.mean(n_knowns)) if n_knowns else 0.0)

        # Collisions: drone pairs <1m apart.
        if self.alive.sum() >= 2:
            live_idx = np.where(self.alive)[0]
            lp = self.positions[live_idx]
            d = np.linalg.norm(lp[:, None, :] - lp[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            n_col = int((d < 1.0).sum() // 2)
            self.metrics.n_collisions.append(n_col)
        else:
            self.metrics.n_collisions.append(0)

        self.metrics.fraction_alive.append(float(self.alive.sum() / max(1, self.n)))

        # Coverage of the original (true) manifold.
        if self.true_manifold is not None and self.alive.any():
            live_pos = self.positions[self.alive]
            d_drone_to_target = np.linalg.norm(
                self.true_manifold[:, None, :] - live_pos[None, :, :], axis=-1
            ).min(axis=1)
            self.metrics.coverage_frac.append(
                float((d_drone_to_target < 3.0).sum() / len(self.true_manifold))
            )
        else:
            self.metrics.coverage_frac.append(0.0)

        # Coverage of the CURRENT elected manifold. Sample from the modal
        # agent's latest_command (the manifold the swarm currently believes
        # is the goal). After reformation this may be smaller than the
        # original; protocol's job is to fill it cleanly.
        elected_manifold = None
        for i in range(self.n):
            agent = self.agents[i]
            if agent is None or not self.alive[i]:
                continue
            cmd = agent.proto.latest_command
            if cmd is not None and cmd.payload.get("manifold_targets") is not None:
                elected_manifold = cmd.payload["manifold_targets"]
                break
        if elected_manifold is not None and self.alive.any():
            live_pos = self.positions[self.alive]
            d2 = np.linalg.norm(
                elected_manifold[:, None, :] - live_pos[None, :, :], axis=-1
            ).min(axis=1)
            self.metrics.coverage_current_manifold.append(
                float((d2 < 3.0).sum() / len(elected_manifold))
            )
            self.metrics.current_manifold_size.append(int(len(elected_manifold)))
        else:
            self.metrics.coverage_current_manifold.append(0.0)
            self.metrics.current_manifold_size.append(0)

    def run(self) -> None:
        while self.current_tick < self.cfg.max_ticks:
            self.step()
        self.metrics.final_summary = {
            "n_drones": self.n,
            "ticks": self.current_tick,
            "comms": self.comms.summary(),
            "final_alive": int(self.alive.sum()),
            "final_leader_consensus": self.metrics.leader_consensus_frac[-1]
                if self.metrics.leader_consensus_frac else 0.0,
            "final_formation_err_mean": self.metrics.formation_error_mean[-1]
                if self.metrics.formation_error_mean else 0.0,
            "final_formation_err_max": self.metrics.formation_error_max[-1]
                if self.metrics.formation_error_max else 0.0,
            "final_n_known_mean": self.metrics.n_known_mean[-1]
                if self.metrics.n_known_mean else 0.0,
            "max_collisions_observed": max(self.metrics.n_collisions, default=0),
        }


# ---------------------------------------------------------------------------
# Smoke test.
# ---------------------------------------------------------------------------

def _smoke() -> int:
    failed = 0
    # 5 drones, equally spaced. Initial command with a 5-leaf manifold.
    rng = np.random.default_rng(7)
    cfg = WorldConfig(
        n_drones=5,
        comms_range_m=30.0,
        sound_speed_m_per_tick=150.0,
        loss_rate=0.0,
        max_ticks=100,
        log_events=False,
    )
    w = World(cfg, seed=7)
    starts = rng.normal(size=(5, 3)) * 10
    for i in range(5):
        a = Agent(drone_id=i, priority=i, position=starts[i].copy())
        w.attach_agent(i, a)

    manifold = np.array([
        [0.0, 0, 0], [5.0, 0, 0], [-5.0, 0, 0], [0.0, 5.0, 0], [0.0, -5.0, 0]
    ])
    w.issue_global_command(manifold, heading=np.array([1.0, 0, 0]))

    w.run()
    summary = w.metrics.final_summary

    # Falsifiability: a working swarm with no failures should converge
    # to high leader consensus and low formation error.
    if summary["final_leader_consensus"] < 0.99:
        print(f"FAIL smoke: leader consensus {summary['final_leader_consensus']} < 0.99")
        failed += 1
    # The five drones with priorities 0..4 should converge on drone 4 as leader
    # (highest priority) -- but the initial command from "operator" has higher
    # priority (10_000) so latest_command stays the operator's. Leader inference
    # uses known_priorities, which only contains drone-origin Vote responses;
    # the operator never sent a Vote. So among drones, 4 should win.

    # Formation error: should be low (drones near their targets).
    if summary["final_formation_err_mean"] > 2.0:
        print(f"FAIL smoke: mean formation error {summary['final_formation_err_mean']:.3f}m too high")
        failed += 1
    print(f"smoke: final_leader_consensus={summary['final_leader_consensus']:.3f} "
          f"final_form_err_mean={summary['final_formation_err_mean']:.3f}m "
          f"final_alive={summary['final_alive']}/{cfg.n_drones} "
          f"comms_delivery={summary['comms']['delivery_rate']:.3f}")
    return failed


if __name__ == "__main__":
    n = _smoke()
    print("world: smoke test passed" if n == 0 else f"world: {n} failures")
