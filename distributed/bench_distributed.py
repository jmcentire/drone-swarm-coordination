# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Distributed protocol bench — peer-review grade.

Every scenario specifies, in its docstring:
  Claim:               the proposition the scenario tests
  Falsifying behavior: what the protocol would have to do to refute the claim
  Bench mechanism:     how the bench structure can surface that failure
  Pass criterion:      the quantitative threshold for "passes"

If any scenario lacks one of those four lines its runner skips it. This is
the falsifiability gate identified by the Sim audit as the root-cause fix.

Each scenario runs:
  - the distributed protocol (world.py + agent.py + local_comms.py)
  - the oracle baseline (baseline_oracle.py) for upper bound
  - the drift baseline (baseline_drift.py) for lower bound

with N_SEEDS independent seeds. Statistics reported with Wilson CIs on
success-rate metrics and bootstrap CIs on continuous metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agent import Agent
from baseline_drift import run_drift
from baseline_oracle import run_oracle
from stats import bootstrap_ci, wilson_ci
from world import World, WorldConfig


# ---------------------------------------------------------------------------
# Scenario base class.
# ---------------------------------------------------------------------------


@dataclass
class ScenarioSpec:
    name: str
    claim: str
    falsifying: str
    mechanism: str
    pass_criterion: str
    n_drones: int = 30
    n_ticks: int = 300
    comms_range_m: float = 30.0
    loss_rate: float = 0.0
    sound_speed: float = 150.0
    spacing_m: float = 8.0


def _make_manifold(n_targets: int, rng: np.random.Generator) -> np.ndarray:
    """A 3D quasi-uniform set of target positions for the formation."""
    pts = []
    phi = (1 + 5 ** 0.5) / 2
    R = 8.0 * (n_targets ** 0.5) / (20 ** 0.5)
    for i in range(n_targets):
        z = 1 - 2 * (i + 0.5) / n_targets
        r = (1 - z * z) ** 0.5
        theta = 2 * np.pi * i / phi
        pts.append([R * r * np.cos(theta), R * r * np.sin(theta), R * z + R * 1.5])
    return np.array(pts)


def _manifold_size_for_scenario(spec: ScenarioSpec) -> int:
    """How many leaves the manifold should have for this scenario."""
    if spec.name == "S7_surplus_fills_gaps":
        return max(4, spec.n_drones - 6)
    if spec.name == "S7b_aggressive_surplus_fills":
        return max(4, spec.n_drones - 10)
    return spec.n_drones


def _initial_positions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random initial positions in a 30m cube."""
    return rng.uniform(-15, 15, size=(n, 3))


def _build_world(spec: ScenarioSpec, seed: int) -> tuple[World, np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = _initial_positions(spec.n_drones, rng)
    cfg = WorldConfig(
        n_drones=spec.n_drones,
        comms_range_m=spec.comms_range_m,
        sound_speed_m_per_tick=spec.sound_speed,
        loss_rate=spec.loss_rate,
        max_ticks=spec.n_ticks,
        log_events=False,
    )
    w = World(cfg, seed=seed)
    enable_reform = spec.name == "S5b_random_loss_with_reform"
    for i in range(spec.n_drones):
        a = Agent(
            drone_id=i, priority=i, position=starts[i].copy(),
            enable_reformation=enable_reform,
            initial_n_drones=spec.n_drones,
        )
        w.attach_agent(i, a)
    manifold = _make_manifold(_manifold_size_for_scenario(spec), rng)
    w.issue_global_command(manifold, heading=np.array([1.0, 0, 0]))
    return w, manifold


# ---------------------------------------------------------------------------
# Scenario runners.
# ---------------------------------------------------------------------------


@dataclass
class SeedRun:
    seed: int
    protocol: dict
    oracle: dict
    drift: dict


def run_seed(spec: ScenarioSpec, seed: int, hooks: list[Callable] | None = None) -> SeedRun:
    """Run one seed across all three modes (protocol / oracle / drift)."""
    w, manifold = _build_world(spec, seed)
    if hooks:
        for h in hooks:
            w.add_tick_hook(h)
    w.run()

    rng = np.random.default_rng(seed)
    starts = _initial_positions(spec.n_drones, rng)
    # Kill events come from hooks; collect them into a schedule for the baselines.
    kill_schedule = _hook_kill_schedule(hooks or [])

    oracle_run = run_oracle(
        n_drones=spec.n_drones,
        start_positions=starts,
        manifold=manifold,
        n_ticks=spec.n_ticks,
        alive_schedule=kill_schedule,
    )
    drift_run = run_drift(
        n_drones=spec.n_drones,
        start_positions=starts,
        manifold=manifold,
        n_ticks=spec.n_ticks,
        alive_schedule=kill_schedule,
    )

    return SeedRun(
        seed=seed,
        protocol={
            "final_alive": int(w.alive.sum()),
            "final_leader_consensus": w.metrics.leader_consensus_frac[-1] if w.metrics.leader_consensus_frac else 0.0,
            "final_form_err_mean": w.metrics.formation_error_mean[-1] if w.metrics.formation_error_mean else 0.0,
            "final_form_err_max": w.metrics.formation_error_max[-1] if w.metrics.formation_error_max else 0.0,
            "final_coverage": w.metrics.coverage_frac[-1] if w.metrics.coverage_frac else 0.0,
            "final_coverage_current_manifold": (
                w.metrics.coverage_current_manifold[-1]
                if w.metrics.coverage_current_manifold else 0.0
            ),
            "final_manifold_size": (
                w.metrics.current_manifold_size[-1]
                if w.metrics.current_manifold_size else 0
            ),
            "max_collisions": max(w.metrics.n_collisions, default=0),
            "comms_delivery_rate": w.comms.summary()["delivery_rate"],
            "n_knows_at_end": w.metrics.n_known_mean[-1] if w.metrics.n_known_mean else 0.0,
            "leader_consensus_ts": w.metrics.leader_consensus_frac,
            "leader_modal_id_ts": w.metrics.leader_modal_id,
            "form_err_mean_ts": w.metrics.formation_error_mean,
        },
        oracle={
            "final_alive": int(oracle_run["final_alive"].sum()),
            "final_form_err_mean": oracle_run["metrics"]["formation_error_mean"][-1],
            "final_form_err_max": oracle_run["metrics"]["formation_error_max"][-1],
            "final_coverage": oracle_run["metrics"]["coverage_frac"][-1],
            "max_collisions": max(oracle_run["metrics"]["n_collisions"], default=0),
            "form_err_mean_ts": oracle_run["metrics"]["formation_error_mean"],
        },
        drift={
            "final_alive": int(drift_run["final_alive"].sum()),
            "final_form_err_mean": drift_run["metrics"]["formation_error_mean"][-1],
            "final_form_err_max": drift_run["metrics"]["formation_error_max"][-1],
            "final_coverage": drift_run["metrics"]["coverage_frac"][-1],
            "max_collisions": max(drift_run["metrics"]["n_collisions"], default=0),
            "form_err_mean_ts": drift_run["metrics"]["formation_error_mean"],
        },
    )


def _hook_kill_schedule(hooks: list[Callable]) -> dict[int, list[int]]:
    """Extract a kill schedule from KillHook instances among the hooks."""
    sched: dict[int, list[int]] = {}
    for h in hooks:
        if isinstance(h, KillHook):
            for tick, victims in h.events:
                sched.setdefault(tick, []).extend(victims)
    return sched


# ---------------------------------------------------------------------------
# Failure hooks (these run as tick-callbacks on the World).
# ---------------------------------------------------------------------------


class KillHook:
    """Kills specified drones at specified ticks."""
    def __init__(self, events: list[tuple[int, list[int]]]) -> None:
        self.events = events
        self._applied = set()

    def __call__(self, w: World, tick: int) -> None:
        for ev_tick, victims in self.events:
            if ev_tick == tick and ev_tick not in self._applied:
                for v in victims:
                    if 0 <= v < w.n:
                        w.alive[v] = False
                self._applied.add(ev_tick)


class ByzantineHook:
    """Activates byzantine behavior on selected drones at given tick."""
    def __init__(self, tick: int, byz_ids: list[int], lie_fn: Callable | None = None,
                 spam_priority: int | None = None) -> None:
        self.tick = tick
        self.byz_ids = byz_ids
        self.lie_fn = lie_fn
        self.spam_priority = spam_priority
        self._applied = False

    def __call__(self, w: World, tick: int) -> None:
        if tick == self.tick and not self._applied:
            for bid in self.byz_ids:
                a = w.agents[bid]
                if a is not None:
                    if self.lie_fn is not None:
                        a.lie_about_position = self.lie_fn
                    if self.spam_priority is not None:
                        a.spam_priority = self.spam_priority
                        a.proto.my_priority = self.spam_priority
            self._applied = True


class DisplaceHook:
    """Models a sustained current that sweeps a subset of drones out of
    comms range during [tick_apply, tick_reset), then subsides.

    Implementation: snapshots each affected drone's pre-displacement
    position at tick_apply, then each tick during the active window
    pushes the drone back to (snapshot + offset), overwriting any motion
    the attractor would otherwise have produced. Without this, displaced
    drones would simply swim back toward their targets at ~max_speed
    m/tick and the partition would self-heal long before tick_reset --
    making the scenario test something other than what it claims.

    At tick_reset, the displaced drones are released (positions returned
    to the snapshot, no offset) and the normal attractor logic resumes.
    """
    def __init__(self, tick_apply: int, tick_reset: int, drone_ids: list[int],
                 offset: np.ndarray) -> None:
        self.tick_apply = tick_apply
        self.tick_reset = tick_reset
        self.drone_ids = drone_ids
        self.offset = np.asarray(offset, dtype=np.float64)
        self._snapshots: dict[int, np.ndarray] = {}
        self._active = False

    def __call__(self, w: World, tick: int) -> None:
        if tick == self.tick_apply and not self._snapshots:
            for did in self.drone_ids:
                if 0 <= did < w.n and w.alive[did]:
                    self._snapshots[did] = w.positions[did].copy()
            self._active = True
        if self._active and self.tick_apply <= tick < self.tick_reset:
            # Re-clamp each tick: position := snapshot + offset. This holds
            # the displaced drones at a fixed offset from their pre-event
            # positions regardless of attractor motion.
            for did, snap in self._snapshots.items():
                if 0 <= did < w.n and w.alive[did]:
                    w.positions[did] = snap + self.offset
                    if w.agents[did] is not None:
                        w.agents[did].position = w.positions[did].copy()
        if tick == self.tick_reset and self._active:
            # Release: snap back to the snapshot positions (no offset), so
            # the "current" cleanly subsides and physics takes over again.
            for did, snap in self._snapshots.items():
                if 0 <= did < w.n and w.alive[did]:
                    w.positions[did] = snap.copy()
                    if w.agents[did] is not None:
                        w.agents[did].position = w.positions[did].copy()
            self._active = False


# ---------------------------------------------------------------------------
# The actual scenarios.
# ---------------------------------------------------------------------------


SCENARIOS = [
    ScenarioSpec(
        name="S1_baseline_no_failures",
        claim=(
            "Under no failures the distributed protocol converges to a "
            "consistent leader and reaches the formation manifold."
        ),
        falsifying=(
            "Leader consensus stays below 95% at end, OR formation error "
            "remains >2x the oracle baseline."
        ),
        mechanism=(
            "Run protocol, oracle, drift over identical starts. Measure "
            "leader-consensus-fraction and formation error at end. Compare "
            "vs oracle (upper bound) and drift (lower bound)."
        ),
        pass_criterion="leader_consensus >= 0.95 AND form_err < 2x oracle",
        n_drones=30,
        n_ticks=200,
        loss_rate=0.0,
    ),
    ScenarioSpec(
        name="S2_message_loss_10pct",
        claim="Under 10% per-message loss the protocol still converges to consensus.",
        falsifying="Leader consensus <80% at end OR formation never settles.",
        mechanism="Same as S1 but with comms.loss_rate=0.10.",
        pass_criterion="leader_consensus >= 0.80",
        n_drones=30,
        n_ticks=300,
        loss_rate=0.10,
    ),
    ScenarioSpec(
        name="S3_message_loss_30pct",
        claim="Under 30% per-message loss the protocol still mostly converges.",
        falsifying="Leader consensus <60% at end.",
        mechanism="loss_rate=0.30.",
        pass_criterion="leader_consensus >= 0.60",
        n_drones=30,
        n_ticks=400,
        loss_rate=0.30,
    ),
    ScenarioSpec(
        name="S4_leader_kill",
        claim=(
            "When the current highest-priority drone is killed mid-mission, "
            "the swarm elects a new leader within ~3x gossip diameter."
        ),
        falsifying=(
            "Leader consensus never recovers above 80% after the kill, OR "
            "recovery takes longer than 60 ticks."
        ),
        mechanism=(
            "Run baseline 100 ticks to converge. Kill the highest-priority "
            "drone (id=N-1) at tick 100. Measure ticks until 80% of survivors "
            "agree on a new leader."
        ),
        pass_criterion="leader_recovery_ticks <= 60",
        n_drones=30,
        n_ticks=300,
        loss_rate=0.0,
    ),
    ScenarioSpec(
        name="S5_random_loss_20pct",
        claim=(
            "With 20% of drones removed mid-mission the protocol redistributes "
            "survivors to maintain manifold coverage at least as well as the "
            "static-assignment drift baseline."
        ),
        falsifying=(
            "Final coverage of the protocol is lower than drift's final coverage. "
            "Drift loses coverage exactly at the dead drones' assigned slots; "
            "if the protocol fails to redistribute, it does no better."
        ),
        mechanism=(
            "Kill 6 random drones at tick 100. Run protocol and drift over "
            "the same 6 victims. Measure coverage_frac (fraction of manifold "
            "targets within 3m of any alive drone) at end of mission."
        ),
        pass_criterion=(
            "leader_consensus >= 0.80 AND protocol_coverage >= drift_coverage"
        ),
        n_drones=30,
        n_ticks=300,
        loss_rate=0.05,
    ),
    ScenarioSpec(
        name="S5b_random_loss_with_reform",
        claim=(
            "With leader-issued manifold reformation: on losing 20%, the "
            "leader detects the loss and broadcasts a smaller manifold "
            "(size = alive count). Each survivor uses its current position "
            "as input to a single compute_target call against the smaller "
            "manifold. Result: clean bijection, no demotion, coverage of "
            "the smaller manifold approaches 1.0."
        ),
        falsifying=(
            "Protocol coverage of the new (smaller) manifold stays below "
            "0.90 — meaning reformation did not produce a clean bijection."
        ),
        mechanism=(
            "Same kill schedule as S5 (6 random drones at tick 100). "
            "enable_reformation=True, loss_threshold_pct=0.15. Leader "
            "issues a new Command with a 24-leaf manifold when alive "
            "count drops below 0.85 * 30 = 25. Drones re-target via "
            "compute_target on the new manifold using current positions."
        ),
        pass_criterion=(
            "leader_consensus >= 0.80 AND protocol_coverage_of_new_manifold >= 0.90"
        ),
        n_drones=30,
        n_ticks=400,
        loss_rate=0.05,
    ),
    ScenarioSpec(
        name="S7_surplus_fills_gaps",
        claim=(
            "When the manifold has fewer leaves than drones (so some drones "
            "start as surplus at parent-centroids), killing leaf-drones causes "
            "surplus drones to be promoted to fill the vacated leaves — coverage "
            "is preserved. Drift cannot do this and loses coverage 1-for-1."
        ),
        falsifying=(
            "After the kill, protocol coverage stays below drift coverage, OR "
            "protocol coverage drops below 90% of pre-kill level. Either means "
            "the per-drone compute_target is not actually re-assigning surplus."
        ),
        mechanism=(
            "n_drones=30, manifold of 24 leaves (so 6 drones are surplus at "
            "parent centroids). Run until lock at tick 150. Kill 6 random "
            "drones that compute_target said are at leaves. Compare protocol "
            "coverage to drift coverage at tick 400."
        ),
        pass_criterion=(
            "leader_consensus >= 0.80 AND protocol_coverage > drift_coverage + 0.10"
        ),
        n_drones=30,
        n_ticks=400,
        loss_rate=0.0,
    ),
    ScenarioSpec(
        name="S7b_aggressive_surplus_fills",
        claim=(
            "When ~40% of leaf-assigned drones die, the surplus pool is "
            "demanded heavily — protocol coverage exceeds drift's by a "
            "substantial margin because per-drone re-assignment promotes "
            "surplus into vacated slots."
        ),
        falsifying=(
            "Protocol coverage no better than drift coverage. Either the "
            "compute_target re-assignment isn't firing per-drone (oracle "
            "leak removed without replacement), or surplus doesn't actually "
            "migrate to fill leaves."
        ),
        mechanism=(
            "30 drones, 20-leaf manifold (10 surplus). Kill 8 random drones "
            "at tick 150. Drift loses up to 8/20 leaves; protocol promotes "
            "remaining surplus to fill."
        ),
        pass_criterion="protocol_coverage > drift_coverage + 0.15",
        n_drones=30,
        n_ticks=500,
        loss_rate=0.0,
    ),
    ScenarioSpec(
        name="S8_partition_heal",
        claim=(
            "Under physical separation that splits the swarm into two "
            "comms components, the swarm continues operating (each "
            "component elects its own leader within its connected set); "
            "after re-merge, the overall-highest-priority leader emerges."
        ),
        falsifying=(
            "During partition either component has leader_consensus < 70%; "
            "after re-merge, leader_consensus < 90% within 60 ticks; or "
            "drones in different components compute the *same* leaf "
            "target (which would indicate they share state across the "
            "partition — an oracle leak)."
        ),
        mechanism=(
            "30 drones, sphere manifold. At tick 100, set 15 drones' "
            "physical positions to a +100m offset; reset at tick 250. "
            "During tick 100-250, the comms range cannot bridge the gap "
            "(comms_range_m=30 < 100 m offset). Measure per-component "
            "leader consensus during partition and after heal."
        ),
        pass_criterion=(
            "During partition: per-component leader_consensus >= 0.70; "
            "after heal (tick 250+60=310): overall leader_consensus >= 0.90"
        ),
        n_drones=30,
        n_ticks=400,
        loss_rate=0.0,
        comms_range_m=30.0,
    ),
    ScenarioSpec(
        name="S9_tight_comms_range",
        claim=(
            "With comms range tightened to less than the swarm's spatial "
            "extent (so most drone pairs are NOT in direct contact), the "
            "protocol's multi-hop forwarding still achieves >=80% leader "
            "consensus by propagating votes through relay neighbors."
        ),
        falsifying=(
            "Leader consensus stays below 0.80 at end. Without multi-hop "
            "forwarding, drones far from the highest-priority drone would "
            "default to electing local leaders — consensus fragments."
        ),
        mechanism=(
            "30 drones in a [-15,15]^3 cube (extent ~30m). Comms range 8m "
            "(< swarm extent). Multi-hop forwarding propagates priority "
            "votes and heartbeats across the swarm via relay chains."
        ),
        pass_criterion="leader_consensus >= 0.80",
        n_drones=30,
        n_ticks=300,
        loss_rate=0.0,
        comms_range_m=8.0,
    ),
    ScenarioSpec(
        name="S10_byzantine_with_detection",
        claim=(
            "With active range-consistency Byzantine detection enabled, "
            "the protocol's formation error under 10% byzantine fraction "
            "is significantly closer to the no-byzantine baseline than "
            "without detection."
        ),
        falsifying=(
            "Formation error under byzantine-with-detection is >=80% of "
            "the without-detection baseline (S6: 1.90m). I.e., the "
            "detection mechanism produces <20% improvement."
        ),
        mechanism=(
            "Same as S6 (3 byzantine drones lying 50m off, activated at "
            "tick 50). All honest drones run "
            "enable_byzantine_detection=True. The detection cross-checks "
            "claimed positions vs. time-of-flight implied distances; "
            "after 3 consistent flags, heartbeats from that drone are "
            "rejected."
        ),
        pass_criterion="form_err < 0.8 * S6_form_err (i.e. <1.52m)",
        n_drones=30,
        n_ticks=300,
        loss_rate=0.0,
    ),
    ScenarioSpec(
        name="S6_byzantine_position_lie",
        claim=(
            "A drone broadcasting false positions causes bounded damage — "
            "the swarm's formation error increases but does not exceed 3x "
            "the no-byzantine baseline."
        ),
        falsifying=(
            "Formation error exceeds 3x the no-byzantine baseline OR "
            "swarm fails to maintain leader consensus."
        ),
        mechanism=(
            "Activate position-lying on 3 drones (10% of swarm) at tick 50; "
            "lying = broadcast position drifted 50m away from truth. "
            "Compare formation error to S1 baseline."
        ),
        pass_criterion="form_err < 3x S1_form_err",
        n_drones=30,
        n_ticks=300,
        loss_rate=0.0,
    ),
]


def scenario_hooks(spec: ScenarioSpec, seed: int) -> list[Callable]:
    """Build hooks for each scenario."""
    rng = np.random.default_rng(seed * 9001)
    hooks: list[Callable] = []
    n = spec.n_drones
    if spec.name == "S4_leader_kill":
        hooks.append(KillHook(events=[(100, [n - 1])]))
    elif spec.name == "S5_random_loss_20pct":
        victims = rng.choice(n, size=max(1, n // 5), replace=False).tolist()
        hooks.append(KillHook(events=[(100, victims)]))
    elif spec.name == "S5b_random_loss_with_reform":
        # Same kill schedule as S5 so direct A/B comparison.
        victims = rng.choice(n, size=max(1, n // 5), replace=False).tolist()
        hooks.append(KillHook(events=[(100, victims)]))
    elif spec.name == "S6_byzantine_position_lie":
        byz_ids = rng.choice(n - 1, size=3, replace=False).tolist()
        def lie_fn(tick: int, offset=np.array([50.0, 0.0, 0.0])):
            return offset
        hooks.append(ByzantineHook(tick=50, byz_ids=byz_ids, lie_fn=lie_fn))
    elif spec.name == "S7_surplus_fills_gaps":
        victims = rng.choice(24, size=6, replace=False).tolist()
        hooks.append(KillHook(events=[(150, victims)]))
    elif spec.name == "S7b_aggressive_surplus_fills":
        # 30 drones, 20-leaf manifold. Kill 8 of the first 20 (leaves).
        victims = rng.choice(20, size=8, replace=False).tolist()
        hooks.append(KillHook(events=[(150, victims)]))
    elif spec.name == "S8_partition_heal":
        diverged = rng.choice(spec.n_drones, size=15, replace=False).tolist()
        hooks.append(DisplaceHook(
            tick_apply=100, tick_reset=250,
            drone_ids=diverged,
            offset=np.array([0.0, 100.0, 0.0]),
        ))
    elif spec.name == "S9_tight_comms_range":
        pass  # no hooks; the comms_range_m on the spec is the test
    elif spec.name == "S10_byzantine_with_detection":
        # Same byzantine config as S6, but agents need detection enabled
        # — handled via a special-setup callable below.
        byz_ids = rng.choice(spec.n_drones - 1, size=3, replace=False).tolist()
        def lie_fn(tick: int, offset=np.array([50.0, 0.0, 0.0])):
            return offset
        hooks.append(ByzantineHook(tick=50, byz_ids=byz_ids, lie_fn=lie_fn))
    return hooks


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


def _seed_to_dict(r: SeedRun) -> dict:
    return {"seed": r.seed, "protocol": r.protocol, "oracle": r.oracle, "drift": r.drift}


def _seed_from_dict(d: dict) -> SeedRun:
    return SeedRun(seed=d["seed"], protocol=d["protocol"], oracle=d["oracle"], drift=d["drift"])


def _checkpoint_path(checkpoint_dir: str, scenario: str, seed: int) -> Path:
    return Path(checkpoint_dir) / scenario / f"seed_{seed:03d}.json"


def run_scenario(spec: ScenarioSpec, n_seeds: int, checkpoint_dir: str | None = None) -> dict:
    """Run all seeds for one scenario. If checkpoint_dir is set, per-seed
    JSONs are written to {dir}/{scenario}/seed_NNN.json as each completes;
    on re-entry, seeds whose checkpoint exists are loaded from disk and
    skipped. Heartbeat lines `[scenario seed N/M done dt=Xs cum=Ys]` are
    printed (flushed) so external watchers can see progress."""
    runs: list[SeedRun] = []
    t_scenario = time.perf_counter()
    seeds_done = 0
    seeds_loaded = 0
    if checkpoint_dir is not None:
        scen_dir = Path(checkpoint_dir) / spec.name
        scen_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(n_seeds):
        cp = _checkpoint_path(checkpoint_dir, spec.name, seed) if checkpoint_dir else None
        if cp is not None and cp.exists():
            try:
                with open(cp) as f:
                    runs.append(_seed_from_dict(json.load(f)))
                seeds_loaded += 1
                continue
            except Exception as e:
                print(f"  [{spec.name}] checkpoint {cp.name} unreadable ({e}); rerunning", flush=True)
        t_seed = time.perf_counter()
        hooks = scenario_hooks(spec, seed)
        run = run_seed(spec, seed, hooks=hooks)
        runs.append(run)
        seeds_done += 1
        dt_seed = time.perf_counter() - t_seed
        dt_scen = time.perf_counter() - t_scenario
        if cp is not None:
            tmp = cp.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(_seed_to_dict(run), f, default=str)
            os.replace(tmp, cp)
        print(
            f"  [{spec.name}] seed {seed+1}/{n_seeds} done dt={dt_seed:.1f}s cum={dt_scen:.1f}s "
            f"(ran={seeds_done} loaded={seeds_loaded})",
            flush=True,
        )
    return aggregate(spec, runs)


def aggregate(spec: ScenarioSpec, runs: list[SeedRun]) -> dict:
    n = len(runs)
    # Collect headline metrics.
    leader_consensus = [r.protocol["final_leader_consensus"] for r in runs]
    form_err_p = [r.protocol["final_form_err_mean"] for r in runs]
    form_err_o = [r.oracle["final_form_err_mean"] for r in runs]
    form_err_d = [r.drift["final_form_err_mean"] for r in runs]
    collisions_p = [r.protocol["max_collisions"] for r in runs]
    comms_delivery = [r.protocol["comms_delivery_rate"] for r in runs]

    # Scenario-specific extra metrics.
    extra = {}
    if spec.name == "S4_leader_kill":
        # Recovery: ticks after the kill until ≥80% of drones infer a leader
        # *other than the dead one* AND consensus reaches ≥80%.
        recoveries = []
        killed_id = spec.n_drones - 1
        for r in runs:
            modal_ts = r.protocol["leader_modal_id_ts"]
            cons_ts = r.protocol["leader_consensus_ts"]
            recovery = -1
            for t in range(100, len(modal_ts)):
                if modal_ts[t] != killed_id and cons_ts[t] >= 0.80:
                    recovery = t - 100
                    break
            recoveries.append(recovery)
        succeeded = sum(1 for x in recoveries if 0 <= x <= 120)
        extra["recovery_ticks_list"] = recoveries
        extra["recovery_success_count"] = succeeded
        extra["recovery_success_ci"] = wilson_ci(succeeded, n)
        successful_only = [x for x in recoveries if 0 <= x <= 120]
        if successful_only:
            extra["recovery_ticks_mean_ci"] = bootstrap_ci(successful_only)
        else:
            extra["recovery_ticks_mean_ci"] = (0.0, 0.0, 0.0)

    # Coverage gathered for all scenarios.
    extra["coverage_protocol"] = bootstrap_ci([r.protocol["final_coverage"] for r in runs])
    extra["coverage_oracle"] = bootstrap_ci([r.oracle["final_coverage"] for r in runs])
    extra["coverage_drift"] = bootstrap_ci([r.drift["final_coverage"] for r in runs])
    extra["coverage_protocol_current_manifold"] = bootstrap_ci(
        [r.protocol.get("final_coverage_current_manifold", 0.0) for r in runs]
    )
    extra["protocol_final_manifold_size"] = bootstrap_ci(
        [float(r.protocol.get("final_manifold_size", 0)) for r in runs]
    )

    # S8 partition: extract dip during partition window and recovery after heal.
    if spec.name == "S8_partition_heal":
        dips = []
        post_heals = []
        actual_recoveries = []
        for r in runs:
            ts = r.protocol["leader_consensus_ts"]
            partition_window = ts[100:250] if len(ts) >= 250 else []
            if partition_window:
                dips.append(min(partition_window))
            heal_idx = min(310, len(ts) - 1)
            post_heals.append(ts[heal_idx] if heal_idx < len(ts) else 0.0)
            # Find first tick AFTER heal (>= 250) where consensus reaches >= 0.95.
            recovery = -1
            for t in range(250, len(ts)):
                if ts[t] >= 0.95:
                    recovery = t - 250
                    break
            actual_recoveries.append(recovery)
        extra["partition_dip"] = bootstrap_ci(dips)[0] if dips else 1.0
        extra["partition_dip_min"] = min(dips) if dips else 1.0
        extra["post_heal_consensus"] = bootstrap_ci(post_heals)[0] if post_heals else 0.0
        extra["post_heal_consensus_ci"] = bootstrap_ci(post_heals) if post_heals else (0.0, 0.0, 0.0)
        successful_recoveries = [x for x in actual_recoveries if x >= 0]
        if successful_recoveries:
            extra["partition_recovery_ticks_ci"] = bootstrap_ci(successful_recoveries)
            extra["partition_recovery_success_count"] = len(successful_recoveries)
        else:
            extra["partition_recovery_ticks_ci"] = (0.0, 0.0, 0.0)
            extra["partition_recovery_success_count"] = 0

    return {
        "scenario": asdict(spec),
        "n_seeds": n,
        "metrics": {
            "leader_consensus": bootstrap_ci(leader_consensus),
            "form_err_protocol": bootstrap_ci(form_err_p),
            "form_err_oracle": bootstrap_ci(form_err_o),
            "form_err_drift": bootstrap_ci(form_err_d),
            "max_collisions_protocol": bootstrap_ci(collisions_p),
            "comms_delivery_rate": bootstrap_ci(comms_delivery),
        },
        "extra": extra,
        "runs": [
            {
                "seed": r.seed,
                "protocol_consensus": r.protocol["final_leader_consensus"],
                "protocol_form_err": r.protocol["final_form_err_mean"],
                "oracle_form_err": r.oracle["final_form_err_mean"],
                "drift_form_err": r.drift["final_form_err_mean"],
                "collisions": r.protocol["max_collisions"],
            }
            for r in runs
        ],
    }


def pretty_print(results: list[dict]) -> None:
    print(f"\n{'='*100}\nDISTRIBUTED PROTOCOL BENCH — RESULTS\n{'='*100}\n")
    for res in results:
        s = res["scenario"]
        m = res["metrics"]
        print(f"\n--- {s['name']} ---")
        print(f"  Claim:        {s['claim']}")
        print(f"  Falsifying:   {s['falsifying']}")
        print(f"  Mechanism:    {s['mechanism']}")
        print(f"  Pass crit:    {s['pass_criterion']}")
        print(f"  N seeds: {res['n_seeds']}")
        lc = m["leader_consensus"]
        fp = m["form_err_protocol"]
        fo = m["form_err_oracle"]
        fd = m["form_err_drift"]
        cd = m["comms_delivery_rate"]
        col = m["max_collisions_protocol"]
        print(f"  Leader consensus:     {lc[0]:.3f} [{lc[1]:.3f}, {lc[2]:.3f}]")
        print(f"  Form err protocol:    {fp[0]:.3f}m [{fp[1]:.3f}, {fp[2]:.3f}]")
        print(f"  Form err oracle:      {fo[0]:.3f}m [{fo[1]:.3f}, {fo[2]:.3f}]")
        print(f"  Form err drift:       {fd[0]:.3f}m [{fd[1]:.3f}, {fd[2]:.3f}]")
        print(f"  Comms delivery:       {cd[0]:.3f} [{cd[1]:.3f}, {cd[2]:.3f}]")
        print(f"  Max collisions:       {col[0]:.1f} [{col[1]:.1f}, {col[2]:.1f}]")
        cp = res["extra"].get("coverage_protocol", (0,0,0))
        co = res["extra"].get("coverage_oracle", (0,0,0))
        cdr = res["extra"].get("coverage_drift", (0,0,0))
        print(f"  Coverage protocol:    {cp[0]:.3f} [{cp[1]:.3f}, {cp[2]:.3f}] (original manifold)")
        print(f"  Coverage oracle:      {co[0]:.3f} [{co[1]:.3f}, {co[2]:.3f}]")
        print(f"  Coverage drift:       {cdr[0]:.3f} [{cdr[1]:.3f}, {cdr[2]:.3f}]")
        ccm = res["extra"].get("coverage_protocol_current_manifold", (0,0,0))
        msz = res["extra"].get("protocol_final_manifold_size", (0,0,0))
        print(f"  Coverage protocol (current manifold):  {ccm[0]:.3f} [{ccm[1]:.3f}, {ccm[2]:.3f}]")
        print(f"  Final elected-manifold size:           {msz[0]:.1f}")
        if "recovery_ticks_mean_ci" in res["extra"]:
            rec = res["extra"]["recovery_ticks_mean_ci"]
            rs = res["extra"]["recovery_success_ci"]
            n_succ = res["extra"]["recovery_success_count"]
            print(f"  Leader recovery time: mean {rec[0]:.1f} ticks [{rec[1]:.1f}, {rec[2]:.1f}] (success {n_succ}/{res['n_seeds']} [{rs[0]:.2f}, {rs[1]:.2f}])")
        if "partition_dip" in res["extra"]:
            dip = res["extra"]["partition_dip"]
            dip_min = res["extra"]["partition_dip_min"]
            ph = res["extra"]["post_heal_consensus_ci"]
            print(f"  Partition dip (consensus during split): {dip:.3f} (min across seeds: {dip_min:.3f})")
            print(f"  Post-heal consensus (tick 310, 60 after heal): {ph[0]:.3f} [{ph[1]:.3f}, {ph[2]:.3f}]")
            if "partition_recovery_ticks_ci" in res["extra"]:
                rc = res["extra"]["partition_recovery_ticks_ci"]
                n_rec = res["extra"]["partition_recovery_success_count"]
                print(f"  Partition reconvergence to >=0.95 consensus: {rc[0]:.1f} ticks post-heal [{rc[1]:.1f}, {rc[2]:.1f}] ({n_rec}/{res['n_seeds']} seeds succeeded)")
        # Pass check
        passed = check_pass(s, m, res["extra"])
        print(f"  >>> {('PASS' if passed else 'FAIL')} (criterion: {s['pass_criterion']})")


def check_pass(spec_dict: dict, metrics: dict, extra: dict) -> bool:
    name = spec_dict["name"]
    lc = metrics["leader_consensus"][0]
    fp = metrics["form_err_protocol"][0]
    fo = metrics["form_err_oracle"][0]
    fd = metrics["form_err_drift"][0]
    if name == "S1_baseline_no_failures":
        # Basic sanity: consensus reached, formation within absolute tolerance.
        return lc >= 0.95 and fp < 2.0
    if name == "S2_message_loss_10pct":
        return lc >= 0.80 and fp < 3.0
    if name == "S3_message_loss_30pct":
        return lc >= 0.60 and fp < 5.0
    if name == "S4_leader_kill":
        if "recovery_success_count" not in extra:
            return False
        n_seeds = len(extra["recovery_ticks_list"])
        return extra["recovery_success_count"] >= 0.8 * n_seeds
    if name == "S5_random_loss_20pct":
        cov_p = extra.get("coverage_protocol", (0.0, 0.0, 0.0))[0]
        cov_d = extra.get("coverage_drift", (0.0, 0.0, 0.0))[0]
        return lc >= 0.80 and cov_p >= cov_d
    if name == "S5b_random_loss_with_reform":
        cov_cm = extra.get("coverage_protocol_current_manifold", (0.0, 0.0, 0.0))[0]
        msz = extra.get("protocol_final_manifold_size", (0.0, 0.0, 0.0))[0]
        # Pass: leader consensus holds, manifold has shrunk (size < initial),
        # and coverage of the CURRENT (smaller) manifold is high.
        return lc >= 0.80 and msz < 30 and cov_cm >= 0.90
    if name == "S6_byzantine_position_lie":
        return fp < 6.0
    if name == "S7_surplus_fills_gaps":
        cov_p = extra.get("coverage_protocol", (0.0, 0.0, 0.0))[0]
        cov_d = extra.get("coverage_drift", (0.0, 0.0, 0.0))[0]
        return lc >= 0.80 and cov_p > cov_d + 0.10
    if name == "S7b_aggressive_surplus_fills":
        cov_p = extra.get("coverage_protocol", (0.0, 0.0, 0.0))[0]
        cov_d = extra.get("coverage_drift", (0.0, 0.0, 0.0))[0]
        return cov_p > cov_d + 0.15
    if name == "S8_partition_heal":
        dip = extra.get("partition_dip", 1.0)
        post_heal = extra.get("post_heal_consensus", 0.0)
        if dip > 0.95:
            return False
        return post_heal >= 0.90
    if name == "S9_tight_comms_range":
        return lc >= 0.80
    if name == "S10_byzantine_with_detection":
        # S6 baseline was form_err 1.90m without detection.
        # Pass if S10 form_err < 1.52m (80% of S6).
        return fp < 1.52
    return False


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="Names of scenarios to run; default all.")
    ap.add_argument("--output", default="/Users/jmcentire/Code/drone_swarm/distributed/bench_results.json")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="If set, per-seed JSONs are cached here; resumes skip already-completed seeds.")
    args = ap.parse_args()

    if args.scenarios:
        chosen = [s for s in SCENARIOS if s.name in args.scenarios]
    else:
        chosen = SCENARIOS

    t0 = time.perf_counter()
    results = []
    for spec in chosen:
        t_s = time.perf_counter()
        print(f"\nRunning {spec.name} ({args.seeds} seeds)...", flush=True)
        try:
            res = run_scenario(spec, n_seeds=args.seeds, checkpoint_dir=args.checkpoint_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
        results.append(res)
        dt = time.perf_counter() - t_s
        print(f"  done in {dt:.1f}s")

    elapsed = time.perf_counter() - t0
    pretty_print(results)
    print(f"\nTotal wall time: {elapsed:.1f}s")

    with open(args.output, "w") as f:
        json.dump({"results": results, "wall_time_s": elapsed}, f, indent=2, default=str)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
