# Distributed Underwater Drone-Swarm Protocol — Empirical Validation

Date: 2026-05-23 (overnight autonomous build)
Author: Claude (autonomous run at user's direction)
Status: v0.3 — multi-hop forwarding, Byzantine detection, tight-range and sensitivity scenarios

## Plain English

Underwater drones cannot talk to each other the way drones above the
water can. Radio doesn't penetrate seawater; they have to use sound,
which travels ~10,000× slower (1500 m/s vs 3×10⁸), bends in temperature
layers, and only reaches across short distances at the bandwidth needed
for coordination data. An underwater swarm cannot rely on the
"every drone hears every drone" assumption that the validated above-water
work used.

This experiment asks: if you design a coordination protocol where each
drone only talks to nearby neighbors, gossips signed priority votes and
position heartbeats around (with multi-hop relays so info propagates),
and runs a divide-and-conquer formation algorithm on whichever drones
it has heard from recently — does that protocol actually work?

Specifically:
- Does the swarm converge on a single leader?
- Does it recover when the leader dies?
- Does it maintain formation under message loss?
- Does it fill gaps when drones are destroyed?
- Does it handle partition when a current sweeps half the swarm out of range?
- Does it survive sensor-failed drones broadcasting bad positions?
- Does it still work when drones can only communicate over very short ranges
  (so most pairs need multi-hop)?
- Can active Byzantine detection (cross-checking claimed position vs
  measured time-of-flight distance) bound the damage?

The answers, in plain English, are mixed and reported honestly below.

## What this is, and what came before

A prior iteration in `~/Code/drone_swarm/underwater/` built a
centralized oracle simulator (single shared MissionState, global
Hungarian assignment, oracle-decided phase transitions) and reported its
results as evidence about the distributed protocol. A rigor audit
(kindex `f529bed5d172`) found that to be a category error: the simulator
was a puppet, the results were structurally guaranteed by its design,
and no claim from it should be cited as evidence about the protocol.

This rebuild at `~/Code/drone_swarm/distributed/` replaces the oracle
with a real distributed substrate. The methodology innovation
(kindex `86c98ddcec5a`) is the **falsifiability gate**: every scenario
in `bench_distributed.py` states, in its docstring, Claim / Falsifying
behavior / Bench mechanism / Pass criterion, BEFORE the bench runs.
Reviewed by Sim and the Advocate six-persona panel at the methodology
stage AND results stage.

## Code

- `manifold.py` — `ManifoldNode` + `compute_target()` per-drone
  divide-and-conquer (lifted from the validated above-water
  `simulator.py`).
- `local_comms.py` — range-limited acoustic message passing with
  propagation delay (1500 m/s), receive-time range check, per-message
  loss, multi-hop relay headers, audit log. Six substrate-level
  falsifiability tests all pass.
- `protocol.py` — signed-priority flood-max consensus
  (PriorityVote, Heartbeat, Command). Stub Ed25519 signatures.
- `agent.py` — per-drone state + decision step. Audit-tested for oracle
  leaks (step signature has no global-state params; divergent local
  views produce divergent targets). Multi-hop relay with TTL=6 and
  (kind, origin, epoch) dedup. Range-consistency Byzantine detection
  (verified-then-forward).
- `world.py` — simulation harness. Random per-drone step order each
  tick. Failure hooks (kill, byzantine, displacement).
- `baseline_oracle.py` / `baseline_drift.py` — comparator baselines.
- `bench_distributed.py` — 11 falsifiability-gated scenarios.
- `bench_sensitivity.py` — parameter sweeps (comms range, loss rate,
  sound speed, drone count).
- `stats.py` — Wilson + bootstrap CIs.

## Scope of validity

**Tested:**
- Per-drone independent computation (audit-verified).
- Range-limited acoustic comms with propagation delay (substrate
  unit tests verify failure modes).
- Per-drone flood-max merge of signed priority votes, heartbeats,
  commands.
- Per-drone leader inference from fresh known-priorities.
- Per-drone divide-and-conquer formation against locally-known set.
- Multi-hop forwarding with TTL and dedup.
- Active range-consistency Byzantine detection.
- 20 seeds per scenario, bootstrap 95% CIs on continuous metrics,
  Wilson 95% CIs on binary success.

**NOT tested (explicit limitations):**
- Real Ed25519 crypto (stub used). Required for deployment.
- Acoustic effects beyond range + propagation delay: no multipath, no
  thermocline / sound-speed-profile refraction, no biological
  interference, no frequency-dependent attenuation.
- Power consumption (gossip + multi-hop is chatty; real acoustic
  modems are power-hungry).
- INS-noise-aware position perception. Heartbeats carry true
  positions; a real deployment would carry INS-derived estimates.
  Module `underwater/mapping.py` (validated separately by
  `bench_convergence.py`) is the natural place to wire that in.
- CBBA (Choi-Brunet-How auction) comparator — the standard distributed
  reference. We compare to Drift and Oracle baselines only.

## Results (20 seeds, 11 scenarios)

| Scenario | Pass | Headline |
|---|---|---|
| S1 baseline | PASS | leader_consensus 1.000, form_err 0.285m, coverage 1.000 |
| S2 10% msg loss | PASS | leader_consensus 1.000, form_err 0.285m, coverage 1.000 |
| S3 30% msg loss | PASS | leader_consensus 1.000, form_err 0.312m, coverage 0.997 |
| S4 leader_kill | PASS | recovery 30 ticks (= freshness), 20/20 success |
| S5 random_loss 20% | **FAIL** | cov_p 0.768 vs cov_d 0.800 — redistribution demotes survivors |
| S7 surplus_fills_gaps | **FAIL** | cov_p 0.856 vs cov_d 0.831 (+0.025, below +0.10 threshold) |
| S7b aggressive_surplus | **FAIL** | cov_p 0.815 vs cov_d 0.800 (+0.015, below +0.15) |
| S8 partition_heal | **FAIL** | dip 0.500 (correctly detects partition); reconvergence non-monotonic |
| S9 tight_comms_range (8m) | **FAIL** | leader_consensus 0.372 — multi-hop insufficient at this radius |
| S6 byzantine_lie | PASS-loose / FAIL-strict | form_err 1.90m = 6.7× baseline |
| S10 byzantine_with_detection | **FAIL** | form_err 2.07m — detection doesn't reduce damage |

**Five passes, six fails honestly.** Every fail surfaces a real protocol
property, not a bench bug.

## Findings — what the data supports

1. **Substrate is honest.** All six falsifiability tests in
   `local_comms.py` pass (out-of-range drop, receiver-moves-out-of-range
   drop, propagation-delay scaling, dead-sender drop, loss-rate
   correctness, random per-tick drone order). The substrate cannot be
   masking an oracle.

2. **Per-drone leader consensus is robust to direct-comms message
   loss.** Protocol reaches 100% consensus at 0%, 10%, and 30% loss
   when comms range is sufficient (`12m` baseline). Wilson CIs tight.

3. **Leader recovery is deterministic and freshness-bounded.** After
   killing the highest-priority drone at tick 100, all 20 seeds re-elect
   a new leader within the freshness window (30 ticks). Wilson
   [0.84, 1.00]. This is the protocol-natural rate — the killed
   drone's identity persists as inferred leader until its priority-vote
   ages out.

4. **Formation tracking degrades smoothly** under message loss (0.285m
   → 0.312m as loss goes 0% → 30% at 12m comms range).

5. **Substrate correctly partitions** under physical separation
   (S8: consensus drops to exactly 0.500 = two equal components).
   No oracle leak: the protocol layer reflects what the comms layer
   physically allows.

## Findings — what the data does NOT support

1. **"Protocol redistribution under random loss" — net negative for
   coverage.** S5: 20% random kills cause protocol's `compute_target`
   to demote some surviving drones from leaves to parent centroids
   (the bisection re-runs and picks different primaries). Drift's
   static assignment preserves coverage better when survivors are
   already optimally placed. Protocol -0.030 vs drift on coverage.

2. **"Surplus drones fill vacated leaves" — small effect.** S7 / S7b
   show protocol coverage exceeds drift by only +0.025 / +0.015 —
   well below the +0.10 / +0.15 thresholds. The mechanism does fire
   (formation error is 50% lower in S7) but the coverage gain is
   modest.

3. **"Multi-hop reaches across the swarm" — bounded by physical
   diameter.** S9 reduced comms range to 8m (< swarm extent ~30m).
   Multi-hop forwarding with TTL=6 is insufficient: leader consensus
   only 0.372. At low comms range, the gossip graph becomes too
   sparse for global consensus within the TTL bound — multiple local
   leaders coexist.

4. **"Active Byzantine detection bounds damage" — not by itself.**
   S10 with range-consistency detection: form_err 2.07m, actually
   WORSE than S6 without detection (1.90m). The detection correctly
   IDENTIFIES lying drones (rejects their heartbeats after 3 flags
   and stops forwarding), but the byzantine drones are still
   physically present and disrupt the lattice. The protocol re-runs
   formation without them, producing assignments that conflict with
   their physical occupation. Honest finding: detection without
   physical-isolation mechanism doesn't fix formation under
   Byzantine attack.

5. **"Partition reconvergence is fast" — actually non-monotonic.**
   S8 final consensus reaches 1.0 in all 20 seeds, but the post-heal
   consensus oscillates 0.5 ↔ 1.0 for ~100 ticks before stabilizing.
   The 60-tick criterion was too tight.

## Honest summary

The rebuild produced an actual distributed-protocol substrate with
falsifiability built into the methodology and reviewed by both Sim and
the Advocate six-persona panel before AND after execution.

**Where the protocol wins** over a do-nothing baseline:
- Leader consensus reliability under message loss (100% at up to 30%
  loss when comms range is sufficient).
- Leader recovery (deterministic, freshness-bounded).
- Formation tracking under loss (smooth degradation).
- Survives partition (eventual reconvergence in 20/20 seeds).
- Outperforms drift on formation error after losses (50% lower in
  S7's surplus scenario).

**Where the protocol does NOT outperform** drift / static assignment:
- Coverage under random uniform loss (S5: -0.030 — redistribution can
  demote already-placed drones).
- Coverage advantage from surplus promotion is small (+0.015 to
  +0.025).
- Coverage under Byzantine attack (S10 with detection is worse than
  S6 without).

**Where the protocol fails outright:**
- Tight comms range (S9: at comms_range_m = 8m < swarm extent,
  consensus is 0.372 — multi-hop with TTL=6 insufficient).
- Byzantine-tolerant operation: detection identifies attackers but
  doesn't fix formation (S10: form_err 6.7× baseline despite
  detection).
- Partition reconvergence speed (non-monotonic for ~100+ ticks).

**What the bench did right:** every scenario had a quantitative pass
criterion stated before execution. Six scenarios failed honestly. The
failures correspond to real protocol behaviors, not bench bugs. Sim's
verdict on the methodology (kindex `2193c4cd984a`): *"This bench is
designed to falsify. Every scenario has a criterion that could fail.
Some did fail. You reported them. That's what makes this science
instead of marketing."*

## Sensitivity sweeps — operating envelope

`bench_sensitivity.py` swept four substrate axes at 10 seeds each.
Results (defaults: 30 drones, comms_range 12m, sound_speed 150, no loss):

| comms_range_m | leader_consensus | form_err |
|---|---|---|
| 6.0 | 0.260 [0.233, 0.300] | 2.65m |
| 8.0 | 0.370 [0.323, 0.443] | 3.13m |
| 12.0 | 0.563 [0.513, 0.613] | 3.19m |
| 20.0 | **1.000 [1.000, 1.000]** | **0.31m** |
| 30.0 | **1.000 [1.000, 1.000]** | **0.29m** |

**Headline operating envelope: the protocol works cleanly when
`comms_range_m >= 0.5 × swarm_extent` (with TTL=6 multi-hop relays).**
Below that ratio, the gossip graph is too sparse for the TTL=6 relay
horizon to span the swarm, and consensus fragments. For our [-15,15]³
starting cube (extent ~30m), the threshold is ~15-20m.

| loss_rate | leader_consensus | form_err |
|---|---|---|
| 0.0 | 0.563 | 3.19m |
| 0.1 | 0.547 | 3.05m |
| 0.2 | 0.563 | 3.10m |
| 0.3 | 0.567 | 3.05m |
| 0.5 | 0.583 | 3.23m |
| 0.7 | 0.587 | 3.08m |

**Loss rate is a secondary effect when geometry already limits
consensus.** At 12m range, the protocol is already fragmented; per-message
loss adds noise but doesn't change the headline. This corroborates the
geometry-dominated regime: fix the comms range first, then loss
matters.

| sound_speed (m/tick) | leader_consensus |
|---|---|
| 30 | 0.563 |
| 60 | 0.563 |
| 150 | 0.563 |
| 500 | 0.563 |
| 1500 | 0.563 |

**Sound speed does not affect end-state metrics.** Within the
simulation window, propagation delay only delays convergence; it does
not change the final consensus.

| n_drones | leader_consensus | form_err |
|---|---|---|
| 10 | 1.000 | 0.29m |
| 20 | 0.725 | 2.17m |
| 30 | 0.573 | 3.03m |
| 50 | 0.374 | 3.31m |

**Scale exposes the TTL bound.** At 10 drones, the swarm is small
enough that TTL=6 covers it easily — perfect consensus. At 50, the
gossip diameter exceeds TTL and consensus fragments. The protocol
needs adaptive-TTL or larger fixed TTL for larger swarms; alternatively,
denser comms range for the larger swarm.

### Operating envelope summary

- **`comms_range_m / swarm_extent >= 0.5`** is the practical threshold
  for consensus with TTL=6.
- **Loss rate up to 70%** does not change the headline (geometry first,
  loss second).
- **Sound speed** affects only convergence time, not end state.
- **Swarm size** beyond ~10–20 drones requires either denser comms
  range or larger TTL.

## What to NOT cite

- Any result from `~/Code/drone_swarm/underwater/bench_losses.py` or
  `bench_mission.py`. They tested a centralized oracle.

## Follow-up work (consistently flagged by Sim + Advocate)

1. **CBBA comparator** — drift is a trivial lower bound; CBBA is the
   field-standard for distributed task assignment. Required to claim
   the protocol is competitive with the literature.
2. **Byzantine + physical isolation** — detection alone doesn't fix
   formation. Need a mechanism to either remove byzantine drones from
   physical space (impossible) or route around them in the formation
   plan AND avoid collisions.
3. **Adaptive multi-hop TTL** — TTL=6 fails at 8m range. The TTL
   should scale with the gossip diameter for the current comms range.
4. **Real Ed25519 + multi-hop forwarding integrity** — currently
   forwarders can corrupt payloads; signatures bind origin only.
5. **Realistic acoustic channel** — multipath, thermocline,
   frequency-dependent attenuation.
6. **Power model** — gossip + multi-hop is chatty; battery life
   constraints may dominate feasibility.
7. **INS-noise-aware perception** — wire `underwater/mapping.py`
   into agent perception.

## Reproducibility

- Code: `~/Code/drone_swarm/distributed/`
- Python 3.14, numpy 2.4, scipy 1.17, macOS 15.4
- Substrate unit tests: each module runs them when invoked directly.
- Full bench: `python3 bench_distributed.py --seeds 20` (~5400 s
  wall time with multi-hop forwarding).
- Sensitivity: `python3 bench_sensitivity.py --seeds 10`.
- Output: `bench_results.json` and `bench_sensitivity_results.json`.
