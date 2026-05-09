# Swarm-A* exploration — v1.2 supplemental

**Status:** companion to NOTE_DRIFT.md. Does not modify any formal claim
in WRITEUP.md or PROOFS.md. Bench: `bench_astar.py`. 20 seeds × 3
environments, N=20 drones, 100×100 grid world.

## Motivation

A natural extension of the swarm architecture: the swarm starts knowing
only its initial position and a goal coordinate; the cost field between
is unknown. It explores by transiting through a sequence of manifolds,
sensing during each transit, and re-planning from the accumulating
shared map.

This is not vanilla A*. Vanilla A* assumes a known graph. The swarm
discovers the graph during traversal — closer to D\* Lite or LPA\* in
the partially-known-terrain literature, but generalised to a swarm of
agents whose collective sensor coverage is the discovery operation.

The architectural contribution is that **every drone runs the same A\*
against the same broadcast map at each manifold transition, reaching
byte-identical decisions on the next manifold without any leader.** The
existing four layers compose without modification:

- Layer 1 (assignment) places drones around each manifold centre.
- Layer 2 (recovery) handles drone failures during transit.
- Layer 3 (priority) is mission-specified and orthogonal.
- Layer 4 (localization) is the substrate for sensor positioning.
- The A\* selection is a deterministic function of broadcast state — the
  same byte-identical-inputs/outputs pattern as Theorem 2.5's mid-flight
  reconfiguration.

## Algorithm

Each iteration:

1. Every drone runs `grid_astar(broadcast_map, current_centre, goal)`.
   Unknown cells use an optimistic baseline cost. Output: a cell-by-cell
   path to goal.
2. The next manifold's centre is `path[STEP_CELLS]` — the swarm commits
   to STEP_CELLS of the planned path before re-planning.
3. Drones transit current → next centre with independent random walks
   (Mode B from NOTE_DRIFT.md), sensing the cost field within their
   footprint as they go.
4. Sensed values broadcast and the shared map updates.
5. Repeat until the centroid enters the goal disk, or one of three
   termination conditions fires: `max_iter`, `timeout`, `stuck` (no
   progress for `NO_PROGRESS_PATIENCE` iterations).

Operational hardening (see "Engineering" below): per-run wall-clock
timeout, no-progress detector, atomic checkpointing per (env, seed)
pair, structured status taxonomy.

## Test environments

| Env | Start → Goal | Threats | Threat magnitude |
|---|---|---|---|
| single_threat | (10, 50) → (90, 50) | 1 Gaussian bump | 0.4–1.0 |
| two_threats | (10, 50) → (90, 50) | 2 Gaussian bumps | 0.4–1.0 |
| dense_field | (10, 10) → (90, 90) | 5 Gaussian bumps | 0.4–0.8 |

Cost field = `BASELINE_COST + Σ exp(-d²/2σ²)` with σ ∈ [5, 12] cells per
threat. Baseline 0.05; sensor noise N(0, 0.05) clipped to ≥ 0.

## Results (20 seeds, bootstrap 95% CIs)

| Env | swarm-A\* cost | oracle cost | straight cost | gap vs oracle | savings vs straight | reached |
|---|---|---|---|---|---|---|
| single_threat | **6.37** [5.21, 7.78] | 4.80 [4.41, 5.25] | 7.51 [5.76, 9.41] | 27.6% [16.2, 40.9] | 8.0% [-0.2, 17.2] | 20/20 |
| two_threats | **7.40** [6.29, 8.54] | 5.23 [4.85, 5.59] | 12.08 [9.03, 15.79] | 39.0% [26.7, 52.0] | 23.9% [13.2, 35.1] | 20/20 |
| dense_field | **20.49** [16.67, 24.58] | 8.46 [7.95, 9.04] | 31.87 [28.33, 35.00] | 140.5% [97.5, 187.2] | 34.2% [23.8, 45.1] | 20/20 |

**Distributed-consensus byte-identity: 100% across all 60 runs.** Every
drone in every iteration arrived at the same `next_centre` decision
without any explicit coordination — the same input (broadcast map +
current position + goal) produces the same output (next manifold) by
construction.

**Wall time:** 15.2 seconds for 60 runs (3 environments × 20 seeds, with three
algorithms). Per-run swarm-A\* time: ~250 ms.

**Sensed coverage:** the swarm sensed 13–18% of the world's cells while
finding a path. The mission is *not* full mapping; it's path-finding
with on-line discovery. A coverage-style mission would extend this with
a different objective.

## Findings

1. **The architecture composes cleanly with on-line A\*.** Every
   drone's decision was byte-identical with every other drone's
   decision in every iteration. No leader, no auction, no consensus
   protocol beyond the broadcast substrate already provides.
2. **Swarm-A\* beats straight-line in environments with structured
   cost.** 24% savings vs straight in two_threats, 34% in dense_field.
   single_threat is closer to a coin flip (8% savings, CI crosses
   zero) because a single threat is often near or off the straight
   path; the swarm pays detour cost to avoid it but doesn't always
   come out ahead.
3. **The gap vs oracle grows with cost-field complexity.** 28% in
   single_threat, 39% in two_threats, 140% in dense_field. This is the
   information-asymmetry penalty: the swarm pays to discover what the
   oracle was given for free. Greedy-best-first re-planning over a
   partial map gets distracted by local minima — known-good detours
   that turn out to be longer than the unknown-baseline alternative
   would have been if explored first.
4. **All 60 seeds reached the goal.** Termination conditions
   (`timeout`, `stuck`, `max_iter`) never fired — the algorithm always
   converged to within `GOAL_TOLERANCE = 5` cells of goal within
   `MAX_ITERATIONS = 60` manifolds.

## Engineering notes (telemetry, fail-fast, recovery)

The first two attempts at this bench wedged on 20 GB of memory and
~14 seconds per A\* call before the bug surfaced. Two issues:

- *Negative cell costs.* Sensor noise on baseline 0.05 cells produced
  negative observed costs. With negative edge weights and an A\*
  implementation that re-pushes nodes when a lower g_score is found,
  the open queue grew without bound and expansions ran to the
  `max_expansions` cap (200K) before returning. **Fix**: clip sensed
  costs to ≥ 0 in `sense()`, plus a defensive clip in `grid_astar`'s
  edge relaxation. Skip already-closed cells before pushing.
- *No telemetry.* The bench was a black box — output only at the end,
  no per-iteration progress, no hard wall-clock cap on individual
  runs. **Fix**: per-iteration log to stderr (flushed), per-run
  `RUN_TIMEOUT_S` budget, no-progress detector with
  `NO_PROGRESS_PATIENCE` iterations, atomic checkpoint after every
  (env, seed) pair via tempfile + rename. The bench is now resumable —
  re-running it picks up where it left off rather than recomputing
  good work.

Status taxonomy returned per run: `reached` / `max_iter` / `timeout` /
`stuck` / `error:<msg>`. Aggregate counts in the per-env summary catch
silent regressions that pure cost averages would hide.

## Generalisation: where the architecture goes from here

The swarm-A\* algorithm composes with the four existing layers without
modifying them. The same generalisation that NOTE_DRIFT.md's
Monte-Carlo manifold paragraph noted applies here: a "manifold" is any
set of N target points, however generated, including:

- a geometric formation (sphere, ring, lattice),
- a sample from a feature density (mineral concentration, threat
  probability, victim likelihood),
- the next-step output of a planner over the broadcast map.

Search-and-rescue, reconnaissance with unknown threats, mineral
mapping, and species survey all fit this pattern. The mission directive
specifies `(start, goal, cost_function, sensor_model)` — the swarm
runs the same loop. No drone knows it's part of a swarm; coordination
is what the system does, not what any drone decides.

## Out of scope

- *Full coverage missions* (map every cell). The current bench
  optimises path cost, not coverage. A coverage variant changes the
  objective from "reach goal" to "sensed_fraction → 1"; the
  architecture still composes.
- *Adversarial cost-field adversaries* (a moving threat that responds
  to the swarm's discovery). The current bench has a static cost
  field. Adaptive adversaries are a known D\* Lite extension and
  compose by the same byte-identical-inputs argument.
- *Real sensor models.* The bench uses a square-radius footprint with
  Gaussian noise. Real sensors (sonar, optical, radio) have direction-
  dependent footprints, occlusion, range falloff. These are platform
  concerns; the planner doesn't change.
- *Communication failure during exploration.* The bench assumes
  perfect broadcast — every sensed cell propagates to every drone.
  Realistic comms-layer behaviour (NOTE_DRIFT.md, §9.1.5 of WRITEUP)
  composes orthogonally: the broadcast substrate's quiescence and
  loss handling apply unchanged.

## Reproducing

```
N_SEEDS=20 RUN_TIMEOUT_S=15 python3 bench_astar.py
```

~15 seconds single-threaded. Output: per-run telemetry on stderr,
summary on stdout, full results in `bench_astar_results.json`. Re-running
the script with the checkpoint file present skips already-completed
(env, seed) pairs.

## References (partial-knowledge planning literature)

- Stentz, A. (1995). The focussed D\* algorithm for real-time
  replanning. *IJCAI*.
- Koenig, S. & Likhachev, M. (2002). D\* Lite. *AAAI*.
- Koenig, S., Likhachev, M. & Furcy, D. (2004). Lifelong Planning A\*.
  *Artificial Intelligence*, 155(1-2).
- Burgard, W., Moors, M., Stachniss, C. & Schneider, F. E. (2005).
  Coordinated multi-robot exploration. *IEEE Transactions on
  Robotics*, 21(3).
- Yamauchi, B. (1998). Frontier-based exploration using multiple
  robots. *Autonomous Agents*.
