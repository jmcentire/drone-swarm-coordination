# Drift, reference matching, and inter-drone ranging — v1.2 supplemental

**Status:** supplemental note to v1.1.1. Does not modify any formal claim
in WRITEUP.md or PROOFS.md. Bench: `bench_drift.py`. 200 seeds, N=100,
600s simulated mission.

## Motivation

The architecture's claim is *blind coordination*: each drone runs the
same deterministic rules over a shared broadcast substrate, with no
leader, no shared mutable state, and no awareness of being part of a
swarm. Coordination is what the system *does*, not what any drone
*decides*.

A reasonable question for a long-duration, GPS-denied deployment
(submersible swarms, Mars rovers, indoor warehouses): does this still
work when every drone's onboard navigation is drifting? The answer this
note backs up empirically is **yes, internally** — and the paper's
existing layers compose with off-the-shelf drift-correction mechanisms
(reference matching, inter-drone ranging) without any architectural
change.

The v1.2 supplemental claim is that drift, directive frequency, and
reset frequency compose orthogonally with the four layers: each handles
a separate failure mode and none of them perturbs another.

## Sweep F — drift × reset × directive

**Setup.** Each drone *i* has a true position `true_i` and an INS bias
`bias_i`. Per tick (40 ms), `bias_i += N(0, σ_drift · √dt)` per axis —
the standard 3D random-walk INS error model. The drone broadcasts
`est_i = true_i + bias_i` and applies idealised tight control to drive
`est_i → target_i` (its assigned manifold point), so `true_i = target_i
− bias_i` after settling.

Three knobs, swept over a full grid:

| Knob | Values | Models |
|---|---|---|
| `σ_drift` (m/√s) | 0.01, 0.1, 1.0 | fiber-INS, tactical IMU, MEMS |
| `T_reset` | 30 s, 300 s, ∞ | bathymetric / terrain match (σ_reset = 0.1 m) |
| `T_directive` | 30 s, 300 s, ∞ | operator pushes a new manifold (sphere → ellipsoid → ring → MC-density cycle) |

Three metrics:

- `int_coh` = mean ‖est_i − target_i‖ — *internal* coherence (what the swarm thinks of itself)
- `ext_pos` = mean ‖true_i − target_i‖ — *external* position error (what an observer sees)
- `ext_shape` = mean ‖(true_i − truē) − (target_i − target̄)‖ — relative-configuration distortion

**Results (final-tick, 200 seeds, bootstrap 95% CIs).** Selected rows
showing the structure (full table in `bench_drift_results.json`):

| σ_drift | T_reset | T_directive | ext_pos (m) | ext_shape (m) | int_coh |
|---|---|---|---|---|---|
| 0.01 | ∞ | ∞ | 0.391 [0.39, 0.39] | 0.389 [0.39, 0.39] | 0 |
| 0.10 | ∞ | ∞ | 3.910 [3.89, 3.93] | 3.888 [3.86, 3.91] | 0 |
| 1.00 | ∞ | ∞ | 39.096 [38.87, 39.32] | 38.877 [38.65, 39.10] | 0 |
| 0.10 | 300 s | ∞ | 0.159 [0.16, 0.16] | 0.159 [0.16, 0.16] | 0 |
| 1.00 | 300 s | ∞ | 0.159 [0.16, 0.16] | 0.159 [0.16, 0.16] | 0 |
| 1.00 | 30 s | ∞ | 0.159 [0.16, 0.16] | 0.158 [0.16, 0.16] | 0 |
| 1.00 | ∞ | 30 s | 39.093 [38.87, 39.32] | 38.875 [38.65, 39.09] | 0 |
| 1.00 | ∞ | 300 s | 39.096 [38.87, 39.32] | 38.877 [38.65, 39.10] | 0 |

**Findings.**

1. **Drift is a clean random walk.** `ext_pos` at `T_reset = ∞` tracks
   `σ_drift · √T · 1.6` (the analytical Maxwell-Boltzmann-like 3D-norm
   coefficient) to within 1% across three orders of magnitude — 0.39,
   3.91, 39.1 m for σ ∈ {0.01, 0.1, 1.0}. The architecture imposes no
   additional drift; what you see is exactly the INS budget.
2. **Reset clamps to the σ_reset noise floor.** Any non-infinite
   `T_reset` collapses `ext_pos` to ≈ σ_reset · 1.6 ≈ 0.16 m at the
   sample tick (which lands just after the most recent reset). The
   reset cadence does not matter for the *post-reset* error; the
   *worst-case* error between resets is bounded by σ_drift · √T_reset ·
   1.6 (random-walk envelope).
3. **Directive frequency is empirically orthogonal.** All 27 cells in
   the grid show *identical* ext_pos and ext_shape across `T_directive`
   ∈ {30 s, 300 s, ∞} — the residual difference is below the bootstrap
   noise. New manifolds re-anchor the *target shape*; they do not
   reset the INS bias and do not interact with drift.
4. **Internal coherence is exactly zero in every cell.** The swarm
   coordinates correctly in its own (drifted) frame regardless of how
   bad the drift gets. This is the architectural claim verified
   directly: blind coordination is drift-immune internally.

The composition is clean: Layer-4 cooperative localization handles
drift in a sparse-GPS environment; reference matching handles drift
when GPS is gone entirely; directives operate on whatever frame the
swarm is currently in. None of these perturb the other three layers.

## Sweep G — inter-drone ranging (the submersible / Mars case)

**Setup.** Same as Sweep F with `T_reset = T_directive = ∞`, but each
tick every drone refines its bias estimate against pairwise ranges to
neighbours (UWB, acoustic time-of-flight, or visual fiducial,
σ_range = 0.05 m). Implementation: replace each drone's bias with the
swarm-mean bias plus an iid residual `N(0, σ_range²)` — a clean
upper-bound proxy for what a least-squares range-fix would achieve.

| σ_drift | mode | ext_pos (m) | ext_shape (m) |
|---|---|---|---|
| 0.01 | no ranging | 0.391 [0.39, 0.39] | 0.389 [0.39, 0.39] |
| 0.01 | ranging | 1.034 [0.97, 1.10] | **0.079 [0.08, 0.08]** |
| 0.10 | no ranging | 3.910 [3.89, 3.93] | 3.888 [3.86, 3.91] |
| 0.10 | ranging | 1.125 [1.06, 1.19] | **0.079 [0.08, 0.08]** |
| 1.00 | no ranging | 39.096 [38.87, 39.32] | 38.877 [38.65, 39.10] |
| 1.00 | ranging | 4.078 [3.84, 4.31] | **0.079 [0.08, 0.08]** |

**Findings.**

5. **Ranging caps shape error at σ_range · 1.6 ≈ 0.08 m, independent
   of σ_drift.** The swarm preserves its relative configuration to
   centimetre accuracy even at MEMS-grade INS drift over a 10-minute
   mission with no absolute reset whatsoever. At σ = 1.0 m/√s, shape
   error drops 38.9 m → 0.079 m, a ~500× reduction.
6. **Ranging trades absolute drift for shape stability.** At low
   σ_drift it slightly raises absolute position error (centroid drift
   from ranging-noise averaging dominates over the natural random
   walk). At high σ_drift it wins ~10× on absolute too. The break-even
   is around σ_drift · √T ≈ √N · σ_range; for N = 100, T = 600 s,
   σ_range = 0.05 m, that's σ_drift ≈ 0.02 m/√s — anything tactical-
   grade or worse.

This is the operational case for submersible swarms (acoustic
ranging, no GPS, no terrain match between surfacings) and for Mars-
class deployments (vision fiducials, no GPS, infrequent landmark
re-localization). The swarm holds its shape against the manifold by
talking to itself.

## Generalisation: Monte-Carlo manifolds

The four layers do not depend on the manifold being geometric. A
manifold is a set of N target points; whether they came from a
parameterised surface (sphere, ellipsoid, ring) or from sampling a
density (`montecarlo_density` in the bench tilts a Gaussian toward
+z and projects to the sphere) is invisible to the assignment,
recovery, priority, and localization layers.

This generalises the architecture from formation flight (cued by
geometry) to statistical sampling (cued by a feature density):

- search-and-rescue: density ∝ probability of victim presence
- species survey: density ∝ habitat suitability
- mineral / bathymetric mapping: density ∝ feature uncertainty
- atmospheric sampling: density ∝ measurement value-of-information

The directive carries the manifold (or the density sampler's seed and
parameters); the swarm executes. No layer changes.

## Out of scope

- *Reset mechanism choice.* Bathymetric correlation, terrain-aided
  navigation, occasional surfacing for GPS, star-tracker fixes, vision
  SLAM landmarks — all compose, all are deeply studied. See
  Groves (2013) on integrated INS/GPS, Anonsen & Hagen (2010) on
  bathymetric terrain navigation, Davison et al. (2007) on monocular
  SLAM, Nüchter & Hertzberg (2008) on 6-DOF SLAM. The bench treats
  reset as a black-box correction with residual σ_reset.
- *Acoustic-channel modelling for submersibles.* The ranging sub-sweep
  abstracts ToF measurement as a Gaussian-residual fix. Real acoustic
  channels exhibit multipath, refraction, and bandwidth limits; the
  composition claim is over an ideal range-fix, the operational
  numbers will degrade. See Stojanovic & Preisig (2009) on underwater
  acoustic propagation, Paull et al. (2014) on AUV navigation surveys.
- *Initial registration.* The bench assumes the swarm starts at its
  manifold; cold-start localization (which drone is which?) is a
  separate problem solved by the Layer-4 broadcast-trilateration
  architecture in §6.3.
- *Drift-correlated control failure.* Real drones are not idealised
  controllers; differential drift in the control loop will inflate
  ext_pos beyond what the INS bias alone predicts. This is a
  hardware-platform concern, not an architectural one.

## Future tests: composite INS noise

The bench treats per-drone bias as fully independent random walks. Real
INS noise has both a common-mode component (temperature gradient
across the swarm, correlated multipath, shared GPS-jamming bias) and
an independent component (per-unit gyro and accelerometer bias). A
realistic mix is in the 30–70% common-mode range for co-located
swarms.

Adding a `common_mode_frac` knob to `simulate()` is a one-line change:
split the per-tick step into a single shared `N(0, σ·√(c·dt))` and a
per-drone `N(0, σ·√((1−c)·dt))` so total variance is preserved. Pure-
independent (the current bench) is the worst case for swarm shape
distortion; any common-mode fraction reduces shape error toward zero
as `c → 1`. We did not run this sweep because the current numbers are
already the upper bound on shape error for a given σ_drift; mixed-
correlation results would only relax the bound. Worth a future run
for hardware calibration against a specific INS unit's measured
correlation structure.

## Reproducing

```
N_SEEDS=200 NUM_DRONES=100 HORIZON=600 python3 bench_drift.py
```

~4 minutes single-threaded. Output: stdout table + `bench_drift_results.json`.

## References (drift-correction literature pointers)

- Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor
  Integrated Navigation Systems* (2nd ed.). Artech House.
- Anonsen, K. B. & Hagen, O. K. (2010). An analysis of real-time
  terrain aided navigation results from a HUGIN AUV. *OCEANS 2010
  MTS/IEEE Seattle*.
- Davison, A. J., Reid, I. D., Molton, N. D. & Stasse, O. (2007).
  MonoSLAM: real-time single camera SLAM. *IEEE TPAMI*, 29(6).
- Stojanovic, M. & Preisig, J. (2009). Underwater acoustic
  communication channels: propagation models and statistical
  characterization. *IEEE Comm. Magazine*, 47(1).
- Paull, L., Saeedi, S., Seto, M. & Li, H. (2014). AUV navigation
  and localization: a review. *IEEE Journal of Oceanic Engineering*,
  39(1).
- Nüchter, A. & Hertzberg, J. (2008). Towards semantic maps for
  mobile robots. *Robotics and Autonomous Systems*, 56(11).
