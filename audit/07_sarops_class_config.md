# SAROPS-class Comparator Config — Rationale

**Date pinned:** 2026-05-08
**Companion config:** `audit/07_sarops_class_config.yaml`
**Source audit:** `audit/01_sarops_lineage.md` §3
**Artifact label:** *SAROPS-class comparator (Stone-Kratzke-Allen; documented choices)*

For each of the 20 OPERATIONAL-UNKNOWNs from `01_sarops_lineage.md` §3, this document records the value pinned in the YAML config, the citation strength, the recommended sensitivity sweep, and the consequence if the choice is wrong. Strength tags follow the audit convention: PRIMARY (cited from K10/FS01/Allen/Stone), SECONDARY (cited via a follow-up such as BAM13 or BA08), INFERRED (extrapolated, inference documented), DOCUMENTED-CHOICE (literature silent; explicit choice with mandatory sweep).

---

## A. Particle Filter

### Particle count (PRIMARY)

K10 §2 documents user-selectable {2,500 / 5,000 / 10,000} per scenario. We pin **5,000** and sweep the full set. At 2,500 the PF moment variance rises ~√2; at 10,000 PF cost roughly doubles. The bench reports headline at 5,000 and confirms 2,500/10,000 do not reverse conclusions.

### Resampling policy (DOCUMENTED-CHOICE — load-bearing)

K10 documents no resampling step; the text is a pure importance-weighted Bayes update via the cumulative `pfail`. SRW16 §4.2 treats SAROPS particles as path samples, not Doucet-Crisan filter samples. We pin **none** (K10-faithful) as default; sweep adds systematic-at-ESS<N/2 and stratified-at-ESS<N/2. Audit §5 flags this explicitly: a no-resample SIS over many sortie iterations degenerates. Either operational SAROPS re-seeds via new scenarios faster than degeneracy occurs, or there is an undocumented resample. Impact-if-wrong: posteriors over ≥10 sortie iterations risk being dominated by a handful of particles; the sweep is mandatory for any multi-sortie bench result.

### Proposal distribution (INFERRED)

Bootstrap. K10 §2.5 propagates particles under prior dynamics only; the Bayes update is applied retroactively per sortie. This is the only proposal consistent with the K10 text. Strength stays INFERRED because K10 never names it.

### Time-step granularity (PRIMARY default, sweep mandatory)

FS01 §6.4.1 documents the CASP-era hourly step. K10 paraphrases "the next whole hour in simulated time, or the desired probability-map time" but does not pin a SAROPS-era step. We pin **60 min** (CASP-faithful) and sweep {5, 20, 60} per audit §3 item 1. The K10 60-min AR(1) half-life is consistent with a 60-min step (one decorrelation per step); a 5-min step gives a more accurate AR(1) discretization. The sweep exposes time-step axis sensitivity.

---

## B. Drift Dynamics

### Allen leeway tables (PRIMARY)

The 85-class SAROPS taxonomy is in OpenDrift `OBJECTPROP.DAT` (Allen → Breivik, 2010-11-15) at `/tmp/OBJECTPROP.DAT`. Column convention documented in `audit/02_leeway_tables.md`. This removes the Allen 2005 paywall risk. No choice.

### Standard vs Rayleigh DWL slope (DOCUMENTED-CHOICE)

K10 §2.5 prescribes Rayleigh "for object types with very small m" to avoid Gaussian-tail negative slopes. No threshold is published in K10, BA08, or BAM11. We pick **0.5%** slope threshold; sweep {0.1, 0.5, 1.0, 2.0}%. Impact-if-wrong: too-low a threshold permits unphysical upwind drift; too-high a threshold artificially suppresses moderate-windage variance and shifts the posterior mode a few percent due to the Rayleigh-vs-Gaussian asymmetry.

### Crosswind sign-flip mean rate (DOCUMENTED-CHOICE — load-bearing for sailboats)

K10 §2.5 says inter-flip time is exponentially distributed but gives no rate. Allen 2005 (CG-D-05-05) tabulates per-class jibing frequencies but is paywalled and DTIC blocks non-browser fetches. We pin **4 hr** mean inter-flip globally — mid of the audit's 6-12 hr range, tighter to match BA08 Fig 7 sailboat dynamics where two-lobed search areas converge after ~4-8 hr. Sweep {2, 4, 8, 12} hr. Impact-if-wrong: this is the dominant axis for sailboat search areas. Fast flipping produces a single broad symmetric distribution; slow flipping produces a persistent two-lobed distribution requiring both lobes searched. For PIW the rate matters less. Per-class overrides will be added when A05 is sourced.

### Per-step current and wind perturbation σ (DOCUMENTED-CHOICE / SECONDARY)

K10 §2.5 documents the AR(1) form (PRIMARY) but no σ. BA08 §3.2 reports σ_u ≈ 0.25 m/s and σ_W ≈ 2.6 m/s for 12-h forecast random walks. We adopt **0.20 m/s** current (slightly conservative; SAROPS uses shorter NCOM/HYCOM forecast horizons than BA08's 12-h frame) and **2.6 m/s** wind (BA08 SECONDARY). Sweep current {0.10, 0.20, 0.30}, wind {1.5, 2.6, 4.0}. BA08 §3.2 measured <2% spread difference when wind/current perturbations were off — leeway perturbation dominates dispersion — so σ matters mostly for short horizons (<6 hr) and shallow-water cases.

### AR(1) correlation half-life (PRIMARY)

K10 §2.5: ρ(Δτ)=exp(−αΔτ) with e^(−60α)=1/2, i.e. **60-min half-life**, applied independently to wind. No choice.

### Current-field source / EDS analog (DOCUMENTED-CHOICE — load-bearing)

K10 §1.1 says EDS feeds gridded (u, v) on "appropriate spatial and temporal grids" but does not pin SAROPS to NCOM/HYCOM/NDFD. BAM13 §2: "the rate of expansion of search areas depends intimately on the quality of the forcing." We default to **stationary_uniform** for deterministic benchmarking; sweep includes single_vortex and recorded_hycom_extract. Audit §5 calls EDS quality "the single largest realism axis the comparator should be honest about." Impact-if-wrong: stationary_uniform isolates algorithmic effects but cannot reproduce non-Gaussian posterior tails seen in real shear regions (e.g. Gulf Stream edge). The bench reports conditional on EDS source.

### Aground-particle policy (DOCUMENTED-CHOICE)

K10 silent for SAROPS; FS01 §6.4.1 documents CASP's "temporarily aground" tag. We pin **freeze_at_coastline** as the operationally honest default; sweep over discard and leave_drifting. Discard biases the posterior offshore; leave_drifting is unphysical (particles penetrate land). Rare for open-ocean benchmarks; non-trivial for coastal cases (most US incidents within 25 nm per BA08 Fig 11).

---

## C. Sensor Likelihood

### LRC parametric family (DOCUMENTED-CHOICE — highest leverage)

K10 §2.4 cites the Frost (2004) USCG SAROPS Lateral Range Curves memo, which is non-public. FS01 §2.3.1 sketches three families: definite-range, exponential, inverse-cube/erf. We pin **exponential** `λ(x) = 1 − exp(−W/(2|x| + ε))` because (a) FS01 establishes exponential coverage as the Planner's initial-placement detection function, (b) it is single-parameter (W matches the Koopman sweep-width definition directly), and (c) it is the Stone-Koopman canonical form. Sweep includes inverse_cube, definite_range, koopman_erf. Audit §5 explicitly flags this as the highest-leverage implementation choice: "Inverse-cube vs exponential changes posterior shape after a search by ~10–20% in tail mass." This sweep is mandatory; without it, comparator results in the regime "alternative architecture matches SAROPS posterior" can be accused of LRC-tuning fabrication.

### Sweep-width tables (DOCUMENTED-CHOICE)

Numeric W per (sensor, target, sea-state) lives in CGTO Pub 3-50.1 / NSS Addendum / Frost 2004 — operationally calibrated, non-public. We provide a proxy table covering surface vessel, fixed-wing aircraft (500 ft), helicopter (300 ft), and small-EO drone (the bench's primary platform), against five targets (PIW, life raft 4-6, life raft 15-25, small boat <5m, sailboat 8-12m), at three sea-state bands. Values are scaled to FS01 narrative range. Mandatory sweep multiplier {0.5, 0.75, 1.0, 1.5, 2.0}. Impact-if-wrong: a 2× error in W produces approximately 2× error in initial-placement POS via the exponential coverage formula. Bench should not interpret POS-magnitude differences against SAROPS as informative; only POS-shape and decision-stability survive the W proxy.

### Detection function (PRIMARY)

K10 §3.3: exponential coverage `b(C) = 1 − exp(−C)` for initial rectangle placement (FS01 eq. 2-4); particle-CPA + LRC for refinement. Both PRIMARY.

---

## D. Optimal-Effort Allocation

### Allocator algorithm (PRIMARY)

SAROPS' Planner does NOT implement Stone's pointwise theorem directly. K10 §3 specifies a rectangle-constrained heuristic: 5-parameter rectangle, accordion-search initial placement, 12-move perturbative refinement, big-jump escape, 1,500-particle POS subsample. We pin **rectangle_constrained_heuristic_K10** as the faithful default; sweep alternatives are Stone's pointwise Lagrangian (SRW16 Theorem 2.1) and Charnes-Cooper closed form. Comparing pointwise-Stone POS to rectangle-Planner POS quantifies the heuristic's optimality gap.

### The 12 perturbation moves (DOCUMENTED-CHOICE)

K10 §3 says "12 move types" but does not enumerate. We pick a defensible 12-move set: ±N/E/S/W translation (0.5 nm), ±length (0.5 nm), ±width (0.25 nm), ±15° rotation, flip turn direction, swap ell/w. Sweep step-size multipliers {0.5, 1.0, 2.0}. Impact-if-wrong: smaller steps slow convergence within wall-clock budget; larger overshoot. Matching K10's exact moves is impossible from the open record.

### Wall-clock budget (DOCUMENTED-CHOICE)

K10 §3 mentions wall-clock budget without numeric. Per audit §3 item 11, range 5-60 s. We default **30 s**; sweep {5, 15, 30, 60}.

### Minimum track spacing γ (DOCUMENTED-CHOICE)

K10 §3.1 mentions γ as the TSV soft-penalty parameter without numeric. Defaults: 0.10 nm surface (~200 yd), 0.50 nm aircraft, 0.05 nm drone. Sweep multipliers {0.5, 1.0, 2.0}. Too-tight γ exceeds platform turn radius / overlap bias; too-loose wastes effort.

### Effective path factor (PRIMARY)

K10 §3.2: `L = 0.85 × speed × time_on_station`. The 85% is a USCG operational sighting-investigation overhead constant.

### Refinement objective (PRIMARY)

K10 §3.3: exponential-coverage POS for initial placement; particle-CPA POS via K10 eq. 3 for refinement. Both replicated.

### POS subsample (PRIMARY)

K10 §3.3 "Computing POS": 1,500 particles drawn with replacement, adaptively enlarged if SD > 5% of estimated POS.

### Search-cell discretization (DOCUMENTED-CHOICE — Stone-pointwise sweep only)

For the Stone-pointwise sweep alternative, we pick 0.5 nm to match the LRC width scale; sweep {0.25, 0.5, 1.0}. Too-fine increases bisection cost; too-coarse biases the Lagrangian solution toward grid-cell rectangles.

### Multi-sortie time propagation (PRIMARY)

K10 §3.3: "For each subsequent SRU, condition on the posterior given failure of all prior rectangles." Sequential posterior conditioning.

---

## E. Replan Trigger

### Trigger mode (DOCUMENTED-CHOICE — operational adaptation)

Faithful operational SAROPS replans on per-sortie debrief (human-in-the-loop, K10 §1.1). The drone-swarm bench is autonomous, so a fixed cadence is required. Per audit §3 item 14, three options: per-sortie, fixed-cadence, data-driven. We pin **fixed_sortie_length** (60 min drone, 240 min aircraft, 720 min surface). Sweep includes per_sortie_debrief_event (event-driven, faithful to operational practice) and particle_degeneracy_ess_below_half (deviates from K10 but exposes what human-in-loop replan implicitly does). Impact-if-wrong: too-frequent replan churns the posterior; too-infrequent lets stale sortie outcomes persist.

### Auto threshold (PRIMARY)

K10 §1.1: no automatic threshold. Honored — no auto trigger in faithful default.

---

## F. Scenario Ensemble

### Supported scenario types (DOCUMENTED-CHOICE)

K10 §2.1 lists {LKP, Area, Voyage, LKP+DR, LOB}. We default **{LKP, Area}**: LKP is the minimum viable comparator; Area adds spatial uncertainty cheaply via inverse-CDF sampling. Voyage and LOB significantly increase implementation surface (correlation, polygons, hour-glassing prevention). Sweep includes Voyage to support audit §4 fidelity test 4 (hour-glassing test).

### Default n scenarios (DOCUMENTED-CHOICE)

K10 §2 says scenarios are operator-assigned. We default **1** (single LKP); the object-type-discrimination fidelity test (audit §4 test 3) requires 2. Sweep {1, 2, 5}.

### Scenario weight update (PRIMARY)

K10 §2: user-assigned, summing to 1. Weight evolution is implicit — scenario weight emerges as the sum of post-update particle weights inside the scenario, normalized across all scenarios. Confirmed by K10 §2.3 + §2.4. No choice.

### Scenario pruning (PRIMARY null default)

K10 silent. Operator adds/removes scenarios. Faithful = no automatic pruning. Sweep includes auto-prune-below-weight = 0.01 as deviation alternative.

### Hazard support (DOCUMENTED-CHOICE)

K10 §2.2 documents region × time × intensity ∈ {1,3,5,10}, multiplicative for overlapping hazards, piecewise-uniform `f(t)`. Most reproductions skip; we default **OFF** (test scenarios don't use hazards). Mechanism implementable from K10 §2.2 if needed. Intensity set {1,3,5,10} is PRIMARY. Omitting hazards limits ability to represent "vessel passes through storm" cases — acceptable for broadcast-determinism evaluation, flagged limitation for SAR-realism evaluation.

### Object-type set (DOCUMENTED-CHOICE)

OBJECTPROP.DAT has 85 classes. Audit §3 item 17 recommends 3-5. We default to 5: PIW-1 (~0.96% slope, Syx=12), LIFE-RAFT-DB-10 (3.52% slope, Syx=6.1), SKIFF-1 (3.15% slope, tight Syx=2.2), SAILBOAT-1 (4.5% slope, Syx=19.4), FISHING-VESSEL-1 (2.47% slope, mean class Syx=12). Spans the leeway-divergence spectrum from low (PIW) to high (sailboat), wide-Syx mean classes to tight-Syx well-instrumented classes. Default object distribution is coastal-SAR-typical per BA08 Fig 11. Sweep covers 2-class minimal, the 5-class default, and "all_objectprop". A PIW-only comparator cannot exhibit the K10 §2.3 raft-vs-PIW differential reweighting, which is a load-bearing fidelity test (audit §4 test 3).

### Survival hazard (DOCUMENTED-CHOICE)

K10 §1.1 mentions EDS provides water temperature for survival; FS01 §2.5.2 introduces multi-state survivor search. Neither pins a hazard form. Hayward / Eckerson USCG operational tables exist but are not in the open record. We default **none** because audit §5 says: "this is significant only if benchmark scenarios involve > 12 hr searches." Sweep alternatives: exponential decay with per-class half-life (PIW-1 = 4 hr cold-water mortality, raft = 72 hr, etc.) and a placeholder for the USCG NSS Addendum table. Impact-if-wrong: omitting survival weighting on long-horizon (>12 hr) cold-water cases overspreads probability into regions where the operational system has weighted survivors out, inflating apparent search-area requirement.

---

## G. Aggregate Fidelity Posture

The 20 OPERATIONAL-UNKNOWNs are addressed with this distribution: **PRIMARY values pinned (no choice):** particle count default, voyage correlation, leeway tables, AR(1) form and half-life, spatial/temporal interpolation, cross-leg/cross-SRU independence, effective path factor, refinement objective, POS subsample, multi-sortie conditioning, weight-update mechanism, hazard intensity set, scenario-pruning null default, no auto-replan threshold. **SECONDARY:** wind perturbation σ (BA08). **INFERRED:** proposal distribution = bootstrap. **DOCUMENTED-CHOICE with mandatory sweeps:** resampling policy, time-step granularity, Rayleigh switch threshold, crosswind sign-flip rate, current σ, EDS source, aground policy, LRC family, sweep-width table, the 12 moves, wall-clock budget, min track spacing, cell discretization, replan mode and sortie length, supported scenario types, default n scenarios, hazard on/off, object-type set, default object distribution, survival model.

The required sweep axes (`sweep_orchestration.required_sweep_axes` in YAML) mirror audit §5's robustness criterion: "if conclusions are robust to (LRC family ∈ {inverse-cube, exponential}, leeway σ ∈ [0.5×, 2×], resample ∈ {no, SIR}, step ∈ {15, 60} min), the comparison is meaningful; if not, the comparator is calibrated to its own choices, not to SAROPS." Bench-published results must include sensitivity reports on these axes; conclusions that reverse under any sweep are calibration artifacts, not SAROPS-class evidence.

The artifact label is **"SAROPS-class comparator (Stone-Kratzke-Allen; documented choices)"** per audit §5. Binding nomenclature — no comparator output should be labeled "SAROPS reproduction" or "SAROPS-equivalent."

---

## H. Open Items for Phase C Implementation

1. **Allen 2005 (CG-D-05-05) acquisition.** If sourced, populate `drift.crosswind_jibing.per_class_overrides` and tighten the global default. Until then, the global 4 hr is a load-bearing DOCUMENTED-CHOICE.
2. **Frost 2004 SAROPS LRC memo.** Non-public; no expected near-term resolution. The LRC-family + sweep-width-proxy strategy is the structural workaround.
3. **HYCOM extract pin.** If recorded_hycom_extract sweeps are reported, the region/date must be pinned in `drift.environmental_data_source.pinned_extract_path` for reproducibility.
4. **Survival table.** If long-horizon (>12 hr) cold-water benchmarks are added, the USCG NSS Addendum survival table must be sourced or a defensible exponential-decay proxy defended in print.
5. **Reporting template.** Phase C bench output must follow `sweep_orchestration.reporting_rule`: pinned values + sweep axes + per-axis deltas + robustness verdict, formatted as in audit §5.
