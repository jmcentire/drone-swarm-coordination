# SAROPS Lineage — Operational Bayesian Search Audit

**Audit date:** 2026-05-08
**Author:** Jeremy McEntire (with Claude Opus 4.7 1M)
**Purpose:** Decompose the open-literature description of the U.S. Coast Guard's Search and Rescue Optimal Planning System (SAROPS) into algorithmic components sufficient to build a "SAROPS-class" comparator. This is **not** a faithful reproduction; the comparator is faithful only to the open record, and several operational choices remain undocumented.

**Sources accessed in full text:**

| # | Source | Status | Where |
|---|---|---|---|
| K10 | Kratzke, T. M., Stone, L. D., Frost, J. R. (2010). *Search and Rescue Optimal Planning System.* FUSION 2010. | Full text | DTIC ADA564779 (8 pp.) |
| FS01 | Frost, J. R. & Stone, L. D. (2001). *Review of Search Theory: Advances and Applications to Search and Rescue Decision Support.* USCG R&D Center CG-D-15-01. | Full text | DTIC ADA397065 (~5,900 lines) |
| BAM13 | Breivik, Ø., Allen, A. A., Maisondieu, C., Olagnon, M. (2013). *Advances in search and rescue at sea.* Ocean Dynamics 63. | Full text (preprint) | arXiv:1211.0805 (11 pp.) |
| SRW16 | Stone, L. D., Royset, J. O., Washburn, A. R. (2016). *Optimal Search for Moving Targets.* Springer ISORMS 237. | Full text | ETH NDL mirror (141 pp.) |
| S07 | Stone, L. D. (2007/1975). *Theory of Optimal Search,* INFORMS reissue. | Cited via SRW16 §2.3.1.1 | Not directly accessed |
| K46/K57 | Koopman, B. O. (1946; 1956–57). *Search and Screening.* | Cited via FS01 §2.3.1, §2.4.1 | Not directly accessed |

**Sources NOT accessed:** the original Stone (1975/2007) monograph (Lagrangian / pointwise theorem text reached via SRW16 §2.3.1.1, which explicitly attributes the theorem to Everett 1963 with Stone (2007) Ch. II showing necessity); Koopman (1946, 1956–57) primary papers; the Allen & Plourde (1999) USCG Leeway report (technical CG-D-08-99); the SAROPS Lateral Range Curves memo (Frost, USCG G-OPR-1, 11 Mar 2004) explicitly cited as ref [3] of K10 — paywalled / non-public.

Claim-strength tags: **PRIMARY** (cited from originating paper), **SECONDARY** (from a follow-up by the same lineage), **INFERRED** (from open literature plus reasonable assumption), **OPERATIONAL-UNKNOWN** (load-bearing, not in the open record).

---

## 1. Six-Component Algorithm Summary

SAROPS is composed of four named subsystems: **EDS** (Environmental Data Server), **SIM** (simulator/particle filter), **Planner** (effort allocator), **GUI** (K10 §1.1). The six audit components below cut across these.

### 1.1 Drift Propagation — PRIMARY (largely)

**State.** Each particle carries: (lat, lon, time), p-fail (cumulative non-detection probability), last analysis-data position (and p-fail there), status ∈ {drifting, temporarily aground, aground, underway}, target ID, location ID, situation ID (FS01 §6.4.1, p. 6-23). K10 §2.5 confirms each particle = a possible path, with weight, for the search object; particles move continuously through space and time, only the path-space is discrete (K10 §2).

**Velocity decomposition.** Drift velocity = current vector + leeway vector (FS01 §5.2.1, K10 §2.5).

- **Currents.** From EDS gridded `(u, v)` fields. SIM uses inverse-distance weighting over the three nearest grid points in space, plus linear interpolation in time between the two bracketing forecast frames (K10 §2.5, p. 5). On top of the deterministic field, SIM adds a **first-order autoregressive perturbation**: a per-particle, per-time-step normal draw to `u`, `v`, with temporal correlation `ρ(Δτ) = exp(−αΔτ)` calibrated so `e^(−α·60) = 1/2` (Δτ in minutes — i.e. 60-minute correlation half-life) (K10 §2.5, eq. on p. 5). The same OU-style structure is applied independently to wind components.
- **Leeway.** K10 §2.5 cites USCG NSS Addendum (IAMSAR) Appendix H for the leeway model. Decomposition into **downwind** (DWL) and **crosswind** (CWL) components. DWL slope `ν` is per-particle (drawn once per particle, **not** redrawn per step). Two methods: (i) **Standard:** `ν = m − zσ/q` with `z ~ N(0,1)`, where `m` = mean slope at nominal wind speed `q`, `σ` = standard deviation of slope; (ii) **Rayleigh:** for object types with very small `m` (would otherwise produce negative slopes), `r = m·√(2/π)·R` with `R ~ Rayleigh`. Crosswind: same standard form with its own `(m_CW, σ_CW)`. The crosswind sign **flips** stochastically: time between sign flips is exponentially distributed (jibing model). Per-step leeway speed = `slope × wind_speed` (K10 §2.5, p. 5–6).
- **Time step.** SIM updates position to "the next whole hour in simulated time, or the desired probability-map time" (FS01 §6.4.1 step 7, p. 6-23). The K10 paper does not state an explicit step size; one secondary article cited in WebSearch results paraphrases "every 20 minutes." **Open-literature consensus: hourly step in CASP, with finer resolution in SAROPS — exact step is OPERATIONAL-UNKNOWN.**

**Reproduction-relevant gaps:**
- Specific values of `(m, σ, q)` per object class — these live in the cited "SAROPS LATERAL RANGE CURVES" memo and in the USCG leeway taxonomy (Allen & Plourde 1999; Allen 2005 CG-D-05-05). Allen's leeway taxonomy enumerates ~63 object classes; values are tabulated in those reports (BAM13 §2 confirms the catalog became operational in 2001 in HMI's Norwegian system).
- The exact rate of crosswind sign-flips: K10 says "exponentially distributed" but does not give the mean. **OPERATIONAL-UNKNOWN.** Allen (2005) reports jibing frequencies per object class.
- Treatment of wave-induced drift / Stokes drift: BAM13 §2 explicitly notes that most SAR trajectory models, including SAROPS, ignore direct wave excitation; this is acknowledged as a missing process.

### 1.2 Particle Filter — PRIMARY

**Particle count.** User-selectable per scenario: **2,500 / 5,000 / 10,000** (K10 §2, p. 3). Total particle population = (per-scenario count) × (number of scenarios). FS01 §6.4.1 notes CASP supported up to 20,000 reps per situation; SAROPS narrowed this.

**Initialization.** Each scenario draws particles from a bivariate prior:
- *LKP scenario:* bivariate normal in space, normal in distress time. Each particle = one position draw + one time draw (K10 §2.1, p. 3).
- *Area scenario:* uniform over a polygon, using inverse-CDF (`cdf_x⁻¹`, `cdf_{y|x}⁻¹`) to avoid the rejection-sampling cost when the polygon is a small fraction of its bounding rectangle (K10 §2.1, p. 3–4).
- *Voyage scenario:* sample two correlated points from defined polygons / circles, connected to define a track, with correlation parameter (default `0.7`, range `−1..1`) controlling "starts-to-the-right-stays-to-the-right" to prevent hour-glassing (K10 §2.1, p. 4).
- *LKP+DR* and *line-of-bearing (LOB)* scenarios mentioned as additional types in SIM v1.3.0.0 (K10 §2.1, p. 4).

**Motion update.** See §1.1 above. Each particle is propagated independently along a stochastic drift; particles within one scenario share the same scenario weight.

**Weight update (Bayesian negative-information update).** For each SRU search increment:
- Decompose the SRU track into legs (between waypoints).
- For each particle `p`, compute `d_k` = closest point of approach (CPA) on leg `k`.
- The Lateral Range Curve `λ(d)` gives single-leg detection probability. Per-leg detections are treated **independent** so:
  `pfail(p, sru) = ∏_{k=1..K} (1 − λ(d_k))` (K10 eq. 1)
  `pfail(p) = ∏_{sru} pfail(p, sru)` (K10 eq. 2)
- Posterior weight: `w_post(p) ∝ w_prior(p) · pfail(p)` (K10 §2.4, p. 4–5; also Bayesian-fashion language repeated in §2). Renormalize across all particles (across all scenarios — the scenarios share a common normalization since they form a single mixture over the prior).

**Resampling — OPERATIONAL-UNKNOWN.** K10 does **not** describe an SIR-style resample, ESS threshold, or stratified-resampling step. The paper's text reads as a pure "weighted-particle Bayes update" with no degeneracy-control resample. SRW16 §4.2 references the SAROPS particle population as path samples (not weighted-particle posterior samples in the Doucet–Crisan sense). FS01 §6.4.1 also describes weight updates via `p-fail` tags without resampling. **Inference:** SAROPS appears to use **importance weighting without resampling** for as long as the case is open, relying on the per-scenario population (≤ 10k) and replenishment via case extension or new scenarios to avoid degeneracy. **A comparator-builder must decide:** (a) replicate this — no resample; or (b) add SIR with ESS threshold (the standard PF practice).

**Proposal distribution — INFERRED.** Particles are propagated under the prior dynamics (drift + leeway); no observation-conditioned proposal is used (i.e. it is a bootstrap / SIS filter, not an auxiliary PF). This is consistent with the Bayes update being applied retroactively per sortie via `pfail`.

### 1.3 Sensor Likelihood — PRIMARY (form), OPERATIONAL-UNKNOWN (parameters)

**Lateral Range Curve (LRC).** `λ(x)` = probability of detection as a function of CPA `x` on a long straight search leg (K10 §2.4, p. 4). Each (SRU, object-type, environmental-condition) combination has its own LRC. The functional form referenced by K10 [ref 3] is the USCG SAROPS Lateral Range Curves memo (Frost 2004) — **not in open literature.**

**Sweep width `W`.** Defined (FS01 §2.3.1, p. 2-4, eq. 2-2) as `W = ∫ λ(x) dx`, equivalent to twice the maximum detection range of the "definite-range" curve that yields the same average detection rate. `W` depends on (object characteristics, sensor capability, environmental conditions: visibility, sea state, cloud cover, sun, rain, etc.) (K10 §1.1; FS01 §2.3.1).

**Search effort definition.** From Koopman:
- *Effort:* total searcher track length `L = v·t`.
- *Search effort / area swept:* `Z = W·v·t`.
- *Coverage / effort density:* `C = Z / A` for region `A` (FS01 §2.3.1–§2.3.2, p. 2-4 to 2-6).

**Detection function in the Planner (initial placement).** SAROPS uses the **exponential** detection function `b(C) = 1 − exp(−C)` (FS01 eq. 2-4, p. 2-5) when computing initial rectangle placements: ratio of swept-area to rectangle-area enters the exponential to give `POD`; `POS = POD × POC` (K10 §3.3 step 1, p. 7). For refinement (steps 2–3), SAROPS uses the **particle-CPA / LRC** computation directly (K10 §3.3 "Computing POS," p. 8).

**LRC family — OPERATIONAL-UNKNOWN.** Open literature sketches plausible parametric forms: (i) **inverse-cube / `erf`** for visual detection: `λ_IC corresponds to P(C) = erf(√(π/2)·C)` (FS01 eq. 2-7, p. 2-9); (ii) **exponential** `λ_exp(x) = 1 − exp(−W/(2|x|))` style; (iii) **definite-range** binary. K10 doesn't say which. **The comparator-builder must pick one** and pick numeric `W` per (sensor, object, condition). FS01 §3 mentions SARP's computed sweep widths from object length, cloud cover, visibility, altitude (FS01 p. 6-4); SARP-era tables are likely the operational basis but are not transcribed in the open record.

### 1.4 Optimal-Effort Allocation — PRIMARY (theory), SECONDARY (operationalization), OPERATIONAL-UNKNOWN (heuristic constants)

**Underlying theorem.** Stone's pointwise / Lagrangian optimization theorem (SRW16 Theorem 2.1, p. 21, attributed to Everett 1963 with necessity by Stone 2007 Ch. II): for cells `j = 1..J` with prior `p(j)`, cost `c(j)`, decreasing-rate detection function `b`, the optimal allocation `f*` satisfies
  `b'(j, f*(j)) · p(j) / c(j) = λ if f*(j) > 0, ≤ λ if f*(j) = 0`,
for some Lagrange multiplier `λ > 0`. The total cost `C(λ)` is decreasing in `λ`, so a 1-D bisection yields the `λ` matching the budget. For the exponential detection function this reduces to the closed-form Charnes–Cooper allocation (FS01 §2.4.2, p. 2-10; original Charnes & Cooper 1958).

**SAROPS' Planner does NOT implement Stone's theorem directly.** Instead, the Planner solves a **constrained heuristic** (K10 §3, p. 6–9):

1. **Configurations are rectangles**, not cells. Five params per rectangle: center (2), `ell` (length parallel to search legs), `w` (signed width — sign encodes turn direction), `θ` (orientation). Path induced by rectangle uses `s` (track-spacing) and `t` (search-leg length); `w = L·s/(t+s)` where `L = 0.85 × speed × time-on-station` (the 85% factor is a USCG operational constant for sighting-investigation overhead; K10 §3.2, p. 7).
2. **Constraints (Table 3.1, K10):** parallel-track pattern inside the rectangle; total leg length = available path length minus one track-spacing; rectangles for same SRU class don't overlap; track-spacing ≤ search-leg length; track-spacing ≥ a minimum `γ`.
3. **Initial placement (Step 1):** seed at the cell with highest prior probability; "accordion search" (greedy add/remove rows/columns) on the rectangle to maximize `POS = POC × (1 − exp(−swept_area/area))` (the closed-form exponential approximation). For each subsequent SRU, condition on the posterior given failure of all prior rectangles.
4. **Refinement (Steps 2–3):** 12 "moves" perturb each rectangle individually. Step 2 reduces overlap minimizing POS loss; Step 3 increases POS minimizing overlap gain. Refinement uses the **particle-based POS** (K10 eq. 3) — `POS = Σ_p w(p)·POS(p)` — not the closed-form exponential.
5. **Big-jump escape (Step 5):** pick a random rectangle, translate it to the cell with the highest posterior given failure of all the *other* rectangles, accordion-search around it.
6. **Stopping:** runs until a wall-clock budget elapses or the user terminates. Reports best solution and POS statistics.

**Sampling shortcut.** Step-2/3 POS is estimated from a fixed sub-sample of **1,500 particles** drawn with replacement from the full population, with the variance monitored: if SD > 5% of estimated POS, the sample is enlarged (K10 §3.3 "Computing POS," p. 8). This typically reduces particle work by 10×. **Trade-off explicitly noted:** the same particle sample is reused across moves — biases POS upward — but this is the price of having a deterministic objective for the local-search optimizer.

**Track Spacing Violation (TSV):** soft penalty when track-spacing < `γ` (K10 §3.1, p. 6–7).

**Operational constants — OPERATIONAL-UNKNOWN:**
- The 12 move types (translation deltas, rotation increments, ell/w deltas) — not enumerated in K10.
- Wall-clock time budget per Planner run.
- The minimum track-spacing `γ` (presumably per-SRU, per-altitude, per-sensor).
- The 85% effective-path factor is given; whether other "Coast Guard factors" intrude (e.g. transit penalties, en-route detection bonuses) — unstated.

### 1.5 Replan Trigger — PRIMARY (cycle), OPERATIONAL-UNKNOWN (operational cadence)

The cycle described in K10 §1.1 (p. 1–2):

1. SIM produces probability distribution at the planned commence-search time.
2. Planner ingests this + SRU availability/capability and produces operationally feasible search plans maximizing the increase in POS.
3. Search executes.
4. **If unsuccessful:** SIM produces a posterior probability map accounting for the unsuccessful search and continued object motion. The posterior becomes prior for the next plan.

**Trigger conditions:**
- Per-sortie debrief (search completes → Bayesian update → replan for next sortie). This is the dominant operational cadence.
- New environmental data ingested via EDS → re-propagation of particles.
- New scenario information (new witness statement, new EPIRB hit, vessel reported missing) → new scenario added or weights changed → re-propagation.
- Operator-initiated replan.

**No internal "trigger threshold" is documented.** The operator decides; the system replans on demand. There is no published evidence of, e.g., "automatic replan if ESS < threshold" or "if posterior mass shifts by > X% out of search box." **Inference:** the trigger is human-in-the-loop, and the open literature does not specify any automatic data-driven trigger. Comparator must choose: (a) per-sortie replan (faithful to operational use); (b) per-environmental-update replan; (c) particle-degeneracy-driven replan (which would deviate from documented SAROPS behavior).

### 1.6 Scenario-Ensemble Management — PRIMARY (mechanism), INFERRED (specifics), OPERATIONAL-UNKNOWN (operator practice)

This is the part of SAROPS that most resists clean decomposition and is also the most operationally distinctive.

**Hypothesis space.** Multiple "scenarios" form a discrete mixture over what may have happened. Each scenario is one of {LKP, Area, Voyage, LKP+DR, LOB} (K10 §2.1) with its own parameters: starting position uncertainty, distress time uncertainty, hazards encountered en route, intent (track segments), and *post-distress object-type distribution* (K10 §2.3, p. 4).

**Scenario weights.** User-assigned, summing to 1 (K10 §2 p. 3, "Each scenario is given a weight by the user and the weights add to 1.0"; FS01 §6.4.1 p. 6-25 confirms situations are weighted by operator assessment in CASP). Within a scenario, particles share the scenario weight; particles form the conditional-on-scenario distribution.

**Hazards and time-of-distress.** Hazards are user-defined (region, effective time interval, intensity ∈ {1, 3, 5, 10}). For a scenario with pre-distress motion, the time of distress is drawn with density `f(t) = a` outside a hazard ∩ effective interval, `f(t) = κ·a` inside; intensities multiply for overlapping hazards (K10 §2.2, p. 4, eq. for f(t)).

**Object-type distribution.** Per scenario, an object-type distribution applies at distress-incident time (e.g. {person-in-water 0.6, life-raft 0.3, debris 0.1}). Object-type determines (a) leeway parameters and (b) LRC for any SRU. Object-type is treated as **independent** of distress-time (K10 §2.3, p. 4). Each particle is tagged with its object type at distress.

**Survival times.** Water temperature drives survival (K10 §1.1 mentions EDS provides water temperature for survival estimates; FS01 §2.5.2 introduces multi-state survivor search and §6.4.1 notes adding survivability would require an extra rep tag). **K10 does not give a survival hazard function.** SAROPS evidently weights particles by survival probability (otherwise "survivor search" is meaningless), but the form is not in K10. SECONDARY/INFERRED: a piecewise-linear or exponential decay in `(water_temp, exposure_state)` is the typical USCG operational table (Hayward, Eckerson — cited in older USCG manuals); the comparator must pick one.

**Negative-information reweighting.** This is the operational core. After an unsuccessful search:
- `pfail(p)` is computed per particle from CPA + LRC (§1.2 above).
- Particle weight `w(p) ← w(p) · pfail(p)`, then renormalize across **all particles in all scenarios jointly** (K10 §2 p. 3 implies single mixture, FS01 §6.4.1 confirms).
- Because LRCs differ by object type, a search through "raft territory" reweights raft particles much more strongly than person-in-water particles (K10 §2.3 explicit example, p. 4: "an unsuccessful search will affect the probability distribution far more in the case of a raft than … a person in the water because rafts are much easier to detect"). Consequently, **scenarios whose particles are predominantly in well-searched regions, or are predominantly easy-to-detect object types, lose mass.** Scenario-level weight is *implicit* — not directly updated; it emerges from particle weights.

**Combination across scenarios for display and Planner input.** The full posterior on object location is the renormalized sum over all particles in all scenarios (K10 §2). The Planner consumes this single distribution.

**Gaps in the open record:**
- Whether SIM ever **adds** scenarios mid-case (e.g. when initial scenarios have all collapsed under negative information). Operationally yes (operator adds them) but no automatic mechanism is published.
- Whether scenario weights are **explicitly** re-displayed or only the marginal location distribution. K10 doesn't say.
- Survival weighting form: not specified.
- How "temporarily aground" particles re-propagate when forecast → analysis data replaces (FS01 §6.4.1 says they're re-computed, but K10 is silent on SAROPS' policy).

---

## 2. Per-Component Citation Table

| Component | Open-literature spec | Implementation choice required | Citation strength |
|---|---|---|---|
| **Drift (currents)** | EDS gridded `(u,v)`; IDW over 3 nearest grid points; linear in time; per-step Gaussian perturbation with 60-min correlation half-life (K10 §2.5) | Step size (K10 silent; CASP=hourly, modern SAROPS finer); spatial grid resolution; perturbation `σ` per object class | PRIMARY (form), OPERATIONAL-UNKNOWN (numerics) |
| **Drift (leeway)** | DWL/CWL decomposition; per-particle slope drawn once via Gaussian or Rayleigh; CWL sign flips with exponential inter-flip time (K10 §2.5) | `(m, σ, q)` per object class — in Allen 2005 / USCG Allen taxonomy; flip rate; choice of standard vs Rayleigh per object | PRIMARY (form), OPERATIONAL-UNKNOWN (parameters) |
| **Particle filter — count** | 2,500 / 5,000 / 10,000 per scenario, user-selected (K10 §2) | Choice; typically 5–10 k for hard cases | PRIMARY |
| **Particle filter — init** | Inverse-CDF for polygons; bivariate normal for LKP; correlated draws for voyage with parameter ≈0.7 (K10 §2.1) | Voyage correlation parameter; how many polygon vertices per voyage region | PRIMARY (form), SECONDARY (default 0.7) |
| **Particle filter — weights** | `w_post ∝ w_prior · ∏ (1 − λ(d_k))` per leg per SRU (K10 eq. 1–2) | None for the form | PRIMARY |
| **Particle filter — resampling** | None documented (K10 silent; FS01 silent) | Whether to add SIR (faithful = no; statistically prudent = yes with ESS trigger) | OPERATIONAL-UNKNOWN |
| **Sensor — LRC form** | Family choice from FS01 (definite, exponential, inverse-cube/erf); SAROPS uses operationally tuned LRCs per (SRU, target, conditions) (K10 §2.4 ref 3) | Pick a parametric family; pick numeric `W`; pick environmental modifiers | PRIMARY (Stone-Koopman framework), OPERATIONAL-UNKNOWN (numeric LRC) |
| **Sensor — sweep width** | `W = ∫ λ(x) dx`; depends on visibility, sea state, cloud, sun, target type (FS01 §2.3.1) | Numeric values | PRIMARY (definition), OPERATIONAL-UNKNOWN (table) |
| **Effort allocation — theorem** | Stone/Everett pointwise Lagrangian; `b'·p/c = λ`; bisect on `λ` for budget (SRW16 Theorem 2.1; FS01 §2.4.2) | Discretization grid; convergence tolerance | PRIMARY |
| **Effort allocation — Planner** | Rectangle-constrained heuristic; accordion init; 12-move perturbative refinement; 1500-particle subsample for POS estimation (K10 §3) | The 12 moves, the move-sizes; the time budget; `γ` (min track-spacing) | PRIMARY (algorithm shape), OPERATIONAL-UNKNOWN (constants) |
| **Replan trigger** | Per-sortie debrief → Bayesian update → re-plan; operator-initiated (K10 §1.1) | Whether to add automatic triggers (deviates from open record) | PRIMARY (cycle), OPERATIONAL-UNKNOWN (auto-triggers) |
| **Scenario ensemble — types** | LKP, Area, Voyage, LKP+DR, LOB (K10 §2.1) | Which types to support | PRIMARY |
| **Scenario ensemble — weights** | User-assigned, sum to 1; particles inherit; no explicit Bayesian update on scenario weight (K10 §2) | Default scenarios for a comparator (e.g. just LKP) | PRIMARY |
| **Scenario ensemble — hazards** | Region × time × intensity ∈ {1,3,5,10}; multiplicative; piecewise-uniform `f(t)` (K10 §2.2 eq.) | Whether to support hazards (skipping reduces fidelity for "vessel passes through storm" cases) | PRIMARY |
| **Scenario ensemble — object types** | Per-scenario discrete distribution; independent of distress time; drives leeway and LRC (K10 §2.3) | Number of object types to model | PRIMARY |
| **Scenario ensemble — survival** | Water-temp dependent; affects particle weight over time (K10 §1.1 EDS, FS01 §2.5.2) | Survival hazard function form & parameters | INFERRED + OPERATIONAL-UNKNOWN |
| **Scenario ensemble — neg-info reweight** | All particles across all scenarios share normalization; scenario weight drift is implicit (K10 §2 + §2.4) | None for the form | PRIMARY |

---

## 3. OPERATIONAL-UNKNOWN Items (Comparator-Builder Decision List)

The comparator implementer **must** pick a value or strategy for each. Document the choice and label clearly: a comparator that decides 20 of these is no longer "SAROPS-class," it is "our interpretation of SAROPS."

1. **Drift integration step.** Pick: 60-min (CASP-faithful), 20-min (often-cited but unverified), or 5-min (PF-conservative).
2. **Per-step current/wind perturbation σ.** No public value. Calibrate against drifter-buoy RMSE if a reference dataset is used; otherwise pick 0.1–0.3 of nominal.
3. **Crosswind sign-flip rate λ_flip.** No public value. Allen (2005) tabulates jibing frequencies per object class; absent that, pick 1 flip per 6–12 hr for raft-like objects.
4. **Standard vs Rayleigh DWL slope.** Pick threshold on `m` (e.g. switch to Rayleigh if `m < 0.02`).
5. **`(m, σ, q)` leeway tables.** Pull from Allen (2005) CG-D-05-05 if the implementer can locate it; otherwise pick 3 representative classes (PIW, raft, sailboat).
6. **PF resampling policy.** Choose: none (K10-faithful) or SIR with ESS threshold = N/2 (PF-canonical).
7. **LRC parametric family.** Choose: exponential `λ(x) = exp(−2|x|/W)`, inverse-cube `λ(x) = 1/(1+(x/r₀)²)`-style, or Koopman erf form. Pick `W` per (sensor, target, env).
8. **Sweep-width modifier table.** Visibility → `W` mapping. FS01 cites SARP's formula (object length × cloud cover × visibility × altitude) — operationally calibrated tables exist in USCG Addendum to NSS / IAMSAR Appendix H.
9. **Planner objective during refinement.** Faithful = particle-CPA POS; simplification = exponential-coverage POS. K10 uses both at different stages.
10. **Number of moves & move sizes.** K10 says "12 types"; not enumerated. Choose: ±cell translations, ±row/col additions, ±θ rotations, ell/w swaps, etc.
11. **Planner wall-clock budget.** Operator-set; pick 5–60 s for benchmarking.
12. **Min track-spacing `γ`.** Per-SRU, per-altitude, per-sensor. Use Coast Guard ship-handling factors or pick one constant (e.g. 200 yd for surface, 0.5 NM for air).
13. **POS-subsample size.** K10 says 1500 with 5% SD adaptive. Use as is.
14. **Replan cadence.** Faithful = per-sortie debrief. For autonomous benchmarking, pick a fixed sortie length (e.g. 4 hr aircraft, 12 hr surface).
15. **Scenario types supported.** Minimum LKP. Add Area for spatial uncertainty in LKP. Voyage and LOB are operationally important but increase implementation surface significantly.
16. **Hazard support.** Faithful = supported. Most reproductions skip hazards (justifiable if test scenarios don't use them).
17. **Object-type set.** Faithful = ~63 USCG classes. Practical reproductions: 3–5 (PIW, raft, small-boat, sailboat).
18. **Survival hazard.** Faithful = water-temp-table. Practical: exponential decay with class-specific half-life, or no survival modeling for short benchmarks.
19. **Aground-particle policy.** Discard, freeze, or leave drifting against coast. K10 silent for SAROPS; FS01 §6.4.1 documents CASP's "temporarily aground" tag.
20. **EDS analog.** What environmental field? Faithful = NOAA/NCEP/HYCOM grids. Reproductions: stationary uniform field, single-vortex, or pre-recorded HYCOM extract for a region/date.

---

## 4. Validation Strategy: Published Case Studies for Reproduction-Fidelity Check

**Direct SAROPS case studies in open literature (as case studies, not raw reconstruction data):**

- **Kratzke et al. 2010 (K10) Figures 2.1, 3.3.** Voyage scenario distribution and a Planner output figure are illustrative but no input parameters, no detailed scenario JSON, no environmental data slice — useful for *qualitative* output-shape comparison only. Specifically: (a) does a voyage scenario produce hour-glassing if the correlation parameter is set to 0? (b) Does an LKP scenario seeded with a bivariate normal produce a posterior that looks like K10 Fig. 3.3 after one accordion-search rectangle? — those are qualitative checks the comparator should pass.
- **Kratzke et al. 2010 §1, p. 1:** "SAROPS has been operational since January, 2007" — historical case studies (Apostolopoulos vessel, El Faro, missing fishing vessels) appear in USCG annual reports and press; none give input parameters.
- **PBS LearningMedia / SIAM News popular-science articles** describe the *F/V Lady Mary* (2009) and the *Aug 2007 sailboat off Long Island* — narrative only, no recoverable inputs.
- **Frost & Stone 2001 §4.2 figures (4-10a–4-10f, p. 4-26 to 4-31)** show CASP-style Monte Carlo distributions for hypothetical scenarios. These are useful to qualitatively check that one's particle filter produces non-circular, non-Gaussian posteriors that follow the current/wind structure.

**Indirect / proxy case studies useful for cross-check:**

- **Air France 447 search (2009–2011).** Stone et al. (2014, J. Naval Eng.) "Search for the Wreckage of Air France 447" reports the Bayesian-search reconstruction. Not strictly SAROPS, but same lineage (Metron, Stone). Has a published prior, posterior, and search history. Explicit numeric outputs at multiple iterations.
- **USS Scorpion search (1968).** Original CASP precursor; documented in Richardson & Stone 1971 (Naval Research Logistics) and Stone 1992 (Operations Research). Useful as a historical particle-filter Bayesian-search benchmark.
- **MH370 search.** Stone et al. (2015) Statistical Science. Particle-filter Bayesian search again from the same group; published prior, posteriors, and effort allocation.

**Recommended fidelity tests for a SAROPS-class comparator:**

1. **LKP-only sanity:** seed one bivariate normal LKP, propagate under a constant uniform current + zero wind for 24 hr → posterior should remain bivariate normal with mean shifted by (current × 24 hr) and variance grown by the per-step perturbation σ²·N. (Closed-form check.)
2. **Negative-information shrinkage:** fly one parallel-track search through the LKP center → posterior probability mass inside the searched rectangle should decrease by factor ≈ exp(−swept_area/area), and the "hole" should be visible in the displayed posterior. (FS01 Fig. 2-3 / Fig. 2-5 visual check.)
3. **Object-type discrimination:** seed two scenarios, raft 50% / PIW 50%, with different leeway slopes; after a search through the wind-downstream raft territory, the raft scenario weight (= summed particle weight) should drop substantially while the PIW scenario weight stays near 0.5. (K10 §2.3 prediction.)
4. **Voyage hour-glassing test:** seed a voyage scenario between two regions with correlation=0 vs correlation=0.7 → check that correlation=0 produces the K10 Fig. 2.2 hour-glass and 0.7 does not.
5. **Hazard time-of-distress shift:** seed a voyage scenario with one hazard intensity κ=10 over a small region → distress-time density should be ~10× higher inside that segment per K10 eq. (`f(t)` piecewise).
6. **Planner accordion search:** initialize with a circular normal, request one SRU rectangle → recovered rectangle should have center at the prior mode and aspect ratio approximately matching the optimal-coverage circle of FS01 Fig. 2-2 for that effort budget.

These tests are **qualitative-quantitative reproduction-fidelity checks**, not absolute-accuracy benchmarks.

---

## 5. Honest Gap Statement

Open-literature SAROPS coverage is **structurally complete but numerically thin.** A practitioner can implement every named subsystem with the right *shape*; almost every numeric parameter that controls the *quality* of the result is in operational documents not in the public record. Specifically:

**Strong (well documented):**
- Particle filter as a weighted-particle Bayesian update over Monte Carlo drift trajectories.
- Lagrangian / pointwise optimal-allocation theory.
- Scenario-mixture structure; hazard mechanism; object-type independence assumption.
- Bayes update via per-leg LRC + CPA; cross-SRU independence.
- Planner heuristic shape: rectangle parameterization, accordion init, perturbative refinement, big-jump escape, 1500-particle POS subsample with adaptive variance check.

**Medium (described conceptually but not numerically):**
- Time step of the SIM (CASP-era hourly is documented; SAROPS finer step is asserted, not specified).
- Crosswind sign-flip exponential rate.
- Standard-vs-Rayleigh DWL switch threshold.
- Survival/water-temp model.
- Specific aground-particle policy (mentioned for CASP, not SAROPS).

**Weak / not in open record:**
- Numeric LRC tables — Frost (2004) memo, USCG-internal.
- Numeric leeway slopes (m, σ, q) per object class — Allen (2005) report.
- The 12 Planner moves and their parameters.
- Planner wall-clock budget and convergence criteria.
- Resampling policy (apparently none, but unconfirmed).
- Whether SAROPS does any scenario-level weight update beyond the implicit one through particle weights.
- Operator workflow detail: how often human operators add new scenarios mid-case; how operators set hazard intensities (the {1,3,5,10} discrete set is documented but its calibration to real-world distress-rate increases is not).

**Could matter for fidelity:**
- Survival hazard form. For long-duration cases (>24 hr, water temp <15 °C), the posterior is dominated by survival weighting. A comparator that omits survival weighting will overspread probability into regions where the operational system has already weighted survivors out. For the drone-swarm comparator in this repository, this is significant only if benchmark scenarios involve > 12 hr searches.
- The non-resampling choice. If the comparator runs many sortie iterations, a no-resample SIS filter degenerates (effective sample size collapses). Either the operational SAROPS workflow re-seeds via new scenarios faster than degeneracy occurs, or there is an undocumented resample. This is a real fidelity question with no public answer.
- LRC family choice. Inverse-cube vs exponential changes posterior shape after a search by ~10–20% in tail mass. If the comparator's purpose is to demonstrate that an alternative architecture matches SAROPS' posterior, the LRC family is the single most leverage-able implementation choice.
- EDS quality. K10 mentions environmental data is on "appropriate spatial and temporal grids" but the SAROPS data path uses NCOM, HYCOM, NDFD, etc. with assimilation at multiple time scales. A comparator using a stationary mean current is *very* different from one using HYCOM. BAM13 §2 explicitly notes: "the rate of expansion of search areas depends intimately on the quality of the forcing." This is the single largest realism axis the comparator should be honest about.
- Wave/Stokes drift omission. BAM13 explicitly flags this as an open gap in operational systems including SAROPS.

**Recommendation for the comparator-builder.** Name it a **"SAROPS-class comparator"** rather than "SAROPS reproduction." Pin all 20 implementation choices in §3 in a single configuration file. For each choice, flag whether it is documented (PRIMARY/SECONDARY) or invented (OPERATIONAL-UNKNOWN). When reporting comparator results, report alongside a sensitivity scan over the OPERATIONAL-UNKNOWN choices: if conclusions are robust to (LRC family ∈ {inverse-cube, exponential}, leeway σ ∈ [0.5×, 2×], resample ∈ {no, SIR}, step ∈ {15, 60} min), the comparison is meaningful; if not, the comparator is calibrated to its own choices, not to SAROPS.

---

## Citations (full)

[K10] Kratzke, T. M., Stone, L. D., Frost, J. R. (2010). "Search and Rescue Optimal Planning System." *Proc. 13th International Conference on Information Fusion (FUSION 2010),* Edinburgh, 26–29 July 2010. DTIC ADA564779. 8 pp.

[FS01] Frost, J. R., Stone, L. D. (2001). *Review of Search Theory: Advances and Applications to Search and Rescue Decision Support.* USCG Research and Development Center, Report CG-D-15-01. DTIC ADA397065.

[BAM13] Breivik, Ø., Allen, A. A., Maisondieu, C., Olagnon, M. (2013). "Advances in search and rescue at sea." *Ocean Dynamics* 63: 83–88. doi:10.1007/s10236-012-0581-1. Preprint: arXiv:1211.0805.

[SRW16] Stone, L. D., Royset, J. O., Washburn, A. R. (2016). *Optimal Search for Moving Targets.* Springer, International Series in Operations Research & Management Science 237. ISBN 978-3-319-26897-2.

[K46] Koopman, B. O. (1946). *Search and Screening.* OEG Report No. 56. Operations Evaluation Group, Office of the Chief of Naval Operations.

[K56] Koopman, B. O. (1956–57). "The Theory of Search I, II, III." *Operations Research* 4(3), 4(5), 5(5).

[A99] Allen, A. A., Plourde, J. V. (1999). *Review of Leeway: Field Experiments and Implementation.* USCG R&D Center, Report CG-D-08-99. (Cited via FS01, BAM13; not directly accessed.)

[A05] Allen, A. A. (2005). *Leeway Divergence Report.* USCG R&D Center, Report CG-D-05-05. (Cited via BAM13; not directly accessed.)

[F04] Frost, J. R. (USCG G-OPR-1) (2004). "SAROPS LATERAL RANGE CURVES." Memorandum, 11 March 2004. (Cited via K10 [3]; **not in open literature.**)

[CC58] Charnes, A., Cooper, W. W. (1958). "The theory of search: optimum distribution of search effort." *Management Science* 5: 44–50. (Foundational; cited via FS01 §2.4.2.)

[E63] Everett, H. (1963). "Generalized Lagrange multiplier method for solving problems of optimum allocation of resources." *Operations Research* 11: 399–417. (Cited via SRW16 §2.3.1.1; the underlying Lagrangian theorem.)

[RD80] Richardson, H. R., Discenza, J. H. (1980). "The United States Coast Guard Computer-Assisted Search Planning System (CASP)." *Naval Research Logistics Quarterly* 27: 141–157. (Cited via K10 [2].)
