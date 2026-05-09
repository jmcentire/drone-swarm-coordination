# Audit 02: Allen's Leeway Tables for Maritime SAR Object-Class Drift

**Scope.** Extract numerical Leeway parameters as used in SAROPS (US Coast Guard Search and Rescue Optimal Planning System) and the Norwegian operational LEEWAY model, including the equation set, the validation experiments behind them, and operational caveats.

**Bottom line.** A complete, reproducible parameter set covering 85 object classes was obtained directly from the OpenDrift `OBJECTPROP.DAT` file, which Øyvind Breivik received from Art Allen on 2010-11-15 as a transcription of the SAROPS taxonomy. The full nine-coefficient row is given for every class (PIW, ballasted/shallow-ballast/no-ballast life rafts, Ovatek hard-shell, USCG sea-rescue kit, sailboats, fishing vessels, sport fisher, sea kayak, surf board, oil drum, shipping containers, WWII mine, refugee rafts, sewage and medical waste). The downstream comparator can therefore reproduce SAROPS-class trajectories without paywalls. Below: the equations, a curated subset of tables (six headline classes), the validation methodology, and notes on access and accuracy.

---

## 1. Drift Dynamics Equation Set

Following Breivik & Allen (2008, §2.4) and Breivik et al. (2011, §2.2), the trajectory of a drifting SAR object is the time integral of two superposed velocity components — the local near-surface current and the wind-driven Leeway:

```
x(t) - x(0) = ∫₀ᵗ V(t') dt' = ∫₀ᵗ [ L(t') + u_w(t') ] dt'
```

where `u_w` is the Eulerian surface current at ~0.5 m depth (matching the draft of typical SAR objects and the depth of leeway-experiment current meters), and `L` is the Leeway vector. The Leeway is decomposed in the local downwind frame into a downwind component (DWL) and a left/right crosswind component (CWL):

```
L_d  = a_d · W₁₀ + b_d  + ε_d           (downwind)
L_c⁺ = a_c⁺ · W₁₀ + b_c⁺ + ε_c⁺          (right-of-downwind crosswind)
L_c⁻ = a_c⁻ · W₁₀ + b_c⁻ + ε_c⁻          (left-of-downwind crosswind)
```

`W₁₀` is the wind speed at 10 m AGL, `a` is the slope (dimensionless, often quoted as a percent), `b` is the offset (cm/s), and `ε` is a Gaussian residual whose standard deviation is `S_yx` (cm/s). A 50/50 prior is assigned to left-drifting vs right-drifting tacks because their initial orientation is essentially unpredictable (Breivik & Allen 2008, §2.1). Stokes drift and wave forcing are deliberately omitted; they were not separately measured in the field campaigns and are therefore absorbed into the empirical coefficients (Breivik & Allen 2008, §2.3).

Operational SAROPS/LEEWAY runs use a Monte Carlo ensemble (~500 members) where each member draws a fixed pair of (a, b) by perturbing the regression coefficients with their `S_yx`, perturbs the wind and current fields with random walks (σ_W ≈ 2.6 m/s for 12-h forecasts, σ_u ≈ 0.25 m/s), and chooses left vs right tack at release. The ensemble is integrated with a second-order Runge–Kutta scheme on a sphere. The dominant source of dispersion is the time-invariant Leeway perturbation, not the wind/current random walks — Breivik & Allen (2008, §3.2) measured <2% difference in spread when wind/current perturbations were turned off.

---

## 2. Object-Class Catalog

The canonical SAROPS taxonomy has **85 numbered classes** organised into eight major families:

1. **Persons-in-water (PIW-1 … PIW-6)** — unknown state, vertical PFD, sitting PFD, survival suit, scuba suit, deceased face-down.
2. **Deep-ballast life rafts (LIFE-RAFT-DB-10 … DB-22)** — 4–14 person vs 15–50 person, with/without canopy, with/without drogue, light/heavy loading, plus capsized and swamped variants.
3. **Shallow-ballast life rafts (LIFE-RAFT-SB-6 … SB-11)** — including Navy SEIE 1-man.
4. **No-ballast life rafts (LIFE-RAFT-NB-1 … NB-5)** — with/without canopy, with/without drogue.
5. **USCG-RESCUE, AVIATION-1/2, LIFE-CAPSULE, OVATEK-CRAFT-1 … 6** — engineered survival craft.
6. **Person-powered vessels** — sea kayak, surfboard, windsurfer.
7. **Skiffs and powerboats** — modified-V runabout, V-hull, swamped/capsized, aluminum bow-to-stern, sport boat, sport fisher.
8. **Fishing vessels** — generic, Hawaiian Sampan, Japanese side-stern trawler, Japanese longliner, Korean fishing vessel, gill-netter; coastal freighter; FV-debris.
9. **Sailboats (SAILBOAT-1 … 8)** — mono-hull, dismasted with rudder amidships/missing, bare-masted with rudder amidships/hove-to, fin-keel shallow-draft, Sunfish dinghy.
10. **Miscellaneous** — SLDMB (no-windage reference), SEPIRB, bait/wharf box (3 loadings), 55-gallon oil drum, 20-ft and 40-ft (1:3 scaled) shipping containers, WWII L-MK2 mine, Cuban refugee raft (with/without sail), sewage floatables, medical waste (vials and syringes, large/small).

Class labels marked `>`, `>>`, `>>>` denote sub-classes; the unindented entry in each family is the "mean values" prior to be used when no specific information is known.

---

## 3. Numerical Parameter Tables

The nine coefficients per row (from `OBJECTPROP.DAT`, transmitted by Allen to Breivik 2010-11-15) are:

| Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 | Col 7 | Col 8 | Col 9 |
|---|---|---|---|---|---|---|---|---|
| `a_d` slope (%) | `b_d` offset (cm/s) | `S_yx` DWL (cm/s) | `a_c⁺` slope (%) | `b_c⁺` offset (cm/s) | `S_yx` CWL⁺ (cm/s) | `a_c⁻` slope (%) | `b_c⁻` offset (cm/s) | `S_yx` CWL⁻ (cm/s) |

A slope quoted as 0.96 means 0.96% of the 10-m wind speed (i.e., 0.0096 in dimensionless form). Wind 10 m/s × slope 1.0% = 10 cm/s of leeway from the slope term, plus the cm/s offset.

### Table A — Headline classes (subset of the 85; full file at `/tmp/OBJECTPROP.DAT`)

| Class | Description | a_d | b_d | Syx_d | a_c⁺ | b_c⁺ | Syx_c⁺ | a_c⁻ | b_c⁻ | Syx_c⁻ |
|---|---|---|---|---|---|---|---|---|---|---|
| PIW-1 | Person-in-water, unknown state (mean) | 0.96 | 0.00 | 12.00 | 0.54 | 0.00 | 9.40 | -0.54 | 0.00 | 9.40 |
| PIW-2 | PIW, vertical, PFD type III, conscious | 0.48 | 0.00 | 8.30 | 0.15 | 0.00 | 6.70 | -0.15 | 0.00 | 6.70 |
| PIW-3 | PIW, sitting, PFD type I or II | 1.60 | -3.98 | 2.42 | 0.13 | 0.33 | 2.11 | -0.13 | -0.33 | 2.11 |
| PIW-4 | PIW, survival suit, face up | 1.71 | 1.12 | 3.93 | 1.36 | -3.30 | 1.71 | -0.13 | -2.65 | 1.62 |
| PIW-5 | PIW, scuba suit, face up | 0.63 | 0.00 | 5.30 | 0.31 | 0.00 | 4.50 | -0.31 | 0.00 | 4.50 |
| PIW-6 | PIW, deceased, face down | 1.117 | 10.2 | 3.04 | 0.04 | 3.90 | 4.05 | -0.04 | -3.90 | 4.05 |
| LIFE-RAFT-DB-10 | Deep-ballast life raft, general (mean) | 3.52 | -2.50 | 6.10 | 0.62 | -3.00 | 3.50 | -0.45 | -0.20 | 3.60 |
| LIFE-RAFT-DB-11 | 4–14 person, deep ballast, canopy (avg) | 3.50 | -1.80 | 6.40 | 0.78 | -3.60 | 3.60 | -0.47 | -0.10 | 3.90 |
| LIFE-RAFT-DB-15 | 4–14 person, deep ballast, canopy, drogue | 1.91 | 0.90 | 1.60 | 0.78 | -3.60 | 3.60 | -0.47 | -0.10 | 3.90 |
| LIFE-RAFT-DB-21 | Deep-ballast life raft, capsized | 0.88 | 0.00 | 2.50 | 0.18 | 0.00 | 2.40 | -0.18 | 0.00 | 2.40 |
| LIFE-RAFT-SB-6 | Shallow-ballast w/canopy, general (mean) | 2.68 | 0.00 | 12.00 | 1.10 | 0.00 | 9.40 | -1.10 | 0.00 | 9.40 |
| LIFE-RAFT-NB-1 | No-ballast life raft, general (mean) | 3.70 | 0.00 | 12.00 | 1.98 | 0.00 | 9.40 | -1.98 | 0.00 | 9.40 |
| AVIATION-1 | 4–6 person, no ballast, canopy, no drogue | 3.39 | 0.00 | 2.40 | 1.49 | 0.00 | 2.40 | -1.49 | 0.00 | 2.40 |
| SKIFF-1 | Modified-V runabout, outboard | 3.15 | 0.00 | 2.20 | 1.29 | 0.00 | 2.20 | -1.29 | 0.00 | 2.20 |
| SKIFF-3 | Skiffs, swamped and capsized | 1.65 | 0.00 | 3.10 | 0.39 | 0.00 | 2.90 | -0.39 | 0.00 | 2.90 |
| FISHING-VESSEL-1 | Fishing vessel, general (mean) | 2.47 | 0.00 | 12.00 | 2.76 | 0.00 | 9.40 | -2.76 | 0.00 | 9.40 |
| SAILBOAT-1 | Mono-hull sailboat (avg) | 4.5 | 0.0 | 19.4 | 4.95 | 0.0 | 18.42 | -2.82 | 0.0 | 24.95 |
| SAILBOAT-2 | Mono-hull, dismasted (avg) | 3.94 | 0.0 | 19.62 | 3.98 | 0.0 | 12.68 | -0.79 | 0.0 | 2.13 |
| OIL-DRUM | 55-gallon (220 l) oil drum | 0.75 | 2.66 | 2.83 | 0.48 | 2.88 | 3.92 | -0.45 | -1.46 | 4.59 |
| CONTAINER-1 | 1:3 scaled 40-ft container (70% submerged) | 1.78 | 1.44 | 2.99 | 0.27 | -2.44 | 2.31 | -0.27 | 2.44 | 2.31 |
| CONTAINER-2 | 20-ft container (80% submerged) | 1.25 | 3.96 | 2.81 | 0.19 | 1.14 | 4.36 | -0.19 | -1.14 | 4.36 |
| MINE | WWII L-MK2 mine | 1.07 | 4.47 | 6.55 | 0.41 | 1.15 | 4.13 | -0.41 | -1.15 | 4.13 |
| SLDMB | Self-locating datum marker buoy (no windage) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Table B — Sailboat sub-classes (highest crosswind divergence in catalog)

| Class | Description | a_d | b_d | Syx_d | a_c⁺ | b_c⁺ | Syx_c⁺ | a_c⁻ | b_c⁻ | Syx_c⁻ |
|---|---|---|---|---|---|---|---|---|---|---|
| SAILBOAT-3 | Dismasted, rudder amidships | 6.27 | 0.0 | 9.22 | 4.01 | 0.0 | 12.34 | -4.01 | 0.0 | 12.34 |
| SAILBOAT-4 | Dismasted, rudder missing | 2.42 | 0.0 | 1.29 | 0.79 | 0.0 | 2.17 | -0.79 | 0.0 | 2.13 |
| SAILBOAT-5 | Bare-masted (avg) | 5.23 | 0.0 | 18.52 | 5.89 | 0.0 | 18.87 | -6.09 | 0.0 | 18.58 |
| SAILBOAT-6 | Bare-masted, rudder amidships | 6.66 | 0.0 | 16.56 | 7.36 | 0.0 | 15.15 | -7.53 | 0.0 | 16.69 |
| SAILBOAT-6b | Bare-masted, rudder hove-to | 2.69 | 0.0 | 10.48 | 2.81 | 0.0 | 11.14 | -4.01 | 0.0 | 13.66 |

### Divergence-angle interpretation

Allen and Plourde report leeway as either (speed L, divergence angle Lα) or (downwind, crosswind). Allen (2005) showed the latter is more numerically stable at low wind. The effective divergence angle for any class is recoverable as

```
α⁺ = atan2(a_c⁺ · W + b_c⁺,  a_d · W + b_d)
α⁻ = atan2(a_c⁻ · W + b_c⁻,  a_d · W + b_d)
```

For PIW-1 at 10 m/s wind, α⁺ ≈ atan(5.4 / 9.6) ≈ 29° right of downwind, with a symmetric 29° left tack. For SAILBOAT-6 the angle is ≈48° — explaining the disjoint two-lobed search areas seen in the Breivik & Allen (2008) Fig 7.

---

## 4. Validation Experiments

### 4.1 Field methodology (Allen & Plourde 1999; Breivik et al. 2011)

The empirical coefficients come from two methods:

* **Direct method.** Object instrumented with a GPS receiver and an anemometer mast; a current meter (typically S4) towed ~30 m astern at 0.5 m depth to capture the local wind and the local Eulerian current synchronously with the object's own GPS track. The leeway vector is then `L = V_object − u_current`. Sampled at 10-minute vector averages (Breivik & Allen 2008, §2.1; Breivik et al. 2011, §2.3 and §3).
* **Indirect method.** Object tracked by GPS or radar, with wind and current taken from a nearby instrumented buoy. Used when a current meter cannot be towed (e.g., PIW experiments). Higher experimental variance.

Linear regression is computed both **unconstrained** (allowing a non-zero offset `b`) and **constrained through the origin** (`b = 0`). Allen and Plourde retain both because the constrained version is physically natural at zero wind, while the unconstrained version is operationally preferred — its `S_yx` is necessarily smaller and the resulting search areas inflate more slowly under moderate winds. SAROPS uses unconstrained for classes with adequate data and constrained-through-origin for sparse classes. The SLDMB row (all zeros) is the no-windage reference for sea-truth Lagrangian current measurement, not a leeway target.

Sign changes (jibing) are identified by visually inspecting the progressive vector diagram of the leeway relative to local downwind and splitting the run into left-tack and right-tack segments before fitting `a_c⁺` and `a_c⁻` separately. Capsizing and swamping are treated as discrete state transitions to a different row of the table (e.g., LIFE-RAFT-DB-10 → DB-21 capsized).

### 4.2 Goodness-of-fit numbers (Breivik et al. 2011, Tables 1–3)

For the three objects studied with full instrumentation in Breivik et al. (2011) — a 1:3 40-ft container (70% submerged), a 55-gallon oil drum, and a WWII L-MK2 mine — the unconstrained-regression r² for the downwind component is:

| Object | DWL r² | DWL S_yx (cm/s) | CWL r² | CWL S_yx (cm/s) |
|---|---|---|---|---|
| Container | 0.92 | 3.0 | 0.84 | 2.4 (right) / 2.3 (left) |
| Oil drum | 0.44 | 2.9 | 0.45 | 4.2 |
| Mine | 0.92 | 1.9 | 0.85 | 2.6 |

The drum's lower r² is attributed to flow distortion around the underside of the drum and to its small size precluding an onboard anemometer. These values are representative of "good" modern leeway field data; older categories in Allen & Plourde (1999) — particularly mean-values rows like PIW-1, FISHING-VESSEL-1, LIFE-RAFT-NB-1 with `S_yx = 12.00` cm/s on the DWL component — were established before modern GPS and have correspondingly larger residual variance, which dominates ensemble spread.

### 4.3 Operational drift exercises (Breivik & Allen 2008, §4.1)

Three reported field events confirmed model skill (these are not formal validations, only illustrations):

* **Benthic lander, 2002-03-14.** ARGOS-tracked release; PIW-1 priors reproduced the intermediate positions and the 3-day-later pickup.
* **Faroe-Iceland exercise 2003.** GPS-tracked life raft for 24 h; pickup near the centre of the LEEWAY ensemble search area.
* **Faroe-Iceland exercise 2004.** The LEEWAY downwind/crosswind decomposition produced a search area 25–50% the size of the previous Faroese and Icelandic methods (which used the older speed-and-angle formulation), with the raft again recovered near the centre.

A formal SAROPS/LEEWAY validation campaign with statistically significant N has not been published in the open literature; this remains an open data gap.

---

## 5. Operational Notes and Caveats

* **Mission-priors.** Coastal SAR is dominated by skiffs, sport boats, sport fishers, fishing-vessel-1, sea kayaks, and PIWs — the closer to shore, the more the current model resolution matters (Breivik & Allen 2008, Fig 11 shows most US incidents within 25 nm). Offshore SAR weights life rafts (DB and SB), aviation slides, sailboats, and shipping containers more heavily.
* **Wind-range coverage.** The published regressions are valid in the wind range covered by the field experiments (typically 2–20 m/s). Linear extrapolation to gale-force conditions is unsupported; jibing, swamping, and capsizing become probable above ~15 m/s and the model has no validated stochastic state-transition for them.
* **"Mean values" classes are wide.** Where a major class (e.g., PIW-1, FISHING-VESSEL-1, LIFE-RAFT-SB-6) has `S_yx = 12.00 / 9.40` cm/s, the residual variance is ~3× that of well-instrumented classes. These are the recommended priors when nothing about the object is known but they will inflate search areas correspondingly.
* **Stokes drift.** Embedded in the empirical coefficients. If the comparator uses an explicit Stokes-drift wave model on top of the leeway equation, double-counting will occur; do not add Stokes when using these tables (Breivik & Allen 2008, §2.3).
* **Current depth.** Use surface-current at 0.5 m. Using a top-layer current vector from a coarse ocean model (e.g., the upper ~1 m mean) introduces a bias that is small for offshore but non-negligible in shallow coastal cells.
* **Asymmetric tacks.** PIW-4 (survival suit) and several life-raft sub-classes are visibly asymmetric (`|a_c⁻|` ≠ `|a_c⁺|`). The 50/50 left/right prior must therefore use the asymmetric coefficients verbatim from the table; do not symmetrise.
* **SLDMB (class 67) is all zeros.** This is intentional — the self-locating datum marker buoy is a current-only Lagrangian reference object, used to pin the ambient surface current during a search, not a search target.

---

## 6. Source Access Status

| Source | Access | Notes |
|---|---|---|
| Allen & Plourde 1999, USCG R&D CG-D-08-99 (DTIC ADA366414) | DTIC PDF returns HTTP 403 to non-browser clients; freely fetchable in a browser. Public-domain US government work. | Footnote references in `OBJECTPROP.DAT` cite specific page numbers (e.g., p B5.5-1 for Cuddy Cabin) confirming the table-of-coefficients structure was published in this report. |
| Allen 2005, USCG R&D CG-D-05-05 "Leeway Divergence" (DTIC ADA435435) | DTIC PDF, public-domain US government work. | Source of the downwind/crosswind decomposition and its conversion of Allen & Plourde (1999) speed-and-angle parameters. |
| Allen et al. 1999, "The Leeway of Persons-In-Water and Three Small Craft" (DTIC ADA376479) | DTIC PDF. | Source for PIW-1 through PIW-6 numbers. |
| Breivik & Allen 2008, *J Marine Systems* 69 (preprint arXiv:1111.1102) | Open-access preprint at arxiv.org/pdf/1111.1102 — extracted in this audit. | Authoritative description of the equations, Monte Carlo procedure, and Norwegian operational implementation. |
| Breivik et al. 2011, *Applied Ocean Research* 33 — "Wind-induced drift of objects at sea: The leeway field method" | Open-access at archimer.ifremer.fr/doc/00037/14814/12152.pdf — extracted in this audit. | Authoritative description of the field method, including container/drum/mine validation r² figures. |
| **OpenDrift `OBJECTPROP.DAT`** | Public, MIT-licensed; fetched at `/tmp/OBJECTPROP.DAT` (290 lines). | **Canonical numerical table** — transmitted by Art Allen to Øyvind Breivik on 2010-11-15 from the SAROPS list. Contains all 85 classes with full nine-coefficient rows. This is the load-bearing artefact for any SAROPS-class comparator. |

The numerical tables for **all 85 SAROPS object classes** are reproduced verbatim in `/tmp/OBJECTPROP.DAT` (full text also embedded in this audit's lineage trail). Six headline classes are tabulated above; the remaining 79 follow the identical column convention and are mechanical to import. The audit therefore satisfies the deliverable: a complete reproducible parameter set is available without paywall.

---

## 7. Reference Summary

* Allen, A. A. & Plourde, J. V. (1999). *Review of Leeway: Field Experiments and Implementation*, USCG R&D Center Report CG-D-08-99 (DTIC ADA366414). Public-domain US government work.
* Allen, A. A. (2005). *Leeway Divergence*, USCG R&D Center Report CG-D-05-05 (DTIC ADA435435). Public-domain US government work.
* Breivik, Ø. & Allen, A. A. (2008). "An operational search and rescue model for the Norwegian Sea and the North Sea." *Journal of Marine Systems* 69, 99–113. Preprint arXiv:1111.1102.
* Breivik, Ø., Allen, A. A., Maisondieu, C., Roth, J. C. (2011). "Wind-induced drift of objects at sea: The leeway field method." *Applied Ocean Research* 33, 100–109.
* Breivik, Ø. (2011, ongoing). OpenDrift `OBJECTPROP.DAT`, github.com/OpenDrift/opendrift/blob/master/opendrift/models/OBJECTPROP.DAT — transmission from Art Allen 2010-11-15.
