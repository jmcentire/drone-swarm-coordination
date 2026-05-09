# Phase A Literature Audit — Synthesis

**Date:** 2026-05-08
**Phase:** A (literature audit) of v7 plan (B+C rigor track)
**Inputs:** `01_sarops_lineage.md`, `02_leeway_tables.md`, `03_decentralized_lineage.md`, `04_negative_search.md`
**Method:** four parallel research agents, plus Phase 5 multi-agent review and three rounds of sim review prior to execution.

## TL;DR

Phase A surfaced two structural updates that the v7 plan must integrate before Phase C bench design:

1. **The three-pillar novelty claim from sim round 1 partially collapsed.** The original claim was (a) byte-identical determinism, (b) broadcast-only / no channel filter, (c) operational SAR. Audit verdict:
   - (a) **PARTIAL** — channel-filter branch (Grime / Manyika / Makarenko-Durrant-Whyte) achieves byte-identical posteriors under reliable, in-order delivery on tree topologies. Our claim is novel relative to consensus-on-graph (Olfati-Saber, Bandyopadhyay-Chung) which is asymptotic, but not relative to channel-filter under different (more restrictive) assumptions.
   - (b) **POSITIONING, NOT MECHANISM** — our broadcast IS a (degenerate, trivially-correct) channel filter. We sidestep the channel-filter problem by topology rather than solve it with new machinery.
   - (c) **COVERED IN THE BROAD SENSE** — Bourgault, Furukawa, Durrant-Whyte (2003) explicitly demonstrate decentralized Bayesian SAR with DDF on a drifting-at-sea target. We cannot honestly claim "decentralized Bayesian SAR is new."

2. **A fourth load-bearing novelty surfaced that wasn't on the original list.** Negative-search and decentralized-lineage agents independently flagged: every prior decentralized Bayesian SAR architecture in the open literature settles for asymptotic agreement on *beliefs*; none produces byte-identical agreement on *decisions* under benign loss with a deterministic tie-break. Our architecture's contribution, narrowed:

   > **(d) Broadcast-as-shared-state substrate yielding byte-identical *per-tick decisions* (not just beliefs) via deterministic ID-based tie-break, with measured graceful degradation under correlated benign loss via majority-vote fallback.**

   This is the load-bearing novel claim. The first three pillars are positioning; (d) is the contribution.

## Honest novelty position (revised from v7)

The architecture is **not** a new decentralized Bayesian filter. The architecture is a **deterministic-decision substrate** for an existing decentralized Bayesian filter, with a measured loss-tolerance envelope.

The right way to position this:

> Bourgault et al. (2003), Furukawa et al. (2006), Bandyopadhyay & Chung (2014, 2018), Hollinger et al. (2015), Hare et al. (2018), Ghassemi & Chowdhury (2019), and Banerjee & Schneider (2024) collectively establish the design space for decentralized Bayesian search. All of these target *agreement on belief* — asymptotically, in a channel-filter graph, or via consensus loops. None target *agreement on per-tick decisions*. Decentralized search architectures that do not specify a deterministic tie-break degrade silently under disagreement. We propose a broadcast-synchronized substrate in which every drone reads the same broadcast snapshot, runs the same deterministic decision rule with explicit ID-based tie-break, and emits the same assignment per tick. Under broadcast loss, drones receive different snapshots; we measure the resulting decision-divergence empirically and characterize the architecture's graceful degradation under independent, correlated-burst, and asymmetric loss patterns.

That position survives the audit. The v7 claim ("functionally equivalent to centralized SAROPS-class") still holds, but the *contribution* of the architecture is not the equivalence — equivalence is necessary but not sufficient. The contribution is the deterministic per-tick decision property and the measured degradation envelope.

## Required citations

Before publication, the paper must cite (at minimum):

- **Kratzke, Stone, Frost (2010)** — SAROPS, the centralized comparator
- **Stone (1975/2007); Stone, Royset, Washburn (2016)** — optimal-search theory foundation
- **Allen & Plourde (1999); Allen (2005); Breivik & Allen (2008); Breivik et al. (2011, 2013)** — Leeway tables and drift dynamics
- **Bourgault, Furukawa, Durrant-Whyte (2003); Furukawa et al. (2006)** — the canonical decentralized-Bayesian-SAR papers; positions our work in their lineage
- **Grime & Durrant-Whyte (1994); Manyika & Durrant-Whyte (1995); Makarenko & Durrant-Whyte (2006)** — channel-filter lineage
- **Olfati-Saber, Fax, Murray (2007)** — consensus
- **Boyd, Ghosh, Prabhakar, Shah (2006)** — randomized gossip
- **Hlinka, Hlawatsch, Djurić (2013)** — distributed PF survey
- **Bandyopadhyay & Chung (2014); Hare, Bandyopadhyay, Chung (2018)** — consensus-on-PDF lineage
- **Hollinger & Singh (2010 ICRA, DOI 10.1109/robot.2010.5509175)** — periodic-connectivity / MIPP-PC. (Earlier audit drafts mistakenly attributed this to a 2012 T-RO version that does not exist; user-supplied correction integrated 2026-05-08.)
- **Tateo, Banfi, Riva, Amigoni, Bonarini (2018 AAAI, DOI 10.1609/aaai.v32i1.11587)** — PSPACE-completeness of the multi-agent connected path planning formalization of MIPP-PC.
- **Hollinger, Yerramalli, Singh, Mitra, Sukhatme (2011 ICRA / 2015 T-RO)** — distributed data fusion for multirobot search; load-bearing source-acquisition gap.
- **Ghassemi & Chowdhury (2019)** — Bayes-Swarm
- **Banerjee & Schneider (2024)** — DecSTER, the most recent Bayesian-search neighbor
- **Minelli, Panerati, Kaufmann, Ghedini, Beltrame, Sabattini (2020 *RAS* 124, DOI 10.1016/j.robot.2019.103384)** — self-optimization of resilient topologies for fallible multi-robots; connectivity-preservation + topology-resilience + area-coverage control with online optimization under fault injection. Closest operational neighbor in the connectivity/resilience-control lineage. Stacked-not-competing relative to our work: their connectivity preservation provides the substrate on which our deterministic-decision agreement could run. (Surfaced 2026-05-08 by user; integrated.)
- **Panerati et al. (2018) *Autonomous Robots*; Ghedini, Ribeiro, Sabattini (2017) *Networks*; Sabattini, Chopra, Secchi (2013) *IJRR*** — the antecedent papers in the Minelli-Panerati-Sabattini line that ground the connectivity-preservation control law and the vulnerability-based topology-resilience metric.
- **Couceiro (2016) Ch. 13 IGI Global *Handbook of Research on Design, Control, and Modeling of Swarm Robotics*** — survey-level operational case for distributed swarm SAR; establishes that decentralization is a requirement for SAR rather than a stylistic preference. (Surfaced 2026-05-08 by user; integrated as introduction-section citation.)

## SAROPS-class comparator: reproduction status

Positive findings:
- **Allen Leeway tables fully extractable.** All 85 object-class rows (downwind slope/offset/Syx + right/left CWL slope/offset/Syx) are in OpenDrift's `OBJECTPROP.DAT`, transcribed from the live SAROPS list by Art Allen via Breivik. Saved to `/tmp/OBJECTPROP.DAT`. The drift propagation component is reproducible without paywall access.
- **Drift dynamics equations are documented** — current vector (EDS gridded `(u, v)`, IDW + linear-time interpolation, AR(1) perturbation with 60-min half-life) + leeway vector (per-particle slope, downwind/crosswind decomposition, exponential jibing).
- **Particle-filter weight update form is documented** — `pfail` accumulates per-leg `(1 − λ(d_k))` products; renormalization at scenario level.
- **Stone's Lagrangian theorem is reproducible** from Stone-Royset-Washburn (2016) §2.3.1.1 with the Everett 1963 attribution.

Load-bearing OPERATIONAL-UNKNOWNs (must be fixed as documented choices in the comparator):
1. Particle-filter resampling step — **K10 documents none.** This is unusual; the comparator must either omit resampling (reproducing K10's apparent design) or add it (e.g., systematic resampling with adaptive ESS threshold) and document the choice.
2. Planner allocator: Stone's pointwise theorem vs. SAROPS's actual rectangle-constrained heuristic (accordion-search init + 12 perturbative moves + big-jump escape, with 1,500-particle subsample for POS estimation).
3. Replan trigger: SAROPS uses human-in-the-loop per-sortie debrief; for an autonomous comparator, this becomes either fixed-period or data-driven (event threshold).
4. Scenario-ensemble reweighting: emerges implicitly from particle-level weight updates and shared normalization in K10; must be made explicit in the comparator.
5. Lateral Range Curves: numeric values per object/sensor are in the cited Frost 2004 LRC memo (paywalled / non-public). Comparator must use proxies.
6. Crosswind sign-flip mean rate: K10 says exponentially distributed; mean is in Allen 2005 (DTIC ADA435435; non-browser blocked).
7. ~14 additional minor choices documented in `01_sarops_lineage.md` §3.

Recommendation: name the artifact **"SAROPS-class comparator (Stone-Kratzke-Allen; documented choices)"**. Pin all OPERATIONAL-UNKNOWN choices in a config file with PRIMARY/SECONDARY/INFERRED/OPERATIONAL-UNKNOWN labels. Run a sensitivity sweep over the OPERATIONAL-UNKNOWN axes when reporting results.

## Decentralized lineage: comparator status

Implementable from this audit alone (sufficient algorithm detail):
- **Bandyopadhyay-Chung 2014 (BCF)** — local Bayesian update + KL-minimizing consensus loop
- **Hare-Bandyopadhyay-Chung 2018 (DBF / log-opinion pool)** — local update + dynamic-average-consensus on log-likelihoods
- **Olfati-Saber 2007 distributed Kalman** — analogue for tracking
- **Boyd 2006 randomized gossip** — substrate for either of the above

Requires primary-source acquisition (load-bearing gap):
- **Hollinger et al. 2015 T-RO** — the closest live competitor in the operational/intermittent-connectivity branch. Must be obtained (IEEE Xplore) before claiming novelty over this branch.
- **Makarenko-Durrant-Whyte 2006** — channel-filter mechanism details for tree-graph case.

These are next-step actions; they don't block Phase B1/B2 work but must be resolved before Phase D (claim) is finalized.

## Updated v7 plan (post-audit)

The plan structure is unchanged. The claim text is updated:

**Claim (post-audit, post-synthesis):** A broadcast-as-shared-state substrate for decentralized Bayesian SAR — built atop the lineage established by Bourgault et al. (2003), Furukawa et al. (2006), and Bandyopadhyay & Chung (2014, 2018) — yielding byte-identical *per-tick decisions* across drones via deterministic ID-based tie-break under reliable broadcast, with characterized graceful degradation under independent, correlated-burst, and asymmetric benign loss via majority-vote fallback. Functionally equivalent to a SAROPS-class centralized comparator (Stone-Kratzke-Allen; documented operational choices) under reliable comms, verified by particle-set / resampling-decision log equivalence. Adversarial threat models out-of-scope; deterministic-decision substrate is the load-bearing contribution.

## Stacked-not-competing relationship to Minelli 2020 (and the connectivity/resilience-control lineage)

A note on architecture composition. The Minelli-Panerati-Sabattini line solves a *different layer* of the problem than we do:

- **Their layer:** preserve the communication graph's algebraic connectivity λ₂ above a threshold under faults (transient comms loss + permanent hardware failure), so that any consensus-or-shared-state mechanism running on top of the graph has a viable substrate. The control objective is topological — keep the graph connected and resilient to single-point failures.

- **Our layer:** assume the broadcast medium exists (or is emulated above an underlying mesh) and run a deterministic per-tick decision-agreement protocol on the broadcast. The control objective is decision-level — every drone reaches the same next-manifold via the same algorithm on the same shared state.

These two layers stack. A real-world deployment of our architecture on a fallible swarm would benefit directly from Minelli-style connectivity preservation as the **substrate that maintains the broadcast property** under fault injection. Conversely, Minelli's connected swarm benefits from a deterministic-decision substrate when the application requires byte-identical decisions rather than just connectivity.

For the v1.2 paper, this is a citation-and-positioning move: cite Minelli 2020 (and the Sabattini-Chopra-Secchi 2013 IJRR antecedent for connectivity preservation, plus Ghedini-Ribeiro-Sabattini 2017 Networks for the vulnerability metric) in the related-work section. Position our contribution as: *given a broadcast substrate (which Minelli-class connectivity-preservation control can provide), here is a deterministic-decision agreement protocol with measured benign-loss tolerance.* The Minelli line is **not a competitor**; it is the connectivity-control substrate our architecture composes with.

## Phase A → B status

Phase A is **substantively complete** for proceeding to Phase B1 (benign-loss envelope variable ranges) and Phase B2 (mission-class taxonomy and scope boundaries).

Two open items from Phase A that flow into B and C:

1. **Hollinger 2015 T-RO must be sourced** before Phase D (claim) is finalized. This is the closest live competitor in the operational/intermittent-connectivity branch; the novelty claim against (c) operational SAR depends partly on whether Hollinger 2015 covers our exact regime.
2. **The 20 OPERATIONAL-UNKNOWN choices in the SAROPS-class comparator** must be enumerated in a config file before Phase C bench implementation begins. Sensitivity sweeps over these will be part of the bench design.

Phase B1 (envelope variable ranges) and Phase B2 (mission-class taxonomy) can begin immediately. Phase C (bench implementation) requires the OPERATIONAL-UNKNOWN config to be pinned, which in turn requires either acquiring the paywalled primary sources or making documented choices.

## Files

- `01_sarops_lineage.md` — SAROPS-class component decomposition, 20 OPERATIONAL-UNKNOWNs enumerated
- `02_leeway_tables.md` — Allen Leeway 85 object classes, full numerical tables, drift equations
- `03_decentralized_lineage.md` — three lineage branches (channel-filter, consensus-on-graph, operational/intermittent), per-paper algorithm summaries, novelty verdicts
- `04_negative_search.md` — 24 searches, 7 deep-dive abstracts, found prior art with overlap scoring (EXACT/PARTIAL/DIFFERENT) on the four conjuncts (leaderless / broadcast / byte-identical / Bayesian SAR)
- `00_synthesis.md` — this document
