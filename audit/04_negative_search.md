# Negative-Search Audit: Prior Art for a Leaderless, Broadcast-Based, Byte-Identical Bayesian SAR Architecture

**Date:** 2026-05-08
**Auditor:** automated literature search (WebSearch / WebFetch over Google Scholar, Semantic Scholar, arXiv abstracts, IEEE/Springer/Elsevier landing pages, ACM DL).
**Claim under test:** A leaderless, broadcast-based, *byte-identical-decisions* distributed Bayesian search-and-rescue architecture for benign-loss comms-constrained swarms, functionally equivalent to centralized SAROPS-class under reliable comms, gracefully degrading under benign loss via deterministic majority-vote tie-break. Operational target: maritime SAR, autonomous swarm reconnaissance, deep-ocean search.

The four conjuncts the audit must individually attack:
1. **Leaderless** — no privileged node, no elected coordinator.
2. **Broadcast-based** — shared per-tick channel, not pairwise gossip or neighbour-graph consensus rounds.
3. **Byte-identical decisions** — every drone deterministically computes the *same* decision from the *same* broadcast snapshot, not an asymptotically-convergent estimate.
4. **Bayesian SAR** — recursive Bayesian belief over target location, applied to search and rescue (not generic distributed estimation, not target tracking, not surface inspection).

The claim is the *combination*. Each conjunct in isolation is well-trodden.

---

## 1. Search log

All searches performed 2026-05-08 from a US WebSearch endpoint and WebFetch where the abstract or hosted PDF was retrievable. "Hits" is the count returned in the top-10 result block; "screened" is the fraction read for content (title + snippet at minimum; abstract for plausibly relevant items; full-text for the strongest candidates).

| # | Database / surface | Query | Hits | Screened |
|---|--------------------|-------|------|----------|
| 1 | Google Scholar / Web | `"decentralized Bayesian search" multi-robot UAV swarm` | 10 | 10/10 (titles + snippets); 2 abstracts pulled |
| 2 | Web | `"distributed Bayesian filter" consensus multi-agent search rescue` | 10 | 10/10; 3 abstracts |
| 3 | Web | `"broadcast Bayesian update" multi-robot` | 10 | 10/10; 2 abstracts |
| 4 | Web | `leaderless Bayesian search and rescue swarm` | 10 | 10/10; 1 abstract |
| 5 | Web | `"consensus particle filter" swarm decentralized` | 10 | 10/10 |
| 6 | Web | `"gossip Bayesian filtering" multi-robot` | 10 | 10/10 |
| 7 | Web | `"log-linear opinion pool" consensus filter multi-robot` | 10 | 10/10; 1 abstract (Hare et al.) |
| 8 | Web | `"channel filter" decentralized data fusion multi-robot search` | 10 | 10/10 |
| 9 | Web | `"Bandyopadhyay" "Chung" Bayesian consensus filter 2018` | 10 | 10/10; arXiv 1403.3117 + 1712.04062 fetched |
| 10 | Web | `Makarenko Durrant-Whyte decentralized Bayesian active sensor` | 10 | 10/10 |
| 11 | Web | `Hollinger Singh decentralized search robots 2015` | 10 | 10/10 |
| 12 | Web | `Bourgault Furukawa Durrant-Whyte coordinated decentralized search Bayesian` | 10 | 10/10; 1 abstract via Nuit Blanche |
| 13 | Web | `"Dec-POMDP" search rescue swarm decentralized` | 10 | 10/10 |
| 14 | Web | `"byte-identical" OR "deterministic replica" multi-robot Bayesian search` | 10 | 10/10 — no hits on the deterministic-replica framing |
| 15 | Web | `"deterministic majority vote" multi-agent tie-break Bayesian` | 10 | 10/10 |
| 16 | Web | `"broadcast-synchronized" Bayesian filter swarm` | 10 | 10/10 — no hits on the exact phrasing |
| 17 | Web | `arxiv 2024 2025 multi-robot Bayesian search decentralized swarm` | 10 | 10/10; 4 abstracts |
| 18 | Web | `"SAROPS" decentralized swarm Bayesian search planning` | 10 | 10/10; SAROPS paper checked |
| 19 | Web | `arxiv 2025 "consensus search" multi-agent UAV maritime` | 10 | 10/10; 3 abstracts |
| 20 | Web | `"deterministic" "broadcast" multi-robot replicated Bayesian belief` | 10 | 10/10 — no direct hit |
| 21 | Web | `"replicated state machine" multi-robot Bayesian search drone` | 10 | 10/10 — no direct hit; closest is Shirsat 2020 (probabilistic consensus on Markov chains) |
| 22 | Web | `"identical posterior" "identical decision" multi-agent Bayesian filter` | 10 | 10/10 — no direct hit |
| 23 | Web | `Charrow Atanasov Pappas decentralized active information acquisition target search` | 10 | 10/10 |
| 24 | Web | `Lanillos Besada-Portas decentralized Bayesian search multi-UAV` | 10 | 10/10 |

**WebFetch deep-dives:** arXiv 1403.3117 (Bandyopadhyay-Chung), 1712.04062 (Hare-Bandyopadhyay-Chung), 1905.09988 (Bayes-Swarm), 2401.03154 (DecSTER), 2511.22225 (Aguirre et al.), 2602.08450 (Valun Bay), Nuit Blanche post on Bourgault et al.

**Access barriers encountered:**
- Bourgault 2003 "Coordinated decentralized search for a lost target in a Bayesian world" — primary PDF host returned a self-signed-certificate error; abstract content recovered via a third-party blog that quotes the paper, plus the Semantic Scholar landing page. Sufficient for overlap scoring but the full algorithm pseudocode was not reread first-hand.
- Springer / Autonomous Robots articles redirect through an SSO authorize endpoint and could not be auto-retrieved; abstracts and snippets were used.
- arXiv PDF body for 1403.3117 and 1712.04062 returned binary-stream errors when piped through the markdown converter; abstract content via Semantic Scholar / arXiv listing pages was used instead.
- IEEE Xplore full-text behind paywall for several Hollinger / Charrow / Atanasov entries — abstracts only.

These barriers do not move the conclusion: the abstracts and the well-known synopses of these works (these are highly-cited, surveyed pieces) are sufficient to score the four-conjunct overlap.

---

## 2. Found prior art (papers that overlap the claim)

For each, scoring is **EXACT / PARTIAL / DIFFERENT** on each of the four conjuncts: leaderless, broadcast-based, byte-identical decisions, Bayesian SAR.

### 2.1 Bourgault, Furukawa, Durrant-Whyte — "Coordinated decentralized search for a lost target in a Bayesian world" (2003); "Optimal Search for a Lost Target in a Bayesian World" (2003); follow-ons through Furukawa et al. 2006 (Recursive Bayesian search-and-tracking using coordinated UAVs)

- **Algorithm summary.** Each UAV runs a local recursive Bayesian filter over a target-location PDF on a grid. UAVs exchange likelihoods through a Decentralized Data Fusion (DDF) network with channel filters to remove double-counted common information. Each decision-maker plans locally based on its own copy of the target PDF; the design intent is that all decision-makers reach an *equivalent representation* of the PDF and so coordinate without exchanging plans. Demonstrated for a stationary lost target and a drifting-at-sea target — i.e., maritime SAR.
- **Leaderless:** EXACT. No central coordinator; every node runs the same DDF.
- **Broadcast-based:** PARTIAL. The DDF uses a channel-filter graph with neighbour-pair message exchange, not a single shared per-tick broadcast slot. Channel filters explicitly assume a tree (or cycle-managed) communication graph. They do *not* model an SSB-style broadcast medium where every node sees every other node's payload in one tick.
- **Byte-identical decisions:** PARTIAL → DIFFERENT. The DDF target is *equivalent* PDFs at each node *under reliable, lossless, in-order delivery*. Under message loss, channel-filter divergence is well-known and the architecture does not specify a deterministic tie-break; differing PDFs lead to differing locally-optimal plans. The architecture is *equivalent-in-the-limit*, not byte-identical-per-tick.
- **Bayesian SAR:** EXACT. This is the canonical decentralized Bayesian SAR paper.

**How our work differs.** We replace the channel-filter / pairwise-DDF substrate with a single shared broadcast medium and prove byte-identical decisions in a stronger sense: every drone reads the *same* broadcast snapshot, runs the *same* deterministic function (with explicit ID-based tie-break), and emits the *same* assignment. We make the byte-identicality property load-bearing — ID tie-breaks, deterministic majority-vote on disagreement, and a degradation curve under benign loss are all first-class. Bourgault et al.'s framework instead degrades through silent PDF divergence with no formal tie-break.

### 2.2 Bandyopadhyay & Chung — "Distributed Estimation using Bayesian Consensus Filtering" (ACC 2014; arXiv 1403.3117); Hare, Bandyopadhyay & Chung — "Distributed Bayesian Filtering using Logarithmic Opinion Pool for Dynamic Sensor Networks" (Automatica 2018; arXiv 1712.04062)

- **Algorithm summary.** A networked group of sensing agents collectively estimates the PDF of a moving target. Each agent locally runs a Bayesian filter and then runs *consensus loops* (KL-divergence-minimising; later replaced by a logarithmic opinion pool with dynamic average consensus) to converge each agent's posterior toward a common posterior. The 2018 paper proves global exponential convergence to an error ball around the centralized joint-likelihood posterior.
- **Leaderless:** EXACT.
- **Broadcast-based:** DIFFERENT. The consensus update operates over a (possibly time-varying) communication graph with neighbour-pair averaging, not a shared broadcast slot. Multiple consensus rounds per measurement cycle are required.
- **Byte-identical decisions:** DIFFERENT. The convergence guarantee is *asymptotic* — agents agree only in the limit (or up to an error ball). Per-tick decisions diverge whenever the consensus has not converged. There is no deterministic tie-break.
- **Bayesian SAR:** PARTIAL. The framework is generic moving-target tracking on a sensor network; SAR is one possible application but is not the framing, and there is no maritime / drift / SAROPS-class evaluation.

**How our work differs.** BCF and DBF deliver asymptotic agreement on a *belief*; we deliver per-tick agreement on a *decision*. We do not need consensus loops because the broadcast itself is the consensus mechanism — every drone has the same shared state by construction whenever the broadcast is received. Under loss, our majority-vote tie-break provides a deterministic discrete-time guarantee where DBF-class methods provide an asymptotic continuous-time guarantee.

### 2.3 Ghassemi & Chowdhury — "Decentralized Informative Path Planning with Exploration-Exploitation Balance for Swarm Robotic Search" (Bayes-Swarm; arXiv 1905.09988, 2019); follow-on "Informative Path Planning with Local Penalization for Decentralized and Asynchronous Swarm Robotic Search" (1907.04396, 2019)

- **Algorithm summary.** Extends batch Bayesian Optimization (Gaussian-process surrogate of a spatially-distributed signal field) to a swarm. Each robot independently runs Bayes-Swarm, sharing observations asynchronously; an acquisition function with a local-penalization term avoids redundant sampling. Demonstrated on a skier/avalanche SAR-style signal.
- **Leaderless:** EXACT.
- **Broadcast-based:** PARTIAL. Communication is described as asynchronous observation sharing; it is not framed as a per-tick broadcast snapshot.
- **Byte-identical decisions:** DIFFERENT. Explicitly *asynchronous*; per-robot decisions are independent and intentionally diverge to spread the swarm.
- **Bayesian SAR:** PARTIAL. The framing is signal-source localization (a Bayesian-optimization problem), evaluated on an avalanche SAR scenario. It's Bayesian optimization rather than recursive Bayesian filtering on a SAROPS-style PDF.

**How our work differs.** Bayes-Swarm asks each robot to be *different* (local penalization spreads them out via different acquisition values); we ask each drone to *agree* on assignment (byte-identical) and the spread comes from a deterministic centroid-tree decomposition of the manifold. Different intent, different formal property.

### 2.4 Banerjee & Schneider — "DecSTER: Decentralized Multi-Agent Active Search and Tracking when Targets Outnumber Agents" (arXiv 2401.03154, 2024)

- **Algorithm summary.** Sequential Monte Carlo PHD filter for joint search-and-track when targets outnumber agents. Uses Thompson sampling for decentralized decision-making. Claims robustness to unreliable inter-agent communication.
- **Leaderless:** EXACT.
- **Broadcast-based:** DIFFERENT. Asynchronous neighbour communication.
- **Byte-identical decisions:** DIFFERENT. Thompson sampling is *intentionally stochastic and per-agent independent* — divergent decisions are the design.
- **Bayesian SAR:** PARTIAL. Active search-and-tracking, generic.

**How our work differs.** Opposite stance on per-agent decision divergence (DecSTER wants stochastic diversity; we want deterministic identity).

### 2.5 Hollinger & Singh and successors — "Efficient Multi-robot Search for a Moving Target" (2009); GSST (2010-2012); decentralized variants 2013-2015

- **Algorithm summary.** Graph-search (FHPE+SA, GSST) and sampling-based (RIG) methods for multi-robot search; later extensions add decentralized planning under spatial-temporal connectivity constraints.
- **Leaderless:** PARTIAL — some variants assume periodic rendezvous; centralized planning during connected windows.
- **Broadcast-based:** DIFFERENT. Connectivity-graph communication.
- **Byte-identical decisions:** DIFFERENT.
- **Bayesian SAR:** PARTIAL. Search-for-target framing; not always Bayesian (combinatorial search and information-gathering objectives).

**How our work differs.** Different substrate (broadcast vs. graph), different decision property (byte-identical vs. coordinated-during-connectivity), different formalism (Bayesian PDF vs. coverage / pursuit-evasion graph).

### 2.6 Atanasov, Le Ny, Pappas, Daniilidis — "Decentralized active information acquisition: Theory and application to multi-robot SLAM" (ICRA 2015) and Schlotfeldt et al. RAL 2018

- **Algorithm summary.** Coordinate-descent decentralization of an entropy-minimization control problem; square-root information filter. Multi-robot SLAM is the demonstration; SAR is mentioned as a target application.
- **Leaderless:** PARTIAL (coordinate descent is sequential under the hood).
- **Broadcast-based:** DIFFERENT.
- **Byte-identical decisions:** DIFFERENT.
- **Bayesian SAR:** PARTIAL.

### 2.7 Charrow et al. — Cooperative multi-robot estimation / target localization (RSS 2014, ICRA 2015)

- **Algorithm summary.** Information-theoretic target localization across a multi-robot team; centralized planning of measurement actions over a Gaussian or grid posterior.
- **Leaderless:** DIFFERENT (centralized planner).
- **Broadcast-based:** DIFFERENT.
- **Byte-identical decisions:** N/A (centralized).
- **Bayesian SAR:** PARTIAL.

### 2.8 Shirsat et al. — "Multi-Robot Target Search Using Probabilistic Consensus on Discrete Markov Chains" (SSRR 2020)

- **Algorithm summary.** Probabilistic consensus on a Markov-chain occupancy model for multi-robot search.
- **Leaderless:** EXACT.
- **Broadcast-based:** PARTIAL (consensus over a graph).
- **Byte-identical decisions:** DIFFERENT (probabilistic).
- **Bayesian SAR:** PARTIAL (target search; not maritime SAR; the target model is a Markov chain rather than a SAROPS-style drift PDF).

### 2.9 Aguirre, Atasoy Bingöl, Hamann, Kuckling — "Bayesian Decentralized Decision-making for Multi-Robot Systems: Sample-efficient Estimation of Event Rates" (arXiv 2511.22225, late 2025)

- **Algorithm summary.** Conjugate-prior Bayesian framework for swarms estimating hazardous-event rates and adapting per-robot behaviour by individual confidence; finite-state-machine behavioural switching.
- **Leaderless:** EXACT.
- **Broadcast-based:** Not specified; appears to be local sharing.
- **Byte-identical decisions:** DIFFERENT (per-robot confidence drives independent transitions).
- **Bayesian SAR:** PARTIAL (hazardous-environment perception; not SAR).

### 2.10 Hare, Bandyopadhyay & Chung 2018 (already covered in §2.2) plus other LogOP / opinion-pool entries

Same scoring as §2.2.

### Summary table of overlap

| Work | Leaderless | Broadcast | Byte-identical decisions | Bayesian SAR |
|------|-----------|-----------|--------------------------|--------------|
| Bourgault et al. 2003-07 | EXACT | PARTIAL | PARTIAL→DIFFERENT | EXACT |
| Bandyopadhyay-Chung 2014 / Hare et al. 2018 | EXACT | DIFFERENT | DIFFERENT (asymptotic) | PARTIAL |
| Bayes-Swarm (Ghassemi-Chowdhury 2019) | EXACT | PARTIAL | DIFFERENT (asynchronous) | PARTIAL |
| DecSTER (Banerjee-Schneider 2024) | EXACT | DIFFERENT | DIFFERENT (Thompson) | PARTIAL |
| Hollinger-Singh & successors | PARTIAL | DIFFERENT | DIFFERENT | PARTIAL |
| Atanasov-Pappas 2015 | PARTIAL | DIFFERENT | DIFFERENT | PARTIAL |
| Shirsat 2020 | EXACT | PARTIAL | DIFFERENT | PARTIAL |
| Aguirre et al. 2025 | EXACT | unclear | DIFFERENT | PARTIAL |

No row hits four-EXACT.

---

## 3. Adjacent work (catalog, one sentence each)

- **Makarenko & Durrant-Whyte 2006**, "Decentralized Bayesian algorithms for active sensor networks" (Information Fusion) — generic decentralized Bayesian sensing-network framework; ancestor of Bourgault et al.; not SAR-specific and not broadcast.
- **Olfati-Saber, Fax, Murray 2007**, "Consensus and Cooperation in Networked Multi-Agent Systems" — foundational consensus theory; not Bayesian SAR.
- **Hlinka, Hlawatsch, Djurić 2013**, "Distributed Particle Filtering" (IEEE SP Mag survey) — taxonomy of distributed PF approaches; survey, not architecture.
- **Madhushani et al. / Stein-VBP 2023** — Stein variational belief propagation for multi-robot coordination; message-passing on factor graphs, not broadcast and not SAR.
- **Patwardhan et al. (gbpplanner) 2022**, Gaussian Belief Propagation multi-robot planning (arXiv 2203.11618) — distributed factor-graph inference; ad-hoc message schedule, asymptotic agreement.
- **Coppolino, Tian et al. 2024-2025** various distributed PHD / variational Bayesian filters for tracking — generic tracking, not SAR architecture.
- **Cooperative Opinion Pool (Pahliani-Lima 2008)** — robot-team sensor fusion using a third opinion-pool variant; not SAR, not broadcast.
- **Sharma & Hollinger 2014**, "Online decentralized information gathering with spatial-temporal constraints" — decentralized info-gathering under intermittent connectivity; not Bayesian-SAR-specific.
- **Lanillos, Besada-Portas, Lopez-Orozco, Cruz 2013-2017** — Bayesian minimum-time-search planner for multi-UAV (decentralized variants, including cooperative gradient-based negotiation); SAR-flavoured but the decentralization is via negotiation and ETTD optimization, not broadcast / byte-identical decisions.
- **Pérez-Carabaza et al. 2016-2019** — evolutionary multi-UAV planners for minimum-time target detection; centralized or weakly-decentralized; SAR.
- **Mathew & Mezic** ergodic-search work — single-robot or centralized multi-robot ergodic exploration; not the broadcast / leaderless niche.
- **SAROPS itself (Kratzke & Stone 2010 IEEE; Stone et al. AAAI 2014)** — *centralized* operational maritime SAR planner; the comparator we benchmark against, not a competitor on architecture.
- **SwarnRaft (arXiv 2508.00622, 2025)** — Raft-style consensus for GNSS-denied UAV swarms; consensus on positioning/state, not Bayesian SAR.
- **Ivić et al. 2026 "UAV-Supported Maritime Search" (Valun Bay)** — operational maritime SAR with HEDAC ergodic control and YOLOv8; ground-station-coordinated, not leaderless and not broadcast.
- **Aerial swarm collective-perception robotics (Hamann, Valentini, Ebert) 2018-2024** — Bayesian collective perception with binary or k-ary decisions; site-inspection / classification, not SAR localisation.
- **Best, Cliff et al. 2019**, "Dec-MCTS" — decentralized Monte Carlo tree search for multi-robot active perception; communication-time-graph, not broadcast.
- **Renzaglia, Reymann, Lacroix 2018** — multi-UAV decentralized stochastic optimization for environmental monitoring.
- **Indelman, Carlone, Dellaert 2014** — decentralized factor-graph inference; asymptotic.
- **Cunningham, Indelman, Dellaert, "DDF-SAM 2.0" 2013** — DDF for SLAM with anti-factor channel filters; not SAR.
- **"Decentralised Data Fusion: A Graphical Model Approach" (Makarenko et al. 2009)** — graphical-model framing of DDF; lineage of Bourgault.
- **Furukawa, Bourgault et al. 2007** "Recursive Bayesian search-and-tracking using coordinated UAVs for lost targets" — direct extension of Bourgault 2003; same scoring as §2.1.
- **Wong, Bourgault, Furukawa 2005** "Multi-vehicle Bayesian Search for Multiple Lost Targets" — multi-target SAR variant.

---

## 4. Recency check (2024-2026 specifically)

Targeted arXiv cs.MA / cs.RO 2024-2025 results in scope:

- **arXiv 2401.03154** Banerjee & Schneider, *DecSTER* (2024) — covered §2.4.
- **arXiv 2404.08390** Mateo et al., *Collective Bayesian Decision-Making in a Swarm of Miniaturized Robots for Surface Inspection* (2024) — Bayesian binary classification swarm; not SAR; not broadcast / not byte-identical (each robot's confidence is private state).
- **arXiv 2502.14743** *Multi-Agent Coordination across Diverse Applications* (survey, 2025) — general survey; no specific architecture matching ours.
- **arXiv 2503.13415** *Comprehensive Survey on Multi-Agent Cooperative Decision-Making* (2025) — survey.
- **arXiv 2506.18126** *Decentralized Consensus Inference-based Hierarchical RL for Multi-Constrained UAV Pursuit-Evasion Game* (2025) — pursuit-evasion, RL-based, not Bayesian SAR.
- **arXiv 2508.00622** *SwarnRaft* (2025) — Raft consensus for positioning under GNSS-denial; orthogonal.
- **arXiv 2511.22225** Aguirre et al. (late 2025) — covered §2.9.
- **arXiv 2602.08450** Ivić et al. (2026) Valun Bay maritime SAR — covered §2.10 of adjacent.
- **arXiv 2604.07575** *Robust Multi-Agent Target Tracking in Intermittent Communication Environments via Analytical Belief Merging* (2026) — analytic belief merging under intermittent comms; closer in spirit, but for tracking, not SAR; merge-based not broadcast-based; agents' beliefs merge asymptotically rather than per-tick byte-identically.

No 2024-2026 paper found that hits all four conjuncts simultaneously. The closest 2024-2026 work is Banerjee-Schneider's DecSTER (search/track, lossy comms) and Hare-Bandyopadhyay-Chung 2018 (Bayesian + lossy + leaderless), neither of which is broadcast-based or byte-identical-per-tick.

---

## 5. Honest assessment

**Verdict: Partially novel — the combination is not in the literature, but the components are well-established.**

Specifically:

- **The leaderless + Bayesian + SAR** combination is *covered* by Bourgault-Furukawa-Durrant-Whyte 2003-2007 and the broader DDF lineage. Anyone who claims novelty on those three conjuncts is wrong. We must cite Bourgault et al. as the canonical predecessor in the SAR niche.
- **The leaderless + Bayesian + lossy-comms** combination is *covered* by Hare-Bandyopadhyay-Chung 2018 (DBF / LogOP) for tracking, and DecSTER 2024 for search-and-track. The asymptotic-convergence-under-loss story has been written.
- **The novel piece** is **broadcast-as-shared-state with byte-identical per-tick decisions and deterministic majority-vote tie-break under loss**, applied to Bayesian SAR. Specifically:
  - The substrate: a single per-tick shared broadcast slot read identically by all drones, used as a *replicated-state-machine substrate* (RSM in the distributed-systems sense). The decentralized-Bayesian SAR literature uses neighbour-graph DDF with channel filters or consensus rounds; we use broadcast.
  - The decision property: per-tick byte-identicality with explicit ID-based tie-break, not asymptotic posterior agreement. This converts the "agents must agree" problem from a continuous-time consensus problem into a discrete deterministic-function-of-shared-state problem.
  - The graceful-degradation story: deterministic majority-vote when broadcasts are partially lost, with a quantified degradation curve. The DDF / DBF literature handles loss only via asymptotic-convergence guarantees with no per-tick discrete tie-break.
- **No paper found** asserts byte-identical decisions across drones in a Bayesian SAR architecture under benign loss. The closest neighbours (Bourgault DDF, BCF/DBF, Bayes-Swarm, DecSTER, Shirsat probabilistic-consensus, Aguirre 2025, GBP-planner) all settle for asymptotic / approximate / divergent-by-design agreement.

**Defensible novelty claim, plain English:**

> Decentralized Bayesian SAR has a 20-year history (Bourgault et al. 2003 through Hare-Bandyopadhyay-Chung 2018 and successors) but assumes a neighbour-graph DDF or consensus-loop substrate and provides only asymptotic posterior agreement. We replace this substrate with a shared per-tick broadcast medium, turning the multi-agent decision into a deterministic function of broadcast state and yielding byte-identical per-tick decisions across all drones. Under benign communication loss the architecture degrades through a deterministic majority-vote tie-break with a quantified per-tick error bound — a discrete-time guarantee absent from prior decentralized Bayesian SAR work.

**What the paper must do to be defensible:**

1. Cite Bourgault, Furukawa & Durrant-Whyte 2003 (Coordinated decentralized search) and Furukawa et al. 2006 (Recursive Bayesian search-and-tracking) as the canonical decentralized Bayesian SAR baselines.
2. Cite Makarenko & Durrant-Whyte 2006 as the DDF lineage.
3. Cite Bandyopadhyay-Chung 2014 and Hare-Bandyopadhyay-Chung 2018 as the canonical leaderless-Bayesian-with-asymptotic-consensus baselines.
4. Cite Ghassemi-Chowdhury 2019 (Bayes-Swarm) as the leaderless-asynchronous Bayesian-optimization SAR baseline (different formalism).
5. Cite Banerjee-Schneider 2024 (DecSTER) as a recent decentralized active-search baseline with a different decision-divergence stance.
6. Cite Stone / Kratzke 2010 (SAROPS) as the centralized SAR comparator we are functionally equivalent to.
7. Narrow the novelty claim to **the broadcast / byte-identical / deterministic-majority-vote substrate** rather than to "decentralized Bayesian SAR" in general.

**What we cannot honestly claim:**

- That a leaderless decentralized Bayesian SAR architecture is new (Bourgault et al. 2003-2007 makes this claim untenable).
- That decentralized agents agreeing on a target PDF under lossy comms is new (BCF/DBF/DDF literature).
- That swarm Bayesian search applied to maritime SAR is new (Bourgault et al. demonstrated this on a drifting-at-sea target in 2003).

**What we can honestly claim:**

- A broadcast-as-shared-state substrate yielding byte-identical-per-tick decisions across all drones under reliable comms — not previously asserted in the decentralized Bayesian SAR literature, where the corresponding property is asymptotic posterior equivalence.
- A deterministic majority-vote tie-break giving a quantified per-tick degradation under benign loss — sharper than the asymptotic / error-ball guarantees in DBF / channel-filter DDF.
- Empirical equivalence to a SAROPS-class centralized comparator on a maritime SAR ensemble — a benchmark not run on any of the prior decentralized Bayesian SAR architectures (Bourgault et al.'s 2003 evaluation predates SAROPS itself).

---

## Sources

- [Coordinated decentralized search for a lost target in a Bayesian world (Bourgault et al. 2003) — Cornell ASL mirror](http://cornell-asl.org/wiki/images/7/76/Bourgault03coord.pdf)
- [Coordinated search for a lost target in a Bayesian world — Semantic Scholar](https://www.semanticscholar.org/paper/Coordinated-search-for-a-lost-target-in-a-Bayesian-Bourgault-G%C3%B6ktogan/c4e6d228c3ed953eeaf32d81fa9e91b05525dda6)
- [Optimal Search for a Lost Target in a Bayesian World — Semantic Scholar](https://www.semanticscholar.org/paper/Optimal-Search-for-a-Lost-Target-in-a-Bayesian-Bourgault-Furukawa/ca8fb0cb8b5474eee17aa06cf3117a3bf95cc377)
- [Recursive Bayesian search-and-tracking using coordinated UAVs for lost targets (Furukawa et al. 2006)](https://www.researchgate.net/publication/224635213_Recursive_Bayesian_search-and-tracking_using_coordinated_uavs_for_lost_targets)
- [Multi-vehicle Bayesian Search for Multiple Lost Targets (Wong, Bourgault, Furukawa 2005)](https://www.semanticscholar.org/paper/Multi-vehicle-Bayesian-Search-for-Multiple-Lost-Wong-Bourgault/27e6102b2a78861ab1d5f6e37e51afdecafb6858)
- [Distributed Estimation using Bayesian Consensus Filtering (Bandyopadhyay & Chung, arXiv:1403.3117)](https://arxiv.org/abs/1403.3117)
- [Distributed Bayesian Filtering using Logarithmic Opinion Pool for Dynamic Sensor Networks (Hare et al., arXiv:1712.04062)](https://arxiv.org/abs/1712.04062)
- [Decentralized Bayesian algorithms for active sensor networks (Makarenko & Durrant-Whyte 2006)](https://www.sciencedirect.com/science/article/abs/pii/S1566253505000813)
- [Decentralised Data Fusion: A Graphical Model Approach (Makarenko et al. 2009)](https://dellaert.github.io/files/Makarenko09fusion.pdf)
- [Decentralized Informative Path Planning with Exploration-Exploitation Balance for Swarm Robotic Search (Bayes-Swarm, Ghassemi-Chowdhury, arXiv:1905.09988)](https://arxiv.org/abs/1905.09988)
- [Informative Path Planning with Local Penalization for Decentralized and Asynchronous Swarm Robotic Search (arXiv:1907.04396)](https://arxiv.org/abs/1907.04396)
- [DecSTER: Decentralized Multi-Agent Active Search-and-Tracking when Targets Outnumber Agents (Banerjee & Schneider, arXiv:2401.03154)](https://arxiv.org/html/2401.03154v2)
- [Bayesian Decentralized Decision-making for Multi-Robot Systems (Aguirre et al., arXiv:2511.22225)](https://www.arxiv.org/pdf/2511.22225)
- [Collective Bayesian Decision-Making in a Swarm of Miniaturized Robots for Surface Inspection (arXiv:2404.08390)](https://arxiv.org/abs/2404.08390)
- [Multi-Robot Target Search Using Probabilistic Consensus on Discrete Markov Chains (Shirsat et al. 2020)](https://faculty.engineering.asu.edu/acs/wp-content/uploads/sites/33/2020/11/Shirsat_SSRR2020_Multi-Robot-Target-Search-Using-Probabilistic-Consensus-on-Discrete-Markov-Chains.pdf)
- [Decentralized active information acquisition: Theory and application to multi-robot SLAM (Atanasov et al. 2015)](https://existentialrobotics.org/ref/Atanasov_ActiveInformationAcquisition_ICRA15.pdf)
- [Anytime Planning for Decentralized Multirobot Active Information Gathering (Schlotfeldt et al. RAL 2018)](http://erl.ucsd.edu/ref/Schlotfeldt_AnytimeInfoGathering_RAL18.pdf)
- [Search and Rescue Optimal Planning System (Kratzke & Stone, MetSci 2010)](https://www.metsci.com/wp-content/uploads/2019/08/Search-and-Rescue-Optimal-Planning-System.pdf)
- [Search and Rescue with Sparsely Connected Swarms (Autonomous Robots 2022)](https://link.springer.com/article/10.1007/s10514-022-10080-7)
- [Distributing Collaborative Multi-Robot Planning with Gaussian Belief Propagation (gbpplanner, arXiv:2203.11618)](https://arxiv.org/abs/2203.11618)
- [SwarnRaft: Leveraging Consensus for Robust Drone Swarm Coordination in GNSS-Degraded Environments (arXiv:2508.00622)](https://arxiv.org/html/2508.00622v1)
- [UAV-Supported Maritime Search System: Valun Bay Field Trials (Ivić et al., arXiv:2602.08450)](https://arxiv.org/html/2602.08450)
- [Robust Multi-Agent Target Tracking in Intermittent Communication Environments via Analytical Belief Merging (arXiv:2604.07575)](https://arxiv.org/html/2604.07575)
- [A Bayesian Approach for Constrained Multi-Agent Minimum Time Search (Lanillos et al. 2013)](https://www.semanticscholar.org/paper/A-bayesian-approach-for-constrained-multi-agent-in-Lanillos-Ya%C3%B1ez-Zuluaga/624816ab4c9d4d9131c2338800aa72ea55af8442)
