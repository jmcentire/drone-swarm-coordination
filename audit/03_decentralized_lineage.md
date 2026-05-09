# Audit 03 — Decentralized Bayesian Filtering and Consensus-Based Estimation Lineage

**Scope.** This audit characterizes how the decentralized-Bayesian-filtering / consensus-estimation lineage handles the same coordination-and-consistency problem our broadcast-synchronized architecture handles, so we can position our work precisely. The load-bearing output is §7 (honest novelty assessment).

**Methodology and access.** Web searches via Google/Semantic Scholar; full-text fetches attempted for arXiv preprints, university-hosted PDFs, and review summaries. Five of the ten primary sources are paywalled (Elsevier ScienceDirect, IEEE Xplore, Springer) and PDF text extraction failed in this environment for several open-access PDFs (binary streams not parsed). Where algorithm details are not directly accessible, this is flagged in-line so the downstream consumer knows whether they can reproduce a comparator from this audit alone or must source primary papers separately. Key mechanism descriptions below were assembled from publisher abstracts, secondary expositions (the Makarenko–Durrant-Whyte 2009 graphical-model summary; the statwiki summary; the Hlinka et al. 2011 likelihood-consensus arXiv preprint), and the Olfati-Saber–Fax–Murray 2007 abstract page at Caltech CDS.

---

## 1. Grime & Durrant-Whyte (1994); Manyika & Durrant-Whyte (1995 book)

**Algorithm.** Decentralized data fusion via the *information form* of the Kalman filter. Each node maintains an information vector and information matrix; updates from observations are additive in information space, which makes local fusion a sum. The 1994 paper demonstrates an implementation across 150 sensors / 30 processors that "yields an identical result to that obtained in a centralized system" (Grime & Durrant-Whyte 1994, abstract). The 1995 Manyika & Durrant-Whyte book extends this to sensor *management* (information-utility-based action selection) on the same substrate.

**Comms model.** Sensor-to-sensor links over a (typically singly-connected) network. The 1994 paper assumes a fixed, fully-connected-or-tree-like topology; the algorithm requires *every* sensor pair to either share a direct link or be connected through a known path so that information can be summed without duplication. Reliability is implicit — there is no message-loss tolerance discussion in the abstract; the algorithm is correctness-by-construction given that all messages eventually arrive.

**Convergence.** *Byte-identical to centralized*, conditional on the topology constraint and full message delivery. Not asymptotic — once all messages have propagated, every node holds the full centralized posterior in information form. This is the strongest convergence claim in the entire lineage.

**Channel-filter mechanism.** Channel filters are explicit data structures stored on each pair of connected nodes that track *which information has already been communicated across this link*, so when fresh information arrives it can be subtracted out of an incoming message before being added to the local estimate. This prevents the same observation from being counted twice when it arrives via two paths. The mechanism is explicit bookkeeping.

**Access.** Paywalled (Elsevier). Detailed channel-filter formulae not directly extractable in this environment; secondary descriptions (statwiki, Makarenko 2009) are sufficient to characterize the mechanism but not to re-derive it.

## 2. Makarenko & Durrant-Whyte (2004 ICRA; 2006 *Information Fusion*)

**Algorithm.** Decentralized Bayesian fusion *and* decentralized active sensor control on a unified framework. Each node runs a local Bayesian filter; nodes exchange posteriors (or likelihood messages) along communication links; a channel filter on each link prevents double-counting. Active control is layered on top: each node selects its next observation to maximize information gain about the global posterior, computed locally.

**Comms model.** Tree-structured (singly-connected) communication graphs are the clean case. For arbitrary topologies, channel filters are necessary; for non-tree (loopy) graphs, exact channel filters are impossible and the framework falls back to *covariance intersection* or other conservative-fusion approximations. Time-varying topology and message loss are acknowledged failure modes of the channel-filter formulation; the 2009 Makarenko et al. graphical-model paper is partly a response to those limitations and proposes Distributed Junction Trees as a generalization.

**Convergence.** Byte-identical to centralized when (a) the graph is a tree, (b) the node-pair channel filters are exact, (c) all messages eventually arrive. Approximate (conservative, possibly biased toward overestimating uncertainty) when covariance intersection is substituted.

**Channel-filter mechanism.** Same explicit per-link bookkeeping as Grime & Durrant-Whyte. The channel filter holds the marginal of "what has already been communicated through this channel"; subtracting it from incoming messages preserves consistency.

**Access.** Paywalled (Elsevier). Mechanism well-described in the Makarenko 2009 graphical-model paper (open PDF; binary-extraction failed in this environment but the statwiki summary captures the salient points). Pseudocode for a comparator implementation: not directly available from this audit; downstream consumer would need the original.

## 3. Olfati-Saber, Fax & Murray (2007, *Proc. IEEE*)

**Algorithm.** A *survey* paper, not a single algorithm. It synthesizes the consensus-protocol literature: average consensus on graphs, Laplacian-based dynamics, switching topologies, time delays, gossip-based randomized algorithms, flocking, formation control, rendezvous. The unifying mathematical object is the graph Laplacian L, and the unifying claim is that algebraic connectivity λ₂(L) governs convergence rate.

**Comms model.** *Local-neighbor message passing*. Each agent talks only to its direct neighbors in some graph G; G can be fixed, switching, or randomly time-varying (gossip). Broadcast is *not* the dominant model in this lineage — the standard assumption is that each agent has its own neighborhood and exchanges values pairwise.

**Convergence.** Asymptotic. For the standard linear consensus protocol ẋᵢ = Σⱼ aᵢⱼ(xⱼ − xᵢ), all agents converge exponentially to the global average (or some weighted variant) at rate λ₂(L). With switching topology the graph need only be "jointly connected" over time; with delays, a margin condition on λ₂ vs delay must hold. Not byte-identical except in the limit.

**Channel-filter mechanism.** None. The mechanism that prevents pathological double-counting is *idempotent linear averaging*: re-receiving the same value just averages it back to the same fixed point. This is structurally different from channel filters — there is no per-link bookkeeping; instead the algebra of the update is self-correcting. The cost is asymptotic-only convergence (you never reach the centralized estimate exactly in finite time, only get arbitrarily close).

**Access.** Open PDFs at ASU and Murray's Caltech CDS page; binary-extraction failed but the survey's content is well-documented in secondary sources.

## 4. Olfati-Saber (2007 CDC) — Distributed Kalman Filtering

**Algorithm.** The *Kalman-Consensus Filter* (KCF). Each node runs a local Kalman update against its own measurement, then performs one or more rounds of consensus on the resulting state estimate (and optionally on covariance/information matrices) with its graph neighbors. Two variants discussed in the literature: (a) consensus on measurements, then local Kalman; (b) local Kalman, then consensus on estimates. A separate 2005 line embeds two consensus filters (low-pass + band-pass) inside each "micro-Kalman" filter to approximate the centralized information-form sums.

**Comms model.** Local-neighbor pairwise message passing on a graph. Each Kalman cycle interleaves with K consensus rounds. As K → ∞ between Kalman steps, the algorithm approaches the centralized Kalman filter; for finite K, it's an approximation.

**Convergence.** Asymptotic, in mean-square. Stability of the KCF requires graph connectivity and a step-size condition; consensus on covariance is the technically delicate part (averaging information matrices is well-behaved; averaging covariances is not, in general).

**Channel-filter mechanism.** None. Same idempotent-averaging story as the Olfati-Saber–Fax–Murray survey: the protection is "averaging the same number again is a no-op at the fixed point."

**Access.** IEEE-paywalled; arXiv has follow-up work (1703.05438, minimum-time consensus). The basic KCF algorithm is well-characterized in secondary sources but the original CDC paper's proofs are not directly accessible from this audit.

## 5. Boyd, Ghosh, Prabhakar & Shah (2006, *IEEE T-IT*)

**Algorithm.** Randomized gossip averaging. At each tick, a random pair (or random asynchronous wakeup) of nodes (i, j) replaces their values xᵢ, xⱼ with the average (xᵢ + xⱼ)/2. The averaging matrix is doubly stochastic; the network-wide state vector converges to the global average in mean-square.

**Comms model.** Pairwise. Strictly P2P, asynchronous, no broadcast. The graph specifies which pairs are *eligible* to gossip; the random schedule selects which pair fires when.

**Convergence.** Asymptotic in mean-square. Convergence rate is governed by the second-largest eigenvalue of the expected averaging matrix; this is the "averaging time," and the paper's central technical contribution is bounds on it for various graph families.

**Channel-filter mechanism.** None. Gossip's robustness to "double counting" comes from the fact that pairwise averaging is idempotent at the fixed point: once x = average everywhere, gossiping again is a no-op.

**Access.** Open PDF at Boyd's Stanford page (extraction failed in this environment, but the paper is canonical and well-summarized in textbooks).

## 6. Hlinka, Hlawatsch & Djurić (2013, *IEEE Sig. Proc. Magazine*)

**Algorithm.** Survey of distributed particle filtering in agent networks, classified by what's communicated:
- Likelihood-based (each node communicates a parametric approximation of its local likelihood; "likelihood consensus" exploits exponential-family form so consensus on the natural parameters approximates the joint likelihood).
- Posterior-based (communicate posterior approximations).
- Particle-based (communicate particles directly; expensive).

**Comms model.** Local-neighbor message passing on a graph; consensus rounds between particle-filter time steps. No broadcast assumption.

**Convergence.** Asymptotic and approximate. The likelihood-consensus result (Hlinka et al. 2012, arXiv:1108.6214) bounds the approximation error to the joint likelihood as a function of the consensus rounds and the exponential-family fit error. Not byte-identical.

**Channel-filter mechanism.** None. Same idempotent-averaging rationale. The approximations introduced by the consensus rounds are the dominant error source, not double-counting.

**Access.** IEEE Sig. Proc. Mag. is paywalled; the closely related likelihood-consensus paper is on arXiv (1108.6214) and was successfully summarized.

## 7. Bandyopadhyay & Chung (2014 ICRA / 2018 *Automatica*)

**Algorithm.** Distributed Bayesian Filtering using *logarithmic opinion pool* (LogOP). Each agent runs a local Bayesian update on its own measurement; agents then combine their normalized likelihood functions via LogOP using *dynamic average consensus*. The LogOP minimizes the sum of KL divergences from each agent's local posterior to a consensual posterior, which is the natural "Bayesian-correct" consensus criterion.

**Comms model.** Time-varying network of heterogeneous agents. Local-neighbor pairwise messaging suffices for the dynamic-average-consensus inner loop. No broadcast assumption; the graph need only satisfy joint connectivity over time.

**Convergence.** *Globally exponentially convergent to an error ball* centered on the centralized joint posterior. The error ball's radius is bounded explicitly in terms of the target dynamics' time-scale, the consensus step size, and the modeling/communication error bounds. This is the strongest distributed-Bayesian convergence result in the lineage — but it is *not* byte-identical: the agents converge to a neighborhood, not to the exact centralized value.

**Channel-filter mechanism.** None. LogOP combined with KL-minimization is provably *not* a double-counting-vulnerable operation: pooling the same likelihood again with KL-minimization recovers the same fixed point. The 2018 *Automatica* paper proves this cleanly. This is the modern Bayesian-correct alternative to channel filters.

**Access.** arXiv preprint 1712.04062 (open); the 2014 conference precursor at arXiv:1403.3117 (open). Abstract and summary sufficient to characterize convergence type and consensus mechanism. Pseudo-code is in Algorithm 1 of the arXiv preprint but text extraction failed; downstream consumer wanting to implement this comparator would need to fetch the PDF natively.

## 8. Hollinger & Singh (2010 ICRA) — Periodic Connectivity

**Citation correction (2026-05-08):** The earlier draft of this audit attributed this work to "2010 ICRA / 2012 T-RO." The user-supplied correction confirms the canonical reference is **Hollinger, G. & Singh, S. (2010). "Multi-robot coordination with periodic connectivity." 2010 IEEE International Conference on Robotics and Automation, 4457–4462. DOI: 10.1109/robot.2010.5509175.** No T-RO version exists. Tateo et al. (2018) AAAI "Multiagent Connected Path Planning: PSPACE-Completeness and How to Deal With It" (DOI 10.1609/aaai.v32i1.11587) is the formal-complexity follow-on.

**Algorithm.** Multirobot coordination under the constraint that the team need only be *connected at periodic intervals*, not continuously. Plans alternate between "spread out and search/cover" phases and "rendezvous and exchange data" phases. An online algorithm scaling linearly in robot count handles arbitrary periodic-connectivity constraints; theoretical inapproximability bounds for connectivity-constrained planning are also provided.

**Comms model.** *Intermittent connectivity by design.* Communication is assumed to fail outside of rendezvous windows; the contribution is the *scheduling* of rendezvous, not the filtering algorithm itself.

**Convergence.** Not a Bayesian-filter paper — convergence in the estimation sense doesn't apply; the relevant guarantees are coverage / search-quality bounds under the connectivity constraint.

**Channel-filter mechanism.** Not addressed at the filter level. The decoupling is structural: data fusion happens at rendezvous, when topology is briefly tree-like or fully-connected; coordination/planning happens between rendezvous on local information.

**Access.** Open PDF at CMU RI; binary extraction failed but the abstract is sufficient to position the paper.

## 9. Hollinger, Yerramalli, Singh, Mitra & Sukhatme (2011 ICRA / 2015 *T-RO*) — Distributed Data Fusion for Multirobot Search

**Note:** This is a different paper from §8 (Hollinger & Singh 2010 periodic-connectivity). The 2011 ICRA / 2015 T-RO line on distributed data fusion for multirobot search remains the load-bearing source-acquisition gap; the periodic-connectivity work is fully cited above.

**Algorithm.** Decentralized search planning for a single moving target under realistic intermittent (underwater acoustic) communication. Each vehicle maintains a particle-filter / probability-map belief over the target. When two vehicles regain contact after a long disconnection, their beliefs must be fused without overcounting the shared informational history. The paper proposes a fusion technique that "avoid[s] overcounting information" so that combining beliefs from different vehicles never *decreases* search performance vs each vehicle's individual estimate.

**Comms model.** Underwater acoustic, intermittent, lossy, with realistic acoustic-channel simulation. Robots may go disconnected for extended periods. The fusion technique runs at reconnection events; coordination runs on whatever information has been shared so far.

**Convergence.** Conservative fusion guarantees (no performance decrease vs solo) rather than a tight asymptotic-consensus claim. Eventual consistency under sufficient reconnection.

**Channel-filter mechanism.** The "avoid overcounting" technique is in spirit a channel-filter analogue adapted to particle-filter beliefs — track which observations have been communicated so they don't get re-fused. The 2015 T-RO paper is paywalled and the fusion-technique details (whether it's exact channel-filtering, covariance-intersection-style conservative fusion, or a hybrid) are not extractable from the abstract alone. The 2011 ICRA precursor is open-access at USC/CMU but PDF extraction failed in this environment. **Flag for downstream consumer:** this is the closest published lineage to operational SAR with real comms loss, and a comparator implementation requires the original paper.

## 9b. Minelli, Panerati, Kaufmann, Ghedini, Beltrame, Sabattini (2020 *RAS* 124) — Self-Optimization of Resilient Topologies

**Citation:** Minelli, M.; Panerati, J.; Kaufmann, M.; Ghedini, C.; Beltrame, G.; Sabattini, L. (2020). "Self-optimization of resilient topologies for fallible multi-robots." *Robotics and Autonomous Systems* 124. DOI: 10.1016/j.robot.2019.103384. Combines and extends Ghedini, Ribeiro, Sabattini (2017) *Networks*; Panerati et al. (2018) *Autonomous Robots*; Minelli et al. (2019) DARS.

**Algorithm.** Three-component superposition control: `u_i = α_i u^c_i + β_i u^r_i + γ_i u^d_i`, where:
- `u^c` is a connectivity-preservation control that performs gradient descent on `V(λ) = coth(λ − ε)`, ensuring algebraic connectivity λ₂ stays above a threshold (Sabattini, Chopra, Secchi 2013 IJRR).
- `u^r` is a topology-resilience control that drives potentially-vulnerable nodes toward the barycenter of their weakly-connected 2-hop neighbors. Vulnerability `P_κ(v) = |Path_κ(v)| / |Γ(v)|` is estimated locally from 2-hop neighbor information per Ghedini, Secchi, Ribeiro, Sabattini (2015) IFAC SYROCO.
- `u^d` is a desired-objective control (Lennard-Jones potential for area coverage in this paper).

The hyperparameters (α, β, γ) are optimized **online** during the mission by each robot via random search (or augmented Lagrangian / grid search) over the scalarizing function `f_obj = λ₂(t) · A(t)`. Optimization period `O_p` and generated-points budget `G_p` are tuning knobs; the paper recommends `O_p = 50`, `G_p = 250` for real-robot deployment after a sensitivity sweep.

**Comms model.** WiFi-based with simulated range cap (R = 60 cm). Two fault-injection protocols: (i) per-link Bernoulli packet drop ramping 0% → 80% over the run; (ii) permanent hardware failures with exponential MTTF (300 iterations / 16 minutes). Tested with 8 K-Team Khepera IV robots using OptiTrack tracking; Buzz scripting language; ARGoS simulator. Information sharing uses *virtual stigmergy* (Pinciroli, Lee-Brown, Beltrame 2016 BICT) — a consensus-based shared-tuple-space, not broadcast.

**Convergence.** Approximate / asymptotic. Robots optimize independently and may select different gain tuples (Figure 4 in the paper shows per-robot gain divergence). The metric of success is the global f_obj trajectory across many starting configurations and random seeds — there is no byte-identical-decisions claim.

**Channel-filter mechanism.** None explicit. Virtual stigmergy provides eventual-consistency for shared tuples; the connectivity-preservation control ensures λ₂ > ε so consensus can propagate.

**Where ours sits relative to Minelli 2020.** This is the most operationally-similar paper in the audit — same fault-injection regime (transient comms loss + permanent hardware faults), same hardware-class (small differential-drive robots with Wi-Fi), same recourse to online optimization. But the *problem* differs at a load-bearing layer:

- They preserve **algebraic connectivity** (λ₂ > ε) so consensus can run on top. The control objective is topological (keep the graph connected and resilient).
- We assume the broadcast medium is given (star topology or its emulation) and run **deterministic per-tick decision agreement** on top. Our control objective is decision-level (every drone reaches the same next-manifold).

The two are **stacked, not competing.** A real deployment of our architecture on a fallible swarm with imperfect broadcast emulation would benefit directly from Minelli's connectivity-preservation control as the substrate that maintains the broadcast property. Conversely, Minelli's connected swarm benefits from a deterministic-decision substrate when the application requires byte-identical decisions (rather than just connectivity).

**Citation status.** Paper supplied directly by the user 2026-05-08; preprint version pasted into the audit thread. Algorithm details extractable in full from the supplied text. **MUST CITE** in the v1.2 paper as the closest operational neighbor in the connectivity/resilience-control lineage; the Minelli/Panerati/Sabattini group is the contemporary practitioner of fault-tolerant multi-robot connectivity research. Related citations to acknowledge from this lineage: Sabattini, Chopra, Secchi (2013) IJRR; Ghedini, Ribeiro, Sabattini (2017) Networks; Panerati et al. (2018) Autonomous Robots; Yang, Freeman, Gordon, Lynch, Srinivasa, Sukthankar (2010) Automatica; Robuffo Giordano, Franchi, Secchi, Bülthoff (2013) IJRR.

## 9c. Couceiro (2016) — Swarm Robotics for SaR (survey)

**Citation:** Couceiro, M. S. (2016). "An Overview of Swarm Robotics for Search and Rescue Applications." Chapter 13 in *Handbook of Research on Design, Control, and Modeling of Swarm Robotics*, IGI Global. DOI: 10.4018/978-1-4666-9572-6.ch013.

**Relevance.** Survey-level positioning. Argues that SAR operations are the canonical real-world target for swarm robotics due to their large scale, dynamic conditions, harsh/faulty environments, and inter-robot communication challenges. Emphasizes that centralized swarm architectures "are computationally expensive and unsuitable as a large number of robots usually generates very dynamic behaviours that a centralized controller cannot handle" (Şahin 2005), and that "centralized architectures lack robustness as the failure of the centralized entity may compromise the performance of the whole MRS" (Parker 2008a). This is direct doctrinal support for the *operational case* for decentralization in SAR — independent of the algorithmic novelty of any particular architecture.

**Where ours sits.** Couceiro frames SAR as the application domain in which decentralization is a *requirement* rather than a stylistic preference. Cite once in the v1.2 paper's introduction or motivation section to establish that the operational case for distributed SAR is well-established in the survey literature; the contribution is then the specific architecture, not the case for decentralization.

**Access.** Paywalled (IGI Global, $37.50 per chapter). Abstract and Key Terms section sufficient to characterize the survey's positioning; full text not load-bearing for our work since we are citing it for context, not for algorithmic detail.

## 10. Comparison Table

| Lineage | Comms assumption | Convergence type | Channel-filter mechanism | Message-loss tolerance | Applicability to SAR |
|---|---|---|---|---|---|
| Grime & Durrant-Whyte 1994 | Tree topology, reliable | Byte-identical (in info form) | Explicit per-link bookkeeping | None — requires eventual delivery | Static or slow systems, not lossy comms |
| Makarenko-DW 2006 | Tree (exact); arbitrary (approx via CI) | Byte-identical on trees; conservative on loops | Explicit channel filters; CI fallback | Limited — degrades to conservative | Active sensor networks; not validated under benign loss |
| Olfati-Saber, Fax, Murray 2007 (survey) | Local-neighbor; switching topology supported | Asymptotic exponential at rate λ₂(L) | None — idempotent linear averaging | Tolerates jointly-connected switching | Generic; SAR not a focus |
| Olfati-Saber 2007 CDC (KCF) | Local-neighbor + multiple consensus rounds | Asymptotic mean-square | None — averaging idempotent | Tolerates link failures if connectivity preserved | Target tracking, not search |
| Boyd et al. 2006 (gossip) | Pairwise random | Asymptotic mean-square at rate λ₂ | None — pairwise averaging idempotent | Tolerates link failure (just slows convergence) | Generic averaging primitive |
| Hlinka et al. 2013 (DPF survey) | Local-neighbor | Approximate, bounded by consensus rounds | None — exponential-family + consensus | Approximate by construction | Particle-filter target tracking |
| Bandyopadhyay-Chung 2014/2018 | Time-varying, jointly-connected | Globally exp. to error ball | None — LogOP+KL fixed point | Robust to link drops if joint connectivity holds | Heterogeneous distributed estimation; closest to broadcast in spirit |
| Hollinger & Singh 2010 ICRA (periodic) | Intermittent by design, scheduled rendezvous | Coverage bounds via MIPP-PC reduction; NP-hard POMDP | N/A — fusion at rendezvous only | Tolerates arbitrary disconnection within rendezvous period | Search under disconnection, planning-side |
| Tateo et al. 2018 AAAI (PSPACE) | Same as Hollinger-Singh, formalized as Connected Path Planning | Existence/optimality results: PSPACE-complete | N/A | Same as 2010 | Theoretical foundation |
| Hollinger et al. 2015 T-RO (multirobot search) | Intermittent acoustic, lossy | Conservative (no performance decrease) | "Avoid overcounting" technique — details paywalled | Tolerates extended disconnection | **Closest published SAR analogue** |
| Minelli et al. 2020 *RAS* | WiFi mesh, simulated range; tested under per-link drop + hardware failure | Approximate; per-robot gain optimization may diverge | None — virtual stigmergy for shared state | Connectivity preserved by control law (λ₂ > ε) under fault injection | Connectivity/resilience under faults; closest operational neighbor |
| **Ours (broadcast-synchronized)** | Lossless reliable broadcast (star/shared medium) | Byte-identical in one round | Trivial — single shared event stream | Graceful under independent + correlated benign loss via deterministic majority-vote tie-break | Operational SAR; the working assumption |

---

## 11. Where Ours Sits — Direct Comparisons

**vs Bandyopadhyay-Chung (asymptotic to error ball vs byte-identical-after-one-round).** This is *not* the same thing at a fixed point. Bandyopadhyay-Chung converges *exponentially* to a *neighborhood* of the centralized posterior; the neighborhood has nonzero radius determined by the target dynamics' time-scale and the consensus step size. They explicitly do not claim equality to centralized in finite time. Our claim is byte-identical *immediately*, after one broadcast round, conditional on lossless delivery. The distinction is real, but it is also *bought* by a stronger comms assumption (lossless reliable broadcast). The honest framing: we trade asymptotic-on-anything-jointly-connected for byte-identical-on-broadcast. Neither is uniformly better.

**vs Makarenko-Durrant-Whyte (channel filter vs no channel filter).** Our broadcast architecture is a degenerate special case: a star topology where every "channel" is the shared broadcast medium and every node receives the identical event stream. The channel-filter problem — preventing the same observation from being summed twice — does not arise because there is exactly one event-arrival event for each observation, witnessed by every node simultaneously. So "no channel filter" is *defensible as "trivially-correct channel filter,"* not as a claim that we have solved the channel-filter problem in general. The right framing: by choosing a comms primitive that makes double-counting structurally impossible, we eliminate the *need* for channel filters; we do not provide an alternative *construction* of one.

**vs Olfati-Saber (broadcast vs gossip).** Broadcast eliminates a failure mode that gossip protects against: in gossip, the *protocol* is the consensus mechanism, and dropped messages just slow convergence (graph still jointly connected → fixed point still reached). In our architecture, the *medium* is the consensus mechanism, and dropped messages can cause divergence if not handled (this is exactly what our majority-vote tie-break and the bench_loss / bench_comms experiments address). So broadcast adds an infrastructure dependency (a reliable-enough shared channel) that gossip does not require. The gain is determinism and one-round consensus; the cost is the dependency. This is a real engineering trade, not a free lunch.

---

## 12. Honest Novelty Assessment

The four candidate novelty claims (a)–(d) from the audit prompt, each evaluated against the lineage:

**(a) Byte-identical determinism rather than approximate consensus.**
*Survives, with caveats.* Grime-Durrant-Whyte 1994 also achieves byte-identical equality to centralized — but only in the information form on tree topologies and only in the limit of full message delivery, with explicit channel-filter bookkeeping. Bandyopadhyay-Chung 2018 explicitly does *not* claim byte-identicality; their result is an error ball. Olfati-Saber and Boyd are asymptotic. So byte-identicality after a single broadcast round is novel *vs the consensus-on-graph branch* (Olfati-Saber, Boyd, Bandyopadhyay-Chung, Hlinka), but is *not novel vs the channel-filter branch* (Grime-DW, Makarenko-DW), which also achieves it under their (different and more restrictive) assumptions. The genuine claim is: byte-identicality without channel-filter bookkeeping, by choosing a comms primitive that makes the bookkeeping trivial.

**(b) Star topology with implicit double-count avoidance (no channel filter).**
*Survives as a positioning claim, but it is more accurately framed as "we sidestep the channel-filter problem by adopting a comms model in which it does not arise" rather than as a novel solution to the channel-filter problem.* The decision to use a shared broadcast medium is itself a known engineering pattern (it is how UAV C2 datalinks, Ethernet broadcast domains, and many cooperative-driving stacks work); what is less common is to *commit to it as the foundation* and design the algorithmic stack to require nothing more than it provides. The novelty here is architectural choice and downstream consequences, not a mechanism nobody has thought of.

**(c) Operational SAR application rather than generic target tracking.**
*Survives — qualified.* Hollinger et al. 2015 is explicitly multirobot search under realistic (acoustic, intermittent, lossy) communications, which is operationally closer to SAR than any other paper in this lineage. Their fusion technique handles overcounting and produces conservative guarantees; we have not been able to extract enough algorithm detail in this audit to compare directly. **This is the lineage's closest existing competitor and the one against which we most need to position carefully.** If the Hollinger-2015 fusion technique is essentially a particle-filter channel filter, then claim (c) reduces to "different comms assumption (broadcast vs intermittent acoustic)" rather than "new SAR algorithm." If it is something weaker, the gap is real. **Action item for downstream consumer:** source the 2015 T-RO paper directly and document the fusion mechanism in detail.

**(d) Graceful degradation under independent and correlated benign loss via deterministic majority-vote tie-break.**
*Survives, conditionally.* The lineage we audited treats message loss either (i) as a slow-down of asymptotic convergence (gossip, Olfati-Saber, Bandyopadhyay-Chung) or (ii) as a topology event handled at rendezvous (Hollinger-Singh, Hollinger et al.). None of them, as documented in the accessible material, defines a deterministic tie-break protocol that produces byte-identical agreement *across the surviving nodes* under correlated loss patterns while preserving the one-round consensus property. This is a defensible novelty *if* (and only if) the bench_comms and bench_loss empirical results in our paper hold up under adversarial loss patterns we haven't yet tested — which is, separately, exactly the scope of the comms-layer empirical validation in v1.1.

### Summary

| Claim | Verdict | Caveat |
|---|---|---|
| (a) Byte-identical | Partial — novel vs consensus-on-graph branch; matched by channel-filter branch under different assumptions | Stronger comms assumption (lossless broadcast) bought it |
| (b) Star + implicit double-count avoidance | Survives as positioning, not as new mechanism | We avoid the problem rather than solve it |
| (c) Operational SAR | Survives, qualified — but **Hollinger-2015 is the live competitor** | Need primary source to confirm gap |
| (d) Graceful degradation via deterministic majority-vote tie-break | Survives, conditionally | Conditional on empirical validation under correlated-loss adversarial patterns |

### Recommended framing for the paper

Drop unqualified novelty language for (a) and (b). Replace with: *"By committing to lossless reliable broadcast as the foundational comms primitive — a known engineering pattern in UAV C2 datalinks — we obtain byte-identical one-round consensus and structurally eliminate the channel-filter bookkeeping that the decentralized-Bayesian-filtering lineage (Grime & Durrant-Whyte 1994; Makarenko & Durrant-Whyte 2006) developed for arbitrary topologies and the consensus-on-graph lineage (Olfati-Saber 2007; Boyd et al. 2006; Bandyopadhyay & Chung 2018) replaced with idempotent averaging at the cost of asymptotic-only convergence."*

For (c), explicitly contrast with Hollinger et al. 2015 once that paper's fusion technique is documented; do not claim novelty over it without evidence.

For (d), present the deterministic tie-break and the empirical loss-tolerance results as the load-bearing novel contribution, since this is where the lineage is genuinely thinnest.

---

## 13. Implementation-Comparator Feasibility

| Lineage | Can we implement a comparator from this audit? |
|---|---|
| Grime-DW 1994 | No — channel-filter formulae paywalled; need primary source |
| Makarenko-DW 2006 | No — same; secondary descriptions are conceptual not implementable |
| Olfati-Saber-Fax-Murray 2007 | Yes — average consensus is canonical; fits our existing stack as a baseline |
| Olfati-Saber 2007 CDC (KCF) | Partially — algorithm sketch is in secondary sources; details on covariance consensus need primary |
| Boyd et al. 2006 (gossip) | Yes — pairwise random averaging is textbook |
| Hlinka et al. 2013 | Partially — likelihood consensus on arXiv 1108.6214 sufficient for a Gaussian-PF comparator |
| Bandyopadhyay-Chung 2018 | Partially — algorithm sketch on arXiv 1712.04062, pseudo-code referenced as Algorithm 1 but extraction failed; need primary PDF for full implementation |
| Hollinger-Singh 2012 | No — periodic-connectivity scheduler is the contribution; not directly comparable to our broadcast architecture |
| Hollinger et al. 2015 | **No, and this is the load-bearing gap.** Need primary source |

---

## Sources accessed

- [Grime & Durrant-Whyte 1994 (Elsevier)](https://www.sciencedirect.com/science/article/abs/pii/0967066194903492) — abstract only
- [Manyika & Durrant-Whyte 1995 book listing](https://searchworks.stanford.edu/view/2878476)
- [Makarenko & Durrant-Whyte 2006 (Elsevier)](https://www.sciencedirect.com/science/article/abs/pii/S1566253505000813) — abstract only
- [Makarenko 2009 graphical-model decentralised data fusion](https://dellaert.github.io/files/Makarenko09fusion.pdf) — fetched as binary, secondary summary used
- [Decentralised Data Fusion: Graphical Model statwiki summary](https://wiki.math.uwaterloo.ca/statwiki/index.php?title=Decentralised_Data_Fusion:_A_Graphical_Model_Approach_(Summary))
- [Olfati-Saber, Fax, Murray 2007 PIEEE (Caltech CDS)](https://www.cds.caltech.edu/~murray/papers/2007c_ofm07-ieeeproc.html)
- [Olfati-Saber 2007 CDC (Semantic Scholar)](https://www.semanticscholar.org/paper/Distributed-Kalman-filtering-for-sensor-networks-Olfati-Saber/8273ba70c47c0c7a31b6637e8996302c69ba0099)
- [Boyd, Ghosh, Prabhakar, Shah 2006 (Stanford)](https://web.stanford.edu/~boyd/papers/pdf/gossip.pdf)
- [Hlinka, Hlawatsch, Djurić — Likelihood Consensus arXiv 1108.6214](https://arxiv.org/abs/1108.6214)
- [Bandyopadhyay & Chung 2018 *Automatica* preprint arXiv:1712.04062](https://arxiv.org/abs/1712.04062)
- [Bandyopadhyay & Chung 2014 BCF preprint arXiv:1403.3117](https://arxiv.org/abs/1403.3117)
- [Hollinger & Singh 2012 T-RO (CMU RI)](https://www.ri.cmu.edu/publications/multi-robot-coordination-with-periodic-connectivity/)
- [Hollinger et al. 2011 ICRA Underwater Search PDF (USC)](http://robotics.usc.edu/~geoff/files/HollingerICRA11.pdf) — certificate expired in this environment
- [Hollinger et al. 2015 T-RO listing (USC RESL)](https://uscresl.org/publication/distributed-data-fusion-for-multirobot-search/) — bibliographic only

**Inaccessible without paid retrieval or local PDF reader:** full text of Grime-DW 1994, Makarenko-DW 2006, Olfati-Saber 2007 CDC, Hollinger 2012 T-RO journal version, Hollinger 2015 T-RO. The Hollinger 2015 paper is the most important one to source directly for paper-positioning purposes.
