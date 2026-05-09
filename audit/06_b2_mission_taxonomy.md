# Phase B2 — Mission-Class Taxonomy and Scope Boundary

**Date:** 2026-05-08
**Phase:** B2 (mission-class taxonomy / scope boundary) of v7 plan
**Purpose:** Define the categorical distinction between benign-loss, adversarial-DoS, and Byzantine mission classes, ground each definition in published threat-model literature, and produce a scope statement that names what our broadcast-synchronized Bayesian SAR architecture is designed for and what it is explicitly not.
**Output role:** the *Mapping table* (Section 3) and the *Scope statement* (Section 7) drop verbatim into the paper's Limitations / Threat Model section.

---

## 1. Why a mission-class taxonomy is the load-bearing scope artifact

Our architecture's contribution (per `00_synthesis.md` (d)) is byte-identical per-tick decisions under a *deterministic-decision substrate*, with measured graceful degradation under *benign* loss patterns. Every claim of correctness, convergence, or graceful degradation depends on the comms layer behaving as a *non-adversarial* random process. The moment an adversary controls *which* messages are delivered (or worse, *what* messages contain), the architecture's invariants no longer follow from its proofs — they would have to follow from a different proof against a different adversary.

The taxonomy below makes this scope boundary explicit and defensible. It is grounded in the established threat-model literature (Lamport-Shostak-Pease 1982 for Byzantine; Castro-Liskov 1999 for the cryptographic-Byzantine system model; Lamport 1998 for crash-only / non-Byzantine consensus; Gilbert 1960 / Elliott 1963 for benign bursty-loss channel models; Pelechrinis-Iliofotou-Krishnamurthy 2011 and Xu et al. 2005 for jamming/DoS taxonomies; Hu-Perrig-Johnson 2003 for the DoS-vs-Byzantine distinction in wireless ad-hoc networks). Each mission-class boundary statement traces to a foundational paper.

## 2. Mission-class definitions

Each class is given a formal threat-model characterization in the form: *(adversary capabilities, message-level effects, observable comms patterns, foundational reference)*.

### 2.1 Class B — Benign-loss missions (in scope)

**Adversary capabilities.** None. Comms loss is environmental and stochastic. No actor is intentionally selecting which messages are delivered, dropped, or corrupted. The "adversary" is the channel: multipath, shadowing, range, weather, occlusion, thermocline, ambient acoustic noise, satellite line-of-sight geometry, etc. This corresponds to Lamport's (1998) Paxos crash-failure model extended to the network layer: messages may be lost, delayed, duplicated, or reordered, but never *forged* and never *selectively dropped by an adversary modeling the protocol* ([Lamport 1998](https://lamport.azurewebsites.net/pubs/lamport-paxos.pdf); [Wikipedia: Paxos](https://en.wikipedia.org/wiki/Paxos_(computer_science))).

**Message-level effects.** A broadcast either arrives intact at a given receiver, or does not arrive at all. Cryptographic integrity (collision-resistant digest, optionally signed) detects corrupted-in-flight packets and discards them — i.e., bit-flip noise in transit is reduced to packet erasure, the standard channel-coding-after-FEC abstraction. There are no maliciously-crafted messages.

**Observable comms patterns.** Three sub-classes, drawn from the channel-modeling literature:

- **B-IID — independent Bernoulli loss.** Each broadcast at each receiver is independently dropped with probability *p*. The standard fluid-approximation model used in distributed-consensus convergence analysis ([Boyd et al. 2006 randomized gossip](https://web.stanford.edu/~boyd/papers/gossip.html); also the model assumed by Bayesian-consensus-filter convergence proofs in Bandyopadhyay-Chung 2014 and Hare-Bandyopadhyay-Chung 2018).
- **B-GE — Gilbert-Elliott bursty / correlated loss.** A two-state Markov chain (Good / Bad) governs the receiver, with within-state Bernoulli loss probabilities. Originally proposed by Gilbert (1960, "Capacity of a Burst-Noise Channel") and extended by Elliott (1963, "Estimates of Error Rates for Codes on Burst-Noise Channels"), this is the canonical model for bursty real-channel loss caused by RF shadowing, fading, or environmental occlusion ([Gilbert-Elliott Model summary](https://people.computing.clemson.edu/~jmarty/projects/lowLatencyNetworking/papers/APPFEC/GEModelForLossinTheRTInternet.pdf); [Burst error – Wikipedia](https://en.wikipedia.org/wiki/Burst_error)). The Sozer-Stojanovic-Proakis (2000) underwater acoustic channel and the standard mountainous-terrain UAV link both fall in this regime ([Sozer-Stojanovic-Proakis 2000](https://www.scirp.org/reference/referencespapers.aspx?referenceid=891140)).
- **B-ASYM — geometric / asymmetric loss.** The set of receivers that successfully receive a given broadcast is determined by per-link budget (range, line-of-sight, antenna gain, terrain shadow). At a given tick, a subset of drones receive a broadcast; the complementary subset does not. Loss is *correlated across receivers in space* but not driven by an adversary modeling the protocol.

**Foundational references.** Gilbert 1960; Elliott 1963; Lamport 1998 (Paxos crash-failure model). Application-domain channel modelling: Sozer-Stojanovic-Proakis 2000 (underwater acoustic); standard UAV link-budget literature.

### 2.2 Class D — Adversarial DoS missions (out of scope)

**Adversary capabilities.** An external attacker can prevent message delivery (jam) but cannot forge, modify, or selectively rewrite the *content* of legitimate messages. The attacker is on the *channel*, not on the *platform*. Hu, Perrig & Johnson (2003) explicitly identify this as a distinct threat class: "the wormhole attack is possible even if the attacker has not compromised any hosts, and even if all communication provides authenticity and confidentiality" — i.e., DoS-class attacks operate independently of cryptographic platform integrity ([Hu-Perrig-Johnson 2003 Packet Leashes](https://www.cs.rice.edu/~dbj/pubs/jsac-wormhole.pdf)).

The DoS sub-taxonomy follows Xu et al. (2005) and is comprehensively surveyed in Pelechrinis, Iliofotou, & Krishnamurthy (2011) ([Pelechrinis et al. 2011](https://www.semanticscholar.org/paper/Denial-of-Service-Attacks-in-Wireless-Networks:-The-Pelechrinis-Iliofotou/f44ed06c868f97b24a28df856646a931318f67e9)):

- **Constant jammer** — broadcasts noise without protocol awareness.
- **Deceptive jammer** — transmits legitimate-looking packets at high rate to occupy the medium.
- **Random jammer** — alternates jam/sleep on a stochastic schedule; produces an apparent loss process that *can statistically mimic Gilbert-Elliott bursts* (the transition case of Section 4).
- **Reactive jammer** — listens passively and jams only on detected legitimate traffic; Xu-Trappe (2005) demonstrate that reactive-jammer RSSI signatures are "too similar [to normal traffic] to rely on spectral discrimination" ([Xu-Trappe et al. 2005](https://scholarcommons.sc.edu/cgi/viewcontent.cgi?article=1018&context=csce_facpub)).

**Message-level effects.** Same as B (intact-or-erased) — under cryptographic packet authentication. Without authentication, the adversary can in principle inject arbitrary packets; but at that point the threat model has crossed into Byzantine territory (cf. Section 2.3). The DoS class is canonically defined *under* a cryptographic-authentication assumption ([Pelechrinis et al. 2011](https://www.semanticscholar.org/paper/Denial-of-Service-Attacks-in-Wireless-Networks:-The-Pelechrinis-Iliofotou/f44ed06c868f97b24a28df856646a931318f67e9)).

**Observable comms patterns.**

- **D-UNIFORM** — all receivers experience similar elevated loss (broad-band jammer).
- **D-SELECTIVE** — high-information drones (e.g., the drone broadcasting the freshest scenario update, or the drone with the highest cell-likelihood mass) are jammed preferentially. This is the protocol-aware adversary.
- **D-INTERMITTENT** — jammer turns on and off, producing apparent burst-loss that statistically resembles Gilbert-Elliott (the Section 4 transition case).

**Foundational references.** Hu-Perrig-Johnson 2003; Xu-Trappe et al. 2005; Pelechrinis-Iliofotou-Krishnamurthy 2011.

### 2.3 Class Z — Byzantine missions (out of scope)

**Adversary capabilities.** Compromised drones — or an adversary with platform-level access — can broadcast arbitrary content: false beliefs, forged identities, selectively-rewritten messages, protocol-violating sequences. Lamport, Shostak & Pease (1982) define this as the *Byzantine* failure model: a faulty process can "send misleading or inconsistent messages" and there is no a priori restriction on its behaviour ([Lamport-Shostak-Pease 1982](https://lamport.azurewebsites.net/pubs/byz.pdf); [ACM TOPLAS DOI](https://dl.acm.org/doi/10.1145/357172.357176)). Castro & Liskov (1999) sharpen this: the PBFT threat model "allows for a very strong adversary that can control faulty nodes and the network in order to cause the most damage … coordinating faulty nodes and delaying messages" ([Castro-Liskov 1999 PBFT](https://css.csail.mit.edu/6.824/2014/papers/castro-practicalbft.pdf)).

**Message-level effects.** Arbitrary. Forged content; replayed content; identity spoofing; selectively-dropped (where the adversary picks *which* legitimate message to suppress before relaying); maliciously-crafted-to-collide content.

**Observable comms patterns.** No simple statistical fingerprint. The Byzantine adversary can mimic any benign-loss or DoS signature *and* additionally inject untrue content. The defining feature is content-level corruption, not just delivery failure.

**Foundational references.** Lamport-Shostak-Pease 1982; Castro-Liskov 1999.

**Bound:** Classical Byzantine consensus requires *3f + 1* total replicas to tolerate *f* Byzantine nodes ([Lamport 1982](https://lamport.azurewebsites.net/pubs/byz.pdf); [Castro-Liskov 1999](https://css.csail.mit.edu/6.824/2014/papers/castro-practicalbft.pdf)). The mobile-robot literature has extended this to *f-locally bounded* Byzantine adversaries on graphs (Mitra & Sundaram 2018, [arXiv:1802.09651](https://arxiv.org/abs/1802.09651)) and to resilient distributed *estimation* (Tzoumas et al. and follow-on work in [Resilient distributed state estimation with mobile agents](https://link.springer.com/article/10.1007/s10514-018-9813-7)). All Byzantine-resilient architectures require either (a) *3f+1* replication, or (b) a graph-connectivity assumption (e.g., (2f+1)-robust graphs), or (c) a cryptographic primitive (signed messages). Our architecture does none of these.

## 3. Mapping table — mission class × appropriate architecture

This table is the load-bearing artifact for procurement / mission-planner use. *Rows*: mission classes and benign-loss sub-classes. *Columns*: candidate architectures from the literature. *Cells*: appropriate / partially appropriate / inappropriate, with reason.

| Mission class | Ours (broadcast-synchronized Bayesian SAR) | Channel-filter DDF (Grime / Manyika / Makarenko-Durrant-Whyte) | Bayesian-consensus filter (Bandyopadhyay-Chung 2014; Hare-Bandyopadhyay-Chung 2018) | Classical BFT (Lamport 1982; Castro-Liskov 1999) | Byzantine-resilient distributed estimation (Mitra-Sundaram 2018; Tzoumas et al.) | Centralized SAROPS (Kratzke-Stone-Frost 2010) |
|---|---|---|---|---|---|---|
| **B-IID** (independent Bernoulli loss) | **Appropriate.** Design envelope. Byte-identical decisions under no-loss; measured graceful degradation as *p* increases (bench_loss.py). | Appropriate when topology is a tree and delivery is in-order. | Appropriate; classical convergence proof assumes IID-like consensus-step success. | Overkill — adds 3f+1 replication cost without threat-model justification. | Overkill — adds graph-robustness cost without threat-model justification. | Appropriate if a reliable backhaul exists; defeats the decentralization motivation otherwise. |
| **B-GE** (Gilbert-Elliott bursts) | **Appropriate.** Bench (`bench_comms.py` Sweep B) explicitly characterises decision divergence under burst loss; majority-vote fallback handles bounded transient disagreement. | Partially appropriate — channel filter recovers if bursts are short relative to consensus rate; long bursts violate in-order delivery assumption. | Partially appropriate — convergence rate degrades with burst length; no per-tick decision guarantee. | Inappropriate — wrong threat model. | Inappropriate — wrong threat model. | Inappropriate if backhaul itself is bursty. |
| **B-ASYM** (geometric / link-budget) | **Appropriate.** This is the modal SAR case (mountains, weather, range). Bench (Sweep on asymmetric loss) characterises decision divergence under per-link asymmetry. | Inappropriate if topology is not a tree; cycle-bias problem. | Appropriate (consensus on graph handles asymmetric edge weights), but no per-tick decision guarantee. | Inappropriate — wrong threat model. | Inappropriate — wrong threat model. | Inappropriate — single point of failure. |
| **D-UNIFORM** (uniform jamming) | **Out of scope** — but degrades to "elevated B-IID" as long as the jammer is not protocol-aware. See Section 4 transition case (a). | Out of scope. | Out of scope. | Inappropriate — Byzantine architecture, not DoS architecture. | Inappropriate — same reason. | Inappropriate. |
| **D-SELECTIVE** (jamming high-information drones) | **Out of scope.** No graceful-degradation guarantee; the adversary can structurally bias which information reaches consensus. Section 4(b). | Out of scope. | Out of scope without a robust-aggregation extension. | Inappropriate — wrong adversary class. | Inappropriate — wrong adversary class (DoS, not Byzantine). | Inappropriate. |
| **D-INTERMITTENT** (jam-on/off mimicking GE bursts) | **Out of scope, but the transition case is benchmarked.** If the adversary's statistical signature is indistinguishable from B-GE, our architecture degrades as if the jammer were a channel — a feature in this case. If the adversary is protocol-aware (jams to maximise decision divergence), no guarantee. | Out of scope. | Out of scope. | Inappropriate — wrong adversary class. | Inappropriate — wrong adversary class. | Inappropriate. |
| **Z (Byzantine, full)** | **Out of scope.** No signed messages, no quorum, no f-local-robustness assumption. Forged messages would be accepted as legitimate broadcasts. | Out of scope — channel filter assumes honest peers. | Out of scope — convergence proof assumes honest peers. | **Appropriate.** Classical 3f+1 architecture with cryptographic signatures. | **Appropriate** when graph topology supports (2f+1)-robust assumptions. | Out of scope — single point of compromise. |
| **Z-PARTIAL (1–2 compromised drones, otherwise honest)** | **Out of scope, but the transition case matters.** See Section 4(c). | Out of scope. | Partially appropriate via robust-aggregation extensions (e.g., trimmed log-opinion-pool). | **Appropriate** if 3f+1 holds (typically requires N ≥ 4 for f=1, N ≥ 7 for f=2). | **Appropriate** with f-local bound. | Out of scope. |

## 4. Transition cases — boundary behaviour

The boundary between mission classes is not crisp in deployment, even if it is crisp in formal specification. Three transition cases matter for our architecture:

### 4.1 Transition (a): DoS that statistically mimics benign loss

Setting: D-UNIFORM or D-INTERMITTENT jamming whose loss process is statistically indistinguishable from B-IID or B-GE. The adversary is not protocol-aware — it does not select *which* drone or *which* message to jam in order to bias the swarm's consensus.

Behaviour of our architecture: **graceful degradation, indistinguishable from benign-loss case.** This is a *feature*, not a coincidence. Our convergence and degradation analysis depends only on the marginal loss process at each receiver; it does not depend on whether that loss was generated by the channel or by an adversary, *provided the adversary is not modelling the protocol*.

Recommended response: **none required.** The architecture's bench-measured degradation envelope (`bench_comms.py`) covers this case to the extent that the adversary's statistical signature is bounded by our envelope's parameters. This is the case Phase B1 grounds quantitatively.

Caveat: this is NOT a claim of DoS-resilience. It is a claim that the architecture's failure mode under benign loss continues to apply when the loss is generated by a non-protocol-aware adversary. A protocol-aware adversary breaks this — see (b).

### 4.2 Transition (b): Protocol-aware DoS (D-SELECTIVE)

Setting: the adversary observes traffic, identifies the drone with the highest information content (e.g., the drone broadcasting the freshest update, or the drone whose ID would win a tie-break), and jams it preferentially. Reactive jammers (Xu et al. 2005) are an example.

Behaviour of our architecture: **collapse, not graceful degradation.** The deterministic ID-based tie-break that produces byte-identical decisions under benign loss now produces *systematically biased* decisions under selective jamming, because the messages from the deterministically-prioritized drones are systematically suppressed. The graceful-degradation guarantee (which assumes the marginal loss process is symmetric across drones in expectation) does not hold.

Recommended response: **explicit detection or fall-back to a stronger architecture.** Either (i) instrument the swarm to detect protocol-aware jamming via the standard Pelechrinis et al. (2011) detection mechanisms (RSSI consistency, packet-delivery-ratio anomaly, location-aware consistency check), and *abort to a centralized fallback* on detection; or (ii) deploy a different architecture entirely (Bayesian-consensus-filter with a robust-aggregation primitive, or a cryptographically-authenticated quorum protocol). The architecture should not silently continue under D-SELECTIVE conditions.

### 4.3 Transition (c): Partial Byzantine (1–2 compromised drones, otherwise honest channel)

Setting: a small number of drones have been compromised at the platform level (cyber-intrusion, supply-chain compromise, hostile insertion) and broadcast arbitrary content. The channel itself is benign.

Behaviour of our architecture: **collapse.** Without message authentication or quorum, a single compromised drone can broadcast a forged "high-likelihood" cell update that *every other drone accepts as ground truth in the next consensus pool*. The deterministic tie-break amplifies the compromise — once the forged message is in the broadcast snapshot, every honest drone's decision rule deterministically incorporates it. There is no "graceful degradation" here; the contamination propagates at consensus speed.

Recommended response: **fall back to a stronger architecture.** Z-PARTIAL is in the design envelope of either classical BFT (with cryptographic signatures and 3f+1 replication; [Castro-Liskov 1999](https://css.csail.mit.edu/6.824/2014/papers/castro-practicalbft.pdf)) or Byzantine-resilient distributed estimation ([Mitra-Sundaram 2018](https://arxiv.org/abs/1802.09651); [Resilient state estimation with mobile agents](https://link.springer.com/article/10.1007/s10514-018-9813-7)). For a SAR mission with a credible insider-threat model, a hybrid is plausible: our broadcast substrate for nominal operation, layered with signed-message authentication to detect (not tolerate) Byzantine drift, and an automatic transition to a Byzantine-resilient consensus protocol on detection.

## 5. Mission-class identification guide

A short decision tree for a mission planner. Each leaf identifies the in-scope/out-of-scope status with respect to *our* architecture and names the appropriate alternative.

```
Q1. Are there platforms in the swarm that may be compromised, controlled by an
    adversary, or running unverified firmware (insider/supply-chain threat)?

    YES  → Class Z (Byzantine). OUT OF SCOPE for our architecture.
            Use classical BFT (Lamport 1982; Castro-Liskov 1999) if 3f+1 is
            affordable, or Byzantine-resilient distributed estimation
            (Mitra-Sundaram 2018; Tzoumas et al.) if the graph-robustness
            assumption holds.

    NO   → continue to Q2.

Q2. Is there a known adversary in the operational environment with the
    capability and intent to jam radio traffic (electronic warfare, contested
    airspace, GPS-denied environment under hostile control)?

    YES  → Class D (Adversarial DoS). OUT OF SCOPE for our architecture.
            BUT: see Q2a.

    NO   → continue to Q3.

Q2a. Is the jammer expected to be protocol-aware (capable of selectively jamming
     the highest-information drones, observing tie-break IDs, etc.)?

     YES → D-SELECTIVE. Our architecture COLLAPSES; deploy a detection
            mechanism (Pelechrinis et al. 2011) and abort to a centralized
            comparator (SAROPS-class) or to a Byzantine-resilient architecture.

     NO  → D-UNIFORM / D-INTERMITTENT. Our architecture DEGRADES as if the
            channel were elevated benign loss (Section 4(a) transition case).
            Bench coverage applies if the jammer's effective loss rate stays
            within our characterized envelope. This is operating-on-borrowed-
            time, not a claim of DoS-resilience.

Q3. Is comms loss expected only from environmental causes (terrain shadowing,
    weather, range, occlusion, multipath, thermocline, satellite line-of-sight
    geometry)?

    YES  → Class B (Benign loss). IN SCOPE.
            Sub-classify (Q3a, Q3b, Q3c) for parameter selection in the
            comms-envelope bench.

Q3a. IID or correlated?
     - Independent Bernoulli per packet → B-IID.
     - Markov-chain bursts (RF shadow, fading) → B-GE.
     - Determined by per-link geometry (range / line-of-sight / antenna gain)
       → B-ASYM.

Q3b. What is the expected loss-rate range and burst-length range?
     Cross-reference with Phase B1 envelope ranges (`05_b1_envelope_ranges.md`)
     and the comms bench (`bench_comms.py` Sweeps A/B/C/D) to confirm the
     mission falls within the bench's measured-graceful-degradation envelope.

Q3c. Is asymmetric loss correlated with information content
     (high-altitude drones get clear comms; low-altitude / shadowed drones
     drop heavily)?

     YES → check whether the bench's asymmetric-loss sweep covers your
            geometry. If not, the architecture is in scope but not yet
            *bench-validated* for your geometry; treat as B-ASYM with
            an open question.

     NO  → standard B-ASYM, in scope.
```

## 6. Access-barrier notes (per audit protocol)

- **Lamport-Shostak-Pease 1982 (Byzantine Generals).** Open access via Lamport's website ([PDF](https://lamport.azurewebsites.net/pubs/byz.pdf)). ACM DL paywall is bypassed. Full text reviewed.
- **Castro-Liskov 1999 (PBFT).** Open access via MIT CSAIL ([PDF](https://css.csail.mit.edu/6.824/2014/papers/castro-practicalbft.pdf)) and via the [Castro-Liskov 2002 ToCS extended journal version](http://www.pmg.csail.mit.edu/papers/bft-tocs.pdf). Full text reviewed.
- **Lamport 1998 (Paxos / Part-Time Parliament).** Open access via Lamport's website ([PDF](https://lamport.azurewebsites.net/pubs/lamport-paxos.pdf)). Full text reviewed.
- **Gilbert 1960 / Elliott 1963 (burst-noise channel).** Original Bell System Technical Journal papers; not directly accessed (BSTJ archives are partially open). Used widely-cited summary texts ([Haßlinger-Hohlfeld 2008 review](https://people.computing.clemson.edu/~jmarty/projects/lowLatencyNetworking/papers/APPFEC/GEModelForLossinTheRTInternet.pdf); [Wikipedia: Burst error](https://en.wikipedia.org/wiki/Burst_error)) for the standard two-state Markov characterization. The model is sufficiently canonical that secondary sources are reliable for taxonomy use.
- **Sozer-Stojanovic-Proakis 2000 (underwater acoustic networks).** Open access via ResearchGate ([PDF](https://www.researchgate.net/publication/3231120_Underwater_Acoustic_Networks)). Note: published in IEEE Journal of Oceanic Engineering (vol. 25, no. 1), not Communications Magazine as sometimes cited. Full text accessed via secondary mirror.
- **Hu-Perrig-Johnson 2003 (Packet Leashes).** Open access via Rice University ([PDF](https://www.cs.rice.edu/~dbj/pubs/jsac-wormhole.pdf)) and ETH ([PDF](https://netsec.ethz.ch/publications/papers/hu_perrig_johnson_wormhole.pdf)). Full text reviewed.
- **Pelechrinis-Iliofotou-Krishnamurthy 2011 (Jamming Survey).** IEEE Comm. Surveys & Tutorials. Paywalled on Xplore; abstract accessible via [Semantic Scholar](https://www.semanticscholar.org/paper/Denial-of-Service-Attacks-in-Wireless-Networks:-The-Pelechrinis-Iliofotou/f44ed06c868f97b24a28df856646a931318f67e9). Used for jammer taxonomy (constant / deceptive / random / reactive).
- **Xu et al. 2005 (jammer taxonomy, MobiHoc).** Open access via Univ. South Carolina ([PDF](https://scholarcommons.sc.edu/cgi/viewcontent.cgi?article=1018&context=csce_facpub)). Full text reviewed.
- **Bandyopadhyay-Chung 2014; Hare-Bandyopadhyay-Chung 2018.** Open access via arXiv ([1403.3117](https://arxiv.org/abs/1403.3117); [1712.04062](https://arxiv.org/abs/1712.04062)). Both papers assume *cooperative* honest agents — no Byzantine resilience claim — confirmed by abstract/introduction inspection. (This matters for the mapping table: log-opinion-pool consensus is *not* in itself Byzantine-resilient; "Bandyopadhyay-Chung-with-authentication" in the audit's task description refers to the natural extension via signed messages and trimmed aggregation, not to the original 2014/2018 papers.)
- **Mitra-Sundaram 2018 (Byzantine-resilient distributed observers).** Open access via arXiv ([1802.09651](https://arxiv.org/abs/1802.09651)). Full text accessed.
- **Tzoumas et al. resilient distributed estimation.** The closest paper to the description is the Springer Autonomous Robots paper "Resilient distributed state estimation with mobile agents" ([Springer link](https://link.springer.com/article/10.1007/s10514-018-9813-7)); paywalled. Abstract confirms scope (Byzantine + comms loss + intermittent measurements).

No primary source for the mission-class boundaries was unavailable in a way that affected the taxonomy. Two paywalled secondary references (Pelechrinis 2011 full text, Tzoumas et al. full text) are cited for completeness; their classifications are widely-replicated in the open-access literature and are not load-bearing single points of failure for this taxonomy.

## 7. Scope statement (for paper Limitations / Threat Model section, verbatim)

> **Threat model and scope.** This architecture is designed for *benign-loss* missions, in the sense of Lamport (1998): communication loss is environmental, not adversarial. We assume a cryptographically authenticated broadcast layer (collision-resistant digest, optionally signed) that reduces in-flight bit corruption to packet erasure, and we make no further assumption about the loss process beyond the bench-characterized envelope (independent Bernoulli, Gilbert-Elliott bursts, and link-budget asymmetric loss). We do not assume, claim, or measure resilience against (i) *adversarial denial of service*, in the sense of the wireless-jamming taxonomy of Pelechrinis, Iliofotou, & Krishnamurthy (2011) — uniform, selective, or reactive jammers — nor (ii) *Byzantine* faults, in the sense of Lamport, Shostak, & Pease (1982) — compromised platforms broadcasting arbitrary or forged content. Under non-protocol-aware DoS whose loss process falls within our benign-loss envelope, the architecture degrades exactly as it does under benign loss; this is a graceful-degradation property, not a security claim. Under protocol-aware DoS or under any Byzantine fault, the deterministic-decision substrate that produces byte-identical per-tick decisions under benign loss instead produces systematically biased decisions, and the architecture should be replaced by, or layered beneath, an architecture designed for the actual threat model: Practical Byzantine Fault Tolerance (Castro & Liskov, 1999) and *3f+1* replication for full Byzantine resilience; Byzantine-resilient distributed estimation on (2f+1)-robust graphs (Mitra & Sundaram, 2018; Tzoumas et al.) for graph-distributed Byzantine resilience; or, for adversarial-DoS only, a centralized SAROPS-class architecture (Kratzke, Stone, & Frost, 2010) on a hardened backhaul. A mission planner whose mission has a credible insider-threat or jamming-adversary model should consult the mission-class identification guide in Section 5 of this document and select an architecture appropriate to that class; this architecture is not it.

This paragraph is the load-bearing scope artifact. It (a) names the in-scope class (Lamport-1998 benign-loss), (b) names the explicit out-of-scope classes with their canonical references (Pelechrinis-2011 DoS; Lamport-1982 Byzantine), (c) acknowledges the transition case (non-protocol-aware DoS within envelope) without overclaiming it, and (d) names the appropriate alternative architecture for each out-of-scope class.

## 8. What this taxonomy does and does not commit us to

**Commits us to:**
- The benign-loss envelope (B-IID, B-GE, B-ASYM) is the *only* class for which we will report convergence, decision-equivalence, or graceful-degradation results.
- All bench results in `bench_comms.py` are framed as benign-loss results. Any claim that "our architecture also degrades gracefully under DoS" is bounded to the non-protocol-aware sub-case (Section 4(a)).
- The paper's threat-model paragraph is the verbatim Section 7 above; no informal weakening (e.g., "robust to most realistic comms environments") is permissible.

**Does not commit us to:**
- Building or evaluating a Byzantine-resilient extension. That is future work and out of scope for the v1 paper.
- Building or evaluating a DoS-detection layer. Section 4(b) recommends one for deployment but our paper does not measure one.
- Claiming our architecture is "the right choice" for a SAR mission with adversarial concerns. The mapping table in Section 3 explicitly recommends *against* our architecture in those cases.

## 9. Summary

The taxonomy partitions the SAR mission space into three threat-model classes (B benign, D adversarial-DoS, Z Byzantine), with three sub-classes within B (IID, Gilbert-Elliott, asymmetric) and three within D (uniform, selective, intermittent). Each class maps to a canonical foundational paper that defines the adversary. Our architecture's design envelope is exactly Class B; the mapping table recommends specific alternative architectures for Classes D and Z. The transition cases (Section 4) clarify the boundary behaviour, in particular distinguishing *graceful degradation under non-protocol-aware DoS* (which our architecture provides as a side effect of its benign-loss design) from *DoS-resilience* (which our architecture does not claim and does not provide). The scope statement (Section 7) is the verbatim drop-in for the paper's Limitations section.

---

## References (in citation order)

- Lamport, L. (1998). The Part-Time Parliament. *ACM Transactions on Computer Systems*, 16(2). [PDF](https://lamport.azurewebsites.net/pubs/lamport-paxos.pdf)
- Lamport, L., Shostak, R., & Pease, M. (1982). The Byzantine Generals Problem. *ACM TOPLAS*, 4(3), 382–401. [PDF](https://lamport.azurewebsites.net/pubs/byz.pdf) | [DOI](https://dl.acm.org/doi/10.1145/357172.357176)
- Castro, M., & Liskov, B. (1999). Practical Byzantine Fault Tolerance. *OSDI '99*. [PDF](https://css.csail.mit.edu/6.824/2014/papers/castro-practicalbft.pdf)
- Gilbert, E. N. (1960). Capacity of a Burst-Noise Channel. *Bell Sys. Tech. J.*, 39, 1253–1265.
- Elliott, E. O. (1963). Estimates of Error Rates for Codes on Burst-Noise Channels. *Bell Sys. Tech. J.*, 42, 1977–1997. [Summary review](https://people.computing.clemson.edu/~jmarty/projects/lowLatencyNetworking/papers/APPFEC/GEModelForLossinTheRTInternet.pdf)
- Sozer, E. M., Stojanovic, M., & Proakis, J. G. (2000). Underwater Acoustic Networks. *IEEE Journal of Oceanic Engineering*, 25(1), 72–83. [Reference page](https://www.scirp.org/reference/referencespapers.aspx?referenceid=891140)
- Hu, Y.-C., Perrig, A., & Johnson, D. B. (2003). Packet Leashes: A Defense Against Wormhole Attacks in Wireless Networks. *IEEE INFOCOM 2003*. [PDF](https://www.cs.rice.edu/~dbj/pubs/jsac-wormhole.pdf)
- Xu, W., Trappe, W., Zhang, Y., & Wood, T. (2005). The Feasibility of Launching and Detecting Jamming Attacks in Wireless Networks. *MobiHoc 2005*. [PDF](https://scholarcommons.sc.edu/cgi/viewcontent.cgi?article=1018&context=csce_facpub)
- Pelechrinis, K., Iliofotou, M., & Krishnamurthy, S. V. (2011). Denial of Service Attacks in Wireless Networks: The Case of Jammers. *IEEE Communications Surveys & Tutorials*, 13(2), 245–257. [Semantic Scholar](https://www.semanticscholar.org/paper/Denial-of-Service-Attacks-in-Wireless-Networks:-The-Pelechrinis-Iliofotou/f44ed06c868f97b24a28df856646a931318f67e9)
- Bandyopadhyay, S., & Chung, S.-J. (2014). Distributed Estimation using Bayesian Consensus Filtering. [arXiv:1403.3117](https://arxiv.org/abs/1403.3117)
- Hare, J. Z., Bandyopadhyay, S., & Chung, S.-J. (2018). Distributed Bayesian Filtering using Logarithmic Opinion Pool for Dynamic Sensor Networks. [arXiv:1712.04062](https://arxiv.org/abs/1712.04062)
- Mitra, A., & Sundaram, S. (2018/2019). Byzantine-Resilient Distributed Observers for LTI Systems. [arXiv:1802.09651](https://arxiv.org/abs/1802.09651)
- Tzoumas, V., et al. Resilient Distributed State Estimation with Mobile Agents (Springer Autonomous Robots, 2018). [Springer link](https://link.springer.com/article/10.1007/s10514-018-9813-7)
- Boyd, S., Ghosh, A., Prabhakar, B., & Shah, D. (2006). Randomized Gossip Algorithms. *IEEE Trans. Information Theory*. (Cited per `00_synthesis.md` lineage list.)
- Kratzke, T. M., Stone, L. D., & Frost, J. R. (2010). Search and Rescue Optimal Planning System (SAROPS). (Cited per `00_synthesis.md` lineage list.)

---

*Document length: ~3,200 words (excluding references and decision-tree pseudocode).*
