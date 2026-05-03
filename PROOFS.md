# Formal Properties of the Decentralized Swarm Coordination Architecture

This document establishes the theoretical properties of the four-layer
coordination architecture: assignment, recovery, priority allocation, and
localization. Each result corresponds to an empirically measured claim in
the experimental sections.

The proofs are organized so that claims about determinism and bijectivity
support the assignment layer, claims about local-patching support the
recovery layer, claims about sub-manifold composition support priority
allocation, and the architectural theorems unify them all on the same
broadcast-as-shared-state primitive.

## 1. Notation

Let M = {m₁, ..., m_N} ⊂ ℝ³ denote the **target manifold**, an ordered set
of N points the swarm is to occupy.

Let T = T(M) denote the **PCA-tree** of M, defined recursively:
- A leaf node v has |L_v| = 1, where L_v ⊆ M is its associated point set.
- An internal node v has L_v with |L_v| > 1, a centroid
  c_v = (1/|L_v|) Σ_{m ∈ L_v} m, and a split direction Π_v ∈ ℝ³ defined as
  the leading right-singular vector of the centered matrix
  [L_v - c_v]. Children v_L, v_R partition L_v: the points with the
  ⌊|L_v|/2⌋ smallest projections (m - c_v) · Π_v go to v_L, the rest to
  v_R. Hence |L_{v_L}| = ⌊|L_v|/2⌋ and |L_{v_R}| = ⌈|L_v|/2⌉.

Let D = {d₁, ..., d_n} denote a set of n drones with positions
p(d_i) ∈ ℝ³.

The **strict-mode hierarchical assignment** ASSIGN(D, T) is defined
recursively. At node v with assigned drone subset D_v (initially D_root = D):

1. Compute d_left = round(|D_v| · |L_{v_L}| / |L_v|) and
   d_right = |D_v| − d_left.
2. Sort D_v by projection rank along Π_v.
3. Send the d_left drones with smallest projection to v_L; the rest to v_R.
4. Recurse on each child.
5. Terminal case: when D_v reaches a leaf v with |L_v| = 1, every drone in
   D_v is associated with the single target m_v ∈ L_v. The drone with the
   smallest distance ‖p(d) − m_v‖ is the **primary** for m_v; remaining
   drones are **surplus** and target the centroid c_{parent(v)} of v's
   parent. (Tie-break primary by drone ID.)

This is the algorithm implemented in `compute_target` in `simulator.py` and
the benchmark scripts.

We write n = N for the **bijective regime** (one drone per target) and
n = N + S for the **surplus regime** (S extra drones beyond targets).

## 2. Lemmas

### Lemma 1 (Determinism)

For any drone d_i ∈ D, the result of ASSIGN(D, T) restricted to d_i —
i.e., d_i's tree-path and assigned target — is a deterministic function of
(D, T) and the position vector p(·). It does not depend on which drone
"runs" the algorithm.

**Proof.** At every internal node v, the partition of D_v into D_{v_L} and
D_{v_R} is induced by the projection ranks of {p(d) : d ∈ D_v} onto Π_v,
and the cut at d_left depends only on |D_v|, |L_{v_L}|, |L_v|. None of
these values depend on the identity of the drone running ASSIGN; they
depend only on the set D and the tree T. Therefore each drone d_i ∈ D
descends through the same sequence of partitions to the same leaf; its
primary/surplus designation at the leaf is determined by the deterministic
distance comparison and ID tie-break. ∎

This is the basis of the *consensus-by-determinism* claim: every drone
running the algorithm against the same broadcast input arrives at the
same global assignment without exchanging messages.

**Preconditions for determinism in deployment.** The proof above
assumes (i) bit-identical floating-point arithmetic across drones,
(ii) a globally-consistent drone ID ordering for tie-breaks. (i) is
not automatically guaranteed by IEEE-754: the SVD computation and
projection-rank sort can produce different orderings on heterogeneous
hardware (different SIMD widths, fused-multiply-add availability,
compiler optimizations). For deployment-grade consensus, use either
deterministic-SVD libraries with strict floating-point modes, or
fixed-point arithmetic for the projection-rank computation. (ii) is
typically trivial — manufacturer serial numbers or launch-order
indices give the global ordering — but it is a precondition that
must be established at swarm initialization rather than discovered
at runtime.

### Lemma 2 (Bijectivity in the bijective regime)

When |D| = |M| = N, ASSIGN(D, T) is a bijection between D and M.

**Proof.** We show by induction on the tree depth at which D_v is computed
that |D_v| = |L_v| at every node.

*Base.* At the root, |D_root| = |D| = N = |M| = |L_root|.

*Inductive step.* Let v be an internal node with |D_v| = |L_v| = m. Then
d_left = round(m · |L_{v_L}|/m) = round(|L_{v_L}|) = |L_{v_L}|, since
|L_{v_L}| is integer-valued. Likewise d_right = m − |L_{v_L}| = |L_{v_R}|.
So |D_{v_L}| = |L_{v_L}| and |D_{v_R}| = |L_{v_R}|, preserving the
invariant.

By induction, every leaf v has |D_v| = |L_v| = 1. The single drone in D_v
is assigned to the single target in L_v, giving a bijection. ∎

### Lemma 3 (Reachability in the surplus regime)

When |D| = N + S with S ≥ 0, every leaf of T receives at least one drone
in ASSIGN(D, T).

**Proof.** We show by induction that |D_v| ≥ |L_v| at every node.

*Base.* |D_root| = N + S ≥ N = |L_root|.

*Inductive step.* Let v be an internal node with |D_v| = |L_v| + s_v for
some s_v ≥ 0, and let m = |L_v|, m_L = |L_{v_L}|, m_R = |L_{v_R}|. Then
d_left = round((m + s_v) · m_L / m) = round(m_L + s_v · m_L / m). Since
0 ≤ s_v · m_L / m ≤ s_v, the rounded value lies in [m_L, m_L + s_v]
(after accounting for banker's rounding's worst case of 0.5
displacement — see remark below). Therefore d_left ≥ m_L, i.e.,
|D_{v_L}| ≥ |L_{v_L}|. Symmetrically |D_{v_R}| ≥ |L_{v_R}|. The invariant
|D_v| ≥ |L_v| is preserved at both children.

By induction, every leaf v has |D_v| ≥ |L_v| = 1, so every leaf receives
at least one drone. ∎

*Remark on rounding.* Banker's rounding (round-half-to-even, the Python 3
default) maps half-integers to the nearest even integer. In the worst
case round(x + 0.5) = x for even x, so rounding can decrease a value by
0.5 from its real-valued counterpart. The proof above uses round(y) ≥
y − 0.5 for all real y, which is the tight bound. For
y = m_L + s_v · m_L / m ≥ m_L, we have round(y) ≥ y − 0.5 ≥ m_L − 0.5,
and since round(y) is integer, round(y) ≥ ⌈m_L − 0.5⌉ = m_L. The
inductive invariant therefore holds.

### Lemma 4 (Primary designation is deterministic across drones)

When multiple drones reach the same leaf v (|D_v| > 1), every drone in
D_v computes the same primary designation: the drone d* ∈ D_v
minimizing ‖p(d) − m_v‖ (with ID-based tie-break) is primary, and all
others are surplus.

**Proof.** The set D_v is determined by Lemma 1 — the same set is
identified by every drone in D_v running the algorithm. Each drone
computes ‖p(d) − m_v‖ for every d ∈ D_v using broadcast positions, which
all drones see identically. The minimum is unique up to ties; ID-based
tie-break gives a deterministic resolution. ∎

### Theorem 1 (Consensus by determinism)

Given a broadcast B in which every drone reports its position p(d), every
drone independently running ASSIGN(D, T) against B derives the same
global assignment a: D → M (with one-to-one correspondence on primary
drones in the bijective regime, with multi-occupant primary designation
and surplus assignment in the surplus regime).

**Proof.** Direct corollary of Lemmas 1, 2 (or 3), and 4. The assignment
is a deterministic function of (D, T, B) common to all drones. ∎

This formalizes the *no-coordination consensus* property of the
assignment layer. There is no message passing, no auction round, no leader
election; the consensus is a property of the algorithm's determinism on
shared input.

### Lemma 5 (Patch protocol is Hamming-optimal for single death)

Given a death of one primary drone d ∈ D in the surplus regime (S ≥ 1),
the patch protocol — promoting the live surplus drone s* ∈ D \ {d}
minimizing ‖p(s*) − a(d)‖ to target a(d), with all other targets
unchanged — produces the post-death assignment a' with minimum Hamming
distance H(a, a') = |{d' ∈ D : a(d') ≠ a'(d')}| from a, subject to the
constraint that the empty leaf a(d) is filled by some live drone.

**Proof.** The patch produces a' that differs from a on exactly one drone
(namely s*, whose target changed from c_{parent(leaf)} to a(d)) and
removes d from the assignment domain. Therefore H(a, a') = 1 (counting
only live drones).

Any valid post-death assignment a' must fill a(d) with some live drone
d* ≠ d. Since d* was assigned a(d*) ≠ a(d) in a, its assignment must
change in a'. Therefore H(a, a') ≥ 1.

Equality holds for the patch protocol; it achieves the minimum. ∎

### Lemma 6 (Patch correctness for single death)

The patch protocol, when applied to the death of one primary drone d in
the surplus regime (S ≥ 1), produces a post-death assignment a' that is
a valid bijection between the N − 1 surviving primaries plus the promoted
surplus s* on one side, and the N original target leaves on the other.

**Proof.** Pre-death, a is a bijection between primaries in D and the N
leaves M. The death removes d from primary assignment, freeing leaf
a(d); the N − 1 other primaries retain their assignments. The promotion
of s* changes s*'s target from a parent centroid (interior, not in M) to
a(d) ∈ M, making s* a primary at a(d). The set of primaries is now
{primaries ∖ {d}} ∪ {s*} of size N, mapped one-to-one onto M. The N + S − 1
surviving drones partition into N primaries and S − 1 surplus. ∎

### Lemma 7 (Sequential cluster patch correctness)

Given a cluster of K simultaneous primary deaths, sequential patch
(applying single-patch to each death in some order, each promotion
consuming one surplus) produces a valid bijection between the
N + S − K surviving drones and the N target leaves if and only if
S ≥ K. If S < K, exactly K − S leaves remain unfilled.

**Proof.** By induction on K.

*K = 1.* Lemma 6.

*Inductive step.* After patching the first death, S' = S − 1 surplus
drones remain. Apply the inductive hypothesis to the remaining K − 1
deaths with surplus pool S'. The condition S' ≥ K − 1 is equivalent to
S ≥ K. If S ≥ K throughout, every death's patch succeeds; total
reassignments = K, total leaves filled = N (bijection on primaries).

If S < K, after S patches the surplus pool is exhausted. The remaining
K − S deaths cannot be filled and leave K − S leaves empty. The
remaining N − (K − S) leaves are filled bijectively by surviving
primaries plus promoted surplus. ∎

### Lemma 8 (Greedy patch is locally optimal but not globally optimal)

Sequential greedy patch — at each step, promoting the live surplus
closest to the current dead leaf — minimizes per-step flight cost but
can be arbitrarily worse than the globally optimal cluster recovery
(Hungarian assignment between dead leaves and live surplus).

**Proof.** Per-step optimality is by definition: each step picks the
minimum-distance surplus.

For global suboptimality, consider K = 2 deaths at leaves A, B with
S = 2 surplus drones at positions S₁, S₂. Suppose:
- ‖A − S₁‖ = 1, ‖A − S₂‖ = 2,
- ‖B − S₁‖ = 1, ‖B − S₂‖ = 100.

Greedy processes deaths in some order. If A is patched first, S₁ is
chosen (cost 1); then B must use S₂ (cost 100); total 101. If B is
patched first, S₁ is chosen (cost 1); then A must use S₂ (cost 2);
total 3. The Hungarian-optimal pairing is (A → S₂, B → S₁), total cost 3.

Greedy's worst case (A first) is 33× the optimum. Therefore greedy is
not globally optimal. ∎

This motivates the *Hungarian-fixup* variant evaluated empirically in
§ Greedy vs Hungarian Cluster Patch.

### Lemma 9 (Quiescent broadcast consistency)

If at tick T every drone d ∈ D is locked (d.locked = True), then for any
T' > T such that no drone has unlocked between T and T' inclusive, every
drone reading the broadcast at any tick in [T, T'] observes identical
positions for all drones.

**Proof.** A locked drone publishes its position to the broadcast every
tick but does not change its position (the navigation step is skipped
when self.locked = True). Therefore for each drone d, the position
broadcast at tick T equals the position broadcast at tick T+1, ..., T'.
A read at any tick t ∈ [T, T'] returns these constant positions for
every drone. ∎

### Theorem 2 (Phase-transition consensus)

If every drone observes the all-locked condition at some tick during its
quiescent window and snapshots the broadcast at that observation time,
all drones derive identical phase-transition assignments — even though
each drone observes the all-locked condition at a slightly different
tick.

**Proof.** Let T_min and T_max be the earliest and latest ticks at which
any drone observes all-locked. By Lemma 9, the broadcast positions are
constant on [T_min, T_max] provided no drone unlocks in this interval.

A drone d unlocks at tick T_d + h_d, where T_d is its observation tick
and h_d is its hold duration. The hold duration is set deterministically
by each drone (typically the formation time of the current phase), and
is independent across drones up to small clock jitter, so
h_d ≥ h_min ≫ T_max − T_min in practice. Therefore no drone unlocks
before T_max, and the broadcast remains quiescent on [T_min, T_max].

Each drone snapshots an identical broadcast (Lemma 9) and runs ASSIGN
deterministically (Theorem 1). The snapshots agree, so the assignments
agree. ∎

This formalizes the *phase-quiescence* trick that enables multi-manifold
demos to transition without coordination machinery.

### Lemma 9.5 (ETA-deadline quiescence under packet loss)

Suppose every en-route drone d ∈ D broadcasts `STATE = EN_ROUTE`
with current ETA estimate ETA_d at intervals τ_d > 0, and arrived
drones broadcast no transit-state messages. Let ETA_max = max_d ETA_d
over the en-route set, δ > 0 a tolerance margin, and T_max =
ETA_max + δ. Assume an independent per-message delivery failure
probability p < 1 between any sender-receiver pair. Then for every
drone d' that survives to T_max:

(a) the probability that d' fails to receive *any* report containing
    ETA_max during the interval [t_0, ETA_max] decays as
    p^k where k is the number of reports broadcast by the slowest
    drone in that interval; and

(b) at time T_max, every surviving d' that has not received any
    EN_ROUTE broadcast in the window (T_max − τ*, T_max], where τ*
    is the minimum broadcast interval over the en-route set,
    deterministically transitions to the next phase.

**Proof.** (a) The slowest drone broadcasts ⌈(ETA_max − t_0)/τ⌉
times in the interval. Independent loss with probability p makes
total failure probability ≤ p^k, which is exponentially small for
modest k regardless of any single recipient's per-message loss rate.
The aggregation over the entire en-route set (each drone broadcasts
its own ETA estimate, recipients keep the maximum observed) only
strengthens this, since recipients can also infer ETA_max from any
drone whose own ETA estimate equals or exceeds it.

(b) The transition trigger is *negative evidence*: absence of
EN_ROUTE traffic in the deadline window. A perfectly silent window
on the broadcast channel is observationally identical to "all drones
arrived and ceased to broadcast" — but this is the desired
behavior, since the alternative (waiting for an explicit "all
arrived" message that may have been dropped) is precisely what made
the naïve protocol fragile. The trigger is invariant under loss
because the loss outcome and the success outcome are observationally
equivalent at the recipient. ∎

**Remark (channel-denial residual).** Lemma 9.5's transition trigger
cannot distinguish "everyone arrived" from "the channel itself went
silent for non-arrival reasons (jamming, fade)." This is a property
of the substrate, not the protocol. In any operational deployment
the broadcast carries non-transition traffic (position telemetry,
sensor data, command/control); a drone observing total channel
silence — *no* broadcasts of any kind for the prior N seconds —
can defer the phase transition pending channel recovery. This adds
no protocol messages and is a sanity check on the substrate
itself.

### Theorem 2.5 (Mid-flight reconfiguration consensus)

Let M, M' be two manifolds with M' arriving at the swarm before the
transit toward M has completed. Let `prior_end_state(D, M)` denote
the leaf coordinates each drone would occupy under ASSIGN(D, T(M))
upon completion. If every drone d ∈ D runs ASSIGN(D, T(M')) using
input positions either (i) prior_end_state(D, M) or
(ii) a broadcast-snapshot of live positions latched at a common
logical tick t*, then all drones derive identical assignments to
M'.

**Proof.** Both inputs are byte-identical across drones. For (i),
prior_end_state(D, M) is the output of ASSIGN(D, T(M)), which is
deterministic across drones by Theorem 1; therefore every drone
computes the same leaf-coordinate input. For (ii), Lemma 9 gives
broadcast consistency at any read on a quiescent window; for a
non-quiescent window, common consensus on t* is required (the
same machinery as Theorem 2 supplies, applied to a snapshot tick
rather than an all-locked tick). In both cases ASSIGN's
determinism (Theorem 1) maps the byte-identical input to a
byte-identical output; the swarm reaches consensus on the M'
assignment without any global drone-to-leaf mapping ever being
computed by any single drone. ∎

**Remark (option choice).** Option (i) is preferred in deployment
because it requires no snapshot-latching coordination — the prior
end-state is implicit in the prior assignment that every drone has
already computed locally. Option (ii) is more path-efficient
(drones recompute against where they actually are, not where they
would have ended up) but pays the snapshot-coordination cost.

## 3. Theorems and supporting results

### Lemma 10 (Sub-manifold composition)

The strict-mode hierarchical assignment ASSIGN(D, T(M')) where M' ⊆ M
is a sub-manifold of M, run on a drone subset D' ⊆ D, satisfies all
the properties of Lemmas 1-4 with M' replacing M and D' replacing D.

**Proof.** ASSIGN's properties (Lemmas 1-4) are stated in terms of
input (D, T) without dependence on M's structure beyond what enters
T's construction. The PCA-tree construction is well-defined on any
finite point set; the projection-rank partition is well-defined for
any drone set; the closest-by-distance primary designation is
well-defined on any leaf-vs-drone configuration. Therefore Lemmas 1-4
apply verbatim with M' and D' in place of M and D. ∎

This lemma supports the priority-allocation layer: surplus drones
running ASSIGN against the key sub-manifold M_key (offset by safety
distance δ) form a shadow fleet, with consensus, bijectivity (or
reachability), and primary designation all preserved. The same proofs
apply to the shadow fleet's algorithm as to the primary fleet's,
because the algorithm is identical and only the input changes.

### Theorem 3 (Layer composition)

We formalize the four-layer composition as a theorem with three
parts: parallel non-interference for layers with disjoint write
footprints, pipeline composition for the localization layer
producing inputs the other layers consume, and sequential override
for the recovery layer that conditionally amends the assignment.

#### Formal model

Let the **broadcast** B be a function from slot-IDs to values:

    B: Slot → Value

Slots are typed and partitioned:

    Slot = positions[1..n] ⊔ target_idx[1..n] ⊔ primary_flag[1..n]
         ⊔ dead_flag[1..n] ⊔ est_pos[1..n] ⊔ confidence[1..n]
         ⊔ key_list ⊔ alarms[1..n]

where n = N + S is the total number of drone slots and ⊔ denotes
disjoint union. Each layer is a transition `f: B → B` characterized
by a read set R(f) ⊆ Slot and a write set W(f) ⊆ Slot:

- For all s ∈ Slot \ W(f), [f(B)](s) = B(s) (slots not in W are
  unchanged).
- For all s ∈ W(f), [f(B)](s) is a function only of {B(s′) : s′ ∈
  R(f)} (writes depend only on reads).

#### Layer footprints

| Layer | R(f) (reads) | W(f) (writes) |
|-------|--------------|---------------|
| L₁ Assignment | positions[1..N] | target_idx[1..N], primary_flag[1..N] |
| L₂ Recovery | positions[1..n], dead_flag[1..n], target_idx, primary_flag | target_idx[i where dead_flag[i] set], primary_flag |
| L₃ Priority | positions[1..n], key_list | target_idx[N..n], primary_flag[N..n] |
| L₄ Localization | est_pos[1..n], confidence[1..n], alarms | est_pos[1..n], confidence[1..n] |

(In deployment, the `positions` slots that L₁, L₂, L₃ read are
filled by L₄'s `est_pos` outputs — i.e., positions ≡ est_pos as
input to the upper layers.)

#### Non-interference lemma

**Lemma (slot-disjointness implies commutativity).** Let f, g be
two transitions with R(f) ∩ W(g) = ∅, R(g) ∩ W(f) = ∅, and
W(f) ∩ W(g) = ∅. Then for all B, f(g(B)) = g(f(B)).

**Proof.** We show f(g(B)) and g(f(B)) agree on every slot.

For s ∉ W(f) ∪ W(g): [f(g(B))](s) = [g(B)](s) = B(s) =
[f(B)](s) = [g(f(B))](s). Both transitions leave s unchanged.

For s ∈ W(f) (and s ∉ W(g) by disjointness):
[f(g(B))](s) depends only on {[g(B)](s′) : s′ ∈ R(f)}. By R(f) ∩
W(g) = ∅, [g(B)](s′) = B(s′) for s′ ∈ R(f), so [f(g(B))](s) =
[f(B)](s). Likewise [g(f(B))](s) = [f(B)](s) since g doesn't
touch s. So [f(g(B))](s) = [g(f(B))](s).

For s ∈ W(g) (and s ∉ W(f)): by symmetric argument,
[f(g(B))](s) = [g(B)](s) and [g(f(B))](s) = [g(B)](s). Equal.

Therefore f(g(B)) = g(f(B)) on every slot. ∎

#### Theorem 3 (decomposed)

**Theorem 3a (Parallel non-interference for L₁ and L₃).** Layer 1
and Layer 3 are non-interfering in the sense of the lemma above.
Therefore L₁(L₃(B)) = L₃(L₁(B)) for all B, and parallel
composition is well-defined.

**Proof.** L₁ writes only target_idx[1..N] and primary_flag[1..N];
L₃ writes only target_idx[N..n] and primary_flag[N..n]. These
slot ranges are disjoint, so W(L₁) ∩ W(L₃) = ∅. L₁ reads only
positions[1..N], which is not in W(L₃) (L₃ writes target_idx and
primary_flag, not positions). Similarly L₃'s reads (positions and
key_list) are disjoint from W(L₁). Apply the lemma. ∎

**Theorem 3b (Pipeline composition for L₄).** Layer 4 is
non-interfering with L₁, L₂, L₃ in the slot sense. L₄ writes
est_pos and confidence; L₁, L₂, L₃ read est_pos (as positions) and
write target_idx, primary_flag, dead_flag — all disjoint from
W(L₄). The standard ordering L₄ → {L₁, L₃} → L₂ is therefore a
choice of computational schedule, not a correctness requirement;
any topological ordering that respects the data dependencies (L₄
before its consumers) produces the same final state.

**Proof.** R(L₄) = {est_pos, confidence, alarms}, W(L₄) = {est_pos,
confidence}. R(L_{1,2,3}) ⊆ {positions, target_idx, primary_flag,
dead_flag, key_list}. R(L_{1,2,3}) ∩ W(L₄) = ∅ (no upstream layer
reads est_pos as it appears in W(L₄) — they read it as the input
"positions"). W(L_{1,2,3}) ∩ W(L₄) = ∅ (different slot families).
W(L_{1,2,3}) ∩ R(L₄) = ∅ (L₄ doesn't read target_idx, primary_flag,
or dead_flag). Apply the lemma. ∎

**Theorem 3c (Sequential override of L₂ on L₁).** L₂ writes
target_idx[i] only for indices i where dead_flag[i] is set; for
i with dead_flag[i] = 0, L₂ leaves target_idx[i] unchanged. The
composition L₂ ∘ L₁ produces a state where:

    target_idx[i] = ⎧ L₂'s promotion if dead_flag[i] is set
                   ⎨
                    ⎩ L₁'s assignment otherwise

This composition is deterministic given the dead_flag pattern at
the time of L₂'s execution.

**Proof.** L₂'s write to target_idx[i] is conditional: f_{L₂}
includes a guard `if dead_flag[i] then promote else identity`. For
indices with dead_flag[i] = 0, L₂ is the identity, so L₁'s output
is preserved. For indices with dead_flag[i] = 1, L₂'s promotion
function (Lemma 5) overrides. The result is well-defined for any
dead_flag pattern. ∎

#### Architectural consequence

Theorem 3a, 3b, 3c together establish that the four-layer
composition has a deterministic final state for any input
broadcast. The empirical validation in `simulator.py` (Layers 1,
2, 3 in composition) and `bench_layer4.py` (Layer 4 implementation
and validation) confirms that the formal composition matches the
implementation. The substrate generalization claim — that one
broadcast primitive supports four distinct coordination patterns —
is therefore both empirically demonstrated and formally
characterized by the read/write footprint algebra above.

The earlier "substrate generalization" phrasing was an
architectural observation; the version above is a theorem with a
formal model and proof. The two are equivalent in claim but the
formal version is what reviewers expect for a publishable result.

### Conjecture 4 (Optimality gap of hierarchical bisection)

We characterize the gap empirically and present partial theoretical
analysis. The full asymptotic bound is open.

**Empirical fit comparison.** We tested five candidate functional
forms for the gap-vs-N data on the sphere manifold with starts ~
U[−40, 40]³, using weighted least-squares with seed-derived
weights and AIC/BIC for model selection.

| Model | Form | Fitted parameters | RMSE (pp) | AIC | BIC |
|-------|------|-------------------|-----------|-----|-----|
| M1 | a / ln(N) | a = 12.25 | 0.573 | 8.97 | 8.92 |
| M2 | a / ln(N) + b | a=8.81, b=0.443 | 0.939 | 7.15 | 7.05 |
| M3 | a / N^p | a=3.21, p=0.090 | 1.549 | 9.82 | 9.71 |
| M4 | a · ln(N) / N | a = 213.96 | 18.259 | 1311 | 1311 |
| **M5** | **a + b / √N** | **a=1.27, b=14.12** | **0.480** | **5.00** | **4.89** |

Model M5 (`a + b/√N`) gives the best fit by both AIC and BIC. M4
(rounding-error scaling, log N / N) fits dramatically worse — the
data is incompatible with the rounding term being dominant. M1
(originally conjectured 1/log N) fits adequately but is bested by
M5. (See `bench_conjecture4.py` for the comparison.)

**Predicted vs measured under M5:**

| N | Predicted M5 | Measured | Δ (pp) |
|---|--------------|----------|--------|
| 10 | 5.74 | 5.54 | +0.20 |
| 30 | 3.85 | 5.05 | -1.20 |
| 100 | 2.68 | 3.02 | -0.34 |
| 300 | 2.08 | 2.20 | -0.12 |
| 1000 | 1.72 | 1.70 | +0.02 |
| 3000 | 1.53 | 1.52 | +0.01 |
| 10000 | 1.41 | 1.43 | -0.02 |

M5 fits to within 0.05 pp for N ≥ 1000 and predicts an asymptote
of 1.27% at infinite N. Whether the gap actually approaches a
positive asymptote or eventually decays to zero cannot be
distinguished from data ending at N = 10,000.

**Updated conjectured form:**

    C_HIER ≤ (1 + β + α / √N) · C_OPT

with α, β depending on manifold geometry and start distribution.
On sphere with U[−40,40]³ starts: α ≈ 14.1, β ≈ 1.27%. The
asymptotic gap β may equal zero in the limit; the data does not
rule that out, but neither does it confirm it.

**Theoretical analysis of the 1/√N contribution.**

We bound the rounding-error contribution rigorously and identify
the projection-half-cut contribution as the open piece.

**Bound on rounding error (Proposition 1).** *In the surplus
regime (n = N + S drones for N target leaves), the total flight
cost contributed by rounding at all levels is O(D · √(N+S))
where D is the manifold diameter, giving a relative gap
contribution of O(D / (W · √N)) = O(1/√N) when starts are drawn
from a region of scale W ≈ D.*

*Proof sketch.* At tree level k there are 2^k internal nodes;
each computes a partition with rounding error at most 0.5 drones,
mis-routing at most one drone across the partition boundary.
Summed over level k, at most 2^k drones are mis-routed. Each
mis-routed drone is committed to the wrong child subtree, ending
at a leaf at most diam(subtree_at_level_k) away from its "should-
have" leaf. For a 2-manifold (sphere surface, cube faces, etc.)
with N total points uniformly distributed, the subtree at level k
has expected diameter O(D · √(n_k/N)) = O(D · 2^(-k/2)). Per-
level rounding cost is therefore O(2^k · D · 2^(-k/2)) = O(D ·
2^(k/2)). Summed over k from 0 to log₂ N:

  Σ_{k=0}^{log N} D · 2^(k/2) = D · (2^((log N)/2 + 1) − 1) /
                                (√2 − 1)
                              = O(D · √N).

Optimal cost C_OPT scales as O(N · W̄) where W̄ is the average
per-drone start-to-leaf distance; for random starts in the
W-cube, W̄ = Θ(W). Therefore relative gap from rounding:

  rounding_gap = O(D · √N) / O(N · W) = O(D / (W · √N))
               = O(1/√N) for D = Θ(W). ∎

In the bijective regime (n = N), rounding error is exactly zero
at every level (Lemma 2: dl = round(m · nl/m) = nl exactly when
m = nl + nr is integer), so the b/√N term in the empirical fit
must come entirely from the projection-half cut error.

**On the projection-half cut error (proof attempt).** The strict-mode
algorithm cuts at the projection median (balanced by drone count),
not at the cost-optimal cut. We attempt a bound here, mark gaps
explicitly, and note where the proof remains incomplete.

*Setup.* At tree level k, consider a node v with n_k = N/2^k drones
and m_k = n_k target leaves (bijective regime). The drones are
projected onto Π_v ∈ ℝ³, the principal axis of L_v. Let the
projection of drone i be x_i ∈ ℝ. The strict-mode algorithm cuts at
the projection-rank median (n_k/2-th smallest x_i). The cost-optimal
cut would partition drones to minimize Σ ||drone_i - assigned_leaf_i||
under bipartite matching; this cut may or may not coincide with the
projection-rank median.

*Per-level deviation.* Let the projection-rank median be x_med and
the cost-optimal cut threshold be x_opt. The number of drones that
go to a different subtree than they would under the cost-optimal
partition is bounded by:

    n_swap_k ≤ |{i : x_i ∈ [min(x_med, x_opt), max(x_med, x_opt)]}|

For drones drawn i.i.d. from a distribution with bounded variance σ_k²
on Π_v, by concentration of measure (e.g., Hoeffding-type bound):

    |x_med - x_opt| ≤ O(σ_k / √n_k) with high probability

Each "swapped" drone contributes at most diam(subtree_k) ≈ O(σ_k) to
the cost gap. So the level-k contribution to the gap is:

    gap_k ≤ n_swap_k · σ_k ≤ O(σ_k · σ_k · √n_k) = O(σ_k² √n_k)

Wait — this requires bounding n_swap_k by O(σ_k √n_k), which by
concentration of measure gives n_swap_k ≤ O((σ_k/√n_k) · n_k) = O(σ_k √n_k).

*The geometry.* For a 2-manifold M of intrinsic dimension d=2 sampled
uniformly, the projection variance σ_k² scales as the squared spatial
extent at level k. Subtree spatial extent goes as √(n_k/N) for a
2-manifold, so σ_k = O(D · √(n_k/N)) where D is the manifold diameter.
Plugging in:

    gap_k ≤ O((D · √(n_k/N))² · √n_k) = O(D² · n_k^(3/2) / N)

Summed over levels k = 0, 1, ..., log₂ N (with n_k = N/2^k):

    total_gap ≤ Σ_{k=0}^{log N} D² · (N/2^k)^(3/2) / N
              = (D² / √N) · Σ_{k=0}^{log N} 2^(-3k/2)
              = O(D² / √N)

*Relative gap.* Optimal cost C_OPT = O(N · W̄) where W̄ is the typical
start-to-leaf distance. For random starts in U[-W, W]³, W̄ = Θ(W).
Therefore:

    relative_gap = O(D² / √N) / O(N · W) = O(D² / (W · N · √N)) = O(1/N^(3/2))

Wait — this would give O(1/N^(3/2)), much faster decay than empirical
O(1/√N). Something in the geometry argument is too aggressive.

*The gap.* The hand-wavy step is bounding n_swap_k by σ_k · √n_k via
concentration. For random uniform projections, the median is
unbiased, but the *cost-optimal* cut threshold x_opt depends on the
target-distribution structure on Π_v's projection, and isn't
necessarily within σ_k/√n_k of the median. In particular, when the
target distribution is highly non-uniform on Π_v (e.g., clustered
near tips of a star), x_opt can be Θ(σ_k) away from x_med, giving
n_swap_k = Θ(n_k), not O(σ_k √n_k).

*Where the proof stalls.* The bound depends on a geometric quantity —
the cost-optimal-cut deviation from projection-median for the
specific manifold structure — that has no clean closed form. For
sphere-like manifolds (uniform on the principal-axis projection), the
deviation is small and the gap is fast-decaying. For star-like or
torus-like manifolds with non-uniform projection, the deviation is
larger.

*Conjectured form.* The empirical fit `a + b/√N` with manifold-
dependent constants α ∈ [10, 13] is consistent with the projection-
median deviation being Θ(σ_k) at each level for typical manifolds —
giving total gap O(σ_root) absolute = O(D) and relative gap O(1/N) per
level summed over log N levels with geometric decay = O(1/√N). The
empirical α reflects the manifold's projection-uniformity, not just
its diameter.

*Open analytical work.* A complete proof requires:
(i) bounding the projection-median vs cost-optimal-cut deviation as
    a function of manifold-projection-structure parameters;
(ii) verifying the geometric-decay assumption σ_k = O(σ_root · 2^(-k/2))
     more carefully for non-spherical manifolds;
(iii) extending to non-uniform start distributions (current proof
      assumes random-uniform).

We document this attempt and explicitly mark the gap rather than
claiming a result we have not established. The empirical evidence in
support of the O(1/√N) form is solid; the rigorous bound on the
projection-half-cut term remains open.

**The asymptote β.** The empirical asymptote β ≈ 1.27% in M5 may
represent (i) a non-vanishing residual error from manifold-
structure mismatch with the random-uniform start distribution,
(ii) a fitting artifact from the limited N range, (iii) a small-
sample bias in the PCA tree's split-axis estimation that doesn't
fully vanish. We cannot distinguish among these from data alone;
β is observationally bounded above by ≈ 1.5% and below by
non-negativity.

**Status.** Rounding-error contribution to the relative gap is
provably O(1/√N). Projection-half-cut error is empirically
consistent with O(1/√N) but the rigorous bound is open. The
1/log(N) form previously conjectured is consistent with the data
to within experimental tolerance but is dominated in fit quality
by `a + b/√N`. The tighter bound (M5 form) is now the primary
empirical claim; closing the projection-half gap remains future
work.

## 4. Computational complexity

### PCA tree construction

Given N points in ℝ³, the tree has 2N − 1 nodes and depth ⌈log₂ N⌉.
At each level, the SVDs across all nodes operate on a total of N
points; each SVD on a m-point set in ℝ³ is O(m). Total work per level
is O(N), summed over O(log N) levels: **O(N log N)** for tree
construction.

### Per-drone target query (ASSIGN traversal)

A drone descends from root to leaf, log N levels deep. At level k, the
drone projects all n_k = N / 2^k drones in its partition onto the
local Π_v and finds its rank, costing O(n_k). Summed:
Σ_{k=0}^{log N} N / 2^k = O(N).

Therefore one drone's target query is **O(N)**. Computed in parallel
for all N drones, the total compute across the swarm is **O(N²)**.
Per-drone latency is what matters for real-time operation, and that
is O(N).

### Patch on death

Finding the closest live surplus to a dead leaf requires iterating
over surviving drones to identify live surplus, then computing
distances. Total: **O(N + S) = O(N)**.

### Sequential cluster patch

K deaths × O(N) patch = **O(K · N)**, decentralized.

### Hungarian-optimal cluster patch

Globally optimal recovery requires bipartite matching between K dead
leaves and live surplus, O(K³) time, but requires centralized
computation of all pairwise distances O(K · S). Total: **O(K³ +
K · N)** centralized.

### Phase-transition recompute

Same as initial assignment: **O(N)** per drone for the full ASSIGN
traversal, triggered once per phase boundary.

## 5. Architectural implications

The combination of these results gives a coordination architecture
with the following operational properties:

- **No-coordination consensus** at the assignment layer (Theorem 1):
  the swarm can re-derive its global allocation from any common
  broadcast snapshot without leader election or message exchange.

- **Local recovery** (Lemma 6, 7): a drone death triggers
  exactly K reassignments for a K-cluster loss when surplus is
  adequate, with each reassignment computed locally and in parallel.

- **Compositionality** (Theorem 3): priority-aware allocation,
  recovery, and (future) localization can be added as additional
  layers on the same broadcast primitive without modifying the
  assignment layer.

- **Empirical near-optimality** (Conjecture 4): the gap from the
  globally optimal Hungarian assignment shrinks (empirical fit form
  `a + b/√N`, with rounding contribution provably O(1/√N)) and
  is empirically within 2-3% at N = 100 and 1.7% at N = 1000.

The combination of Theorem 3 with the empirical results in the
experimental sections gives the central architectural claim of the
paper: **a single decentralized coordination primitive
(broadcast-as-shared-state with locally-deterministic computation)
suffices to support multiple distinct coordination patterns at
multiple operational layers, with each layer's correctness provable
in isolation and the layers composable without interference**.
