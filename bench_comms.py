# /// script
# dependencies = ["numpy<3"]
# ///
"""Comms-layer empirical validation.

Three sweeps validate the operational claims added to WRITEUP §9.1
(quiescence detection, mid-flight reconfiguration, channel-denial
deferral) and the formal results in PROOFS Lemma 9.5 / Theorem 2.5.

Sweep A — Quiescence detection under packet loss
  Compares two protocols:
    NAIVE       — drones broadcast ARRIVED on arrival; each drone
                  latches when it has heard ARRIVED from every drone.
                  Optional periodic re-broadcast.
    INVERTED    — en-route drones broadcast EN_ROUTE + ETA on a τ
                  schedule that shrinks as the en-route count drops;
                  each drone latches at T_max = max(observed ETA) + δ
                  if no EN_ROUTE arrived in (T_max − τ*, T_max].
  Sweeps per-message loss p over {0, 0.1, 0.3, 0.5, 0.7, 0.9} for
  iid loss, plus a Gilbert-Elliott bursty channel.

  Metrics:
    false_quiescence  — fraction of (drone, seed) pairs that
                        transitioned before the slowest drone arrived
    missed_deadline   — fraction that never transitioned within the
                        observation horizon (deadlock)
    consensus_spread  — max - min transition tick across the swarm
                        (smaller = tighter consensus)
    bandwidth         — total broadcasts/sec across the swarm

Sweep B — Mid-flight reconfiguration
  Compares Option 1 (prior-end-state input) vs Option 2 (live
  broadcast snapshot) for consensus and path efficiency, sweeping
  the mid-transit reconfig moment f ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
  and snapshot jitter (Option 2's risk).

  Metrics:
    consensus_rate    — fraction of seeds where all N drones derive
                        a byte-identical assignment to M'
    path_overhead_pct — extra distance traveled vs the ideal direct
                        path (drone original start → final M' leaf)

Sweep C — Channel-denial deferral
  Injects a jamming window of width W during the deadline. Drones
  observing total channel silence (no broadcasts of any kind) for
  > S seconds defer the transition. Measures false-quiescence with
  and without the deferral check, plus the false-defer rate when
  the channel is naturally sparse.
"""

import os
import sys
import numpy as np

NUM_DRONES = int(os.environ.get("NUM_DRONES", "100"))
N_SEEDS = int(os.environ.get("N_SEEDS", "30"))
# Sweep B is O(N²) per drone for Option 2; keep it at a smaller seed count
# even when the other sweeps run with 200+ seeds. Option 1 is a real per-drone
# byte-equality test (100% expected) and Option 2's jitter cliff is a
# qualitative finding (50-77%) that doesn't need tighter CIs.
N_SEEDS_B = int(os.environ.get("N_SEEDS_B", "30"))
TICK_DT = 0.04           # seconds per tick (25 Hz)

# Phase model: each drone has an ETA drawn from N(μ, σ²), clipped to [μ-3σ, μ+3σ].
PHASE_MU = 20.0
PHASE_SIGMA = 5.0
HORIZON = 40.0           # observation horizon (seconds) — past this, "missed deadline"

# Inverted protocol parameters
TAU_FAR = 5.0            # broadcast interval when far from deadline
TAU_MID = 1.0            # broadcast interval mid-transit
TAU_NEAR = 0.2           # broadcast interval close to deadline
DEADLINE_DELTA = 2.0     # tolerance margin past max observed ETA

# Naive protocol: optionally re-broadcast ARRIVED every REBROADCAST_INTERVAL
REBROADCAST_INTERVAL = float(os.environ.get("REBROADCAST_INTERVAL", "0"))  # 0 = no re-broadcast


def bootstrap_ci(samples, n_resamples=1000, seed=0):
    samples = np.asarray(samples, dtype=float)
    if len(samples) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.array([
        samples[rng.integers(0, len(samples), len(samples))].mean()
        for _ in range(n_resamples)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# -----------------------------------------------------------------------------
# Channel models
# -----------------------------------------------------------------------------

class IIDChannel:
    """Each (sender, receiver, message) is dropped iid with probability p."""
    def __init__(self, p, rng):
        self.p = p
        self.rng = rng
    def deliver(self, n_recipients, sender=None):
        # Returns boolean mask: True = received
        return self.rng.random(n_recipients) >= self.p


class GilbertElliott:
    """Two-state Markov channel (single global state).
      Good state: per-message loss = p_good
      Bad state: per-message loss = p_bad
      Transitions per tick: Good→Bad = q_GB, Bad→Good = q_BG.
    """
    def __init__(self, p_good, p_bad, q_gb, q_bg, rng):
        self.p_good = p_good; self.p_bad = p_bad
        self.q_gb = q_gb; self.q_bg = q_bg
        self.rng = rng
        self.bad = False
    def step(self):
        if self.bad:
            if self.rng.random() < self.q_bg:
                self.bad = False
        else:
            if self.rng.random() < self.q_gb:
                self.bad = True
    def deliver(self, n_recipients, sender=None):
        p = self.p_bad if self.bad else self.p_good
        return self.rng.random(n_recipients) >= p


class ShadowedClusterChannel:
    """Spatially-correlated loss: K drones in a designated cluster experience
    high-loss links to/from anyone (shadowed by terrain, occluded antenna,
    etc.); links between drones outside the cluster are low-loss.

    Loss model (per-message): if either sender or recipient is in the cluster,
    drop with probability p_inside; otherwise drop with probability p_outside.
    Within-cluster traffic is also high-loss (everyone in the shadow shares
    the degraded RF environment).
    """
    def __init__(self, cluster_ids, p_inside, p_outside, rng):
        self.cluster_set = set(int(x) for x in cluster_ids)
        self.p_inside = p_inside
        self.p_outside = p_outside
        self.rng = rng
    def deliver(self, n_recipients, sender=None):
        # Per-recipient loss probability depends on whether sender or recipient
        # is in the shadowed cluster.
        sender_shadowed = (sender is not None and sender in self.cluster_set)
        ps = np.array([
            self.p_inside if (sender_shadowed or i in self.cluster_set)
            else self.p_outside
            for i in range(n_recipients)
        ])
        return self.rng.random(n_recipients) >= ps


# -----------------------------------------------------------------------------
# Sweep A: quiescence detection
# -----------------------------------------------------------------------------

def tau_for_remaining_count(en_route_count, n_total):
    """Broadcast interval in seconds based on remaining en-route fraction."""
    if en_route_count == 0:
        return TAU_NEAR
    frac = en_route_count / n_total
    if frac > 0.5:
        return TAU_FAR
    if frac > 0.1 or en_route_count > 1:
        return TAU_MID
    return TAU_NEAR


def simulate_naive(arrival_ticks, channel, horizon_ticks, rebroadcast_ticks=0):
    """Naive protocol: broadcast ARRIVED on arrival; transition when all heard."""
    n = len(arrival_ticks)
    # received_from[i][j] = drone i has heard ARRIVED from drone j
    received_from = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(received_from, True)
    transition_tick = np.full(n, -1, dtype=int)
    arrived = np.zeros(n, dtype=bool)
    last_rebroadcast = np.full(n, -1, dtype=int)
    n_messages = 0

    for t in range(horizon_ticks):
        # Newly arrived this tick
        new_arrived = (arrival_ticks == t)
        for j in range(n):
            if new_arrived[j]:
                arrived[j] = True
                last_rebroadcast[j] = t
                # Broadcast ARRIVED from j to all
                delivery = channel.deliver(n, sender=j)
                for i in range(n):
                    if i != j and delivery[i]:
                        received_from[i][j] = True
                n_messages += 1
        # Periodic re-broadcast
        if rebroadcast_ticks > 0:
            for j in range(n):
                if arrived[j] and (t - last_rebroadcast[j]) >= rebroadcast_ticks:
                    last_rebroadcast[j] = t
                    delivery = channel.deliver(n, sender=j)
                    for i in range(n):
                        if i != j and delivery[i]:
                            received_from[i][j] = True
                    n_messages += 1
        # Step channel state (for bursty)
        if hasattr(channel, 'step'):
            channel.step()
        # Each drone checks: have I heard from everyone? (and have I arrived
        # myself — required so a drone doesn't latch its own self-loop early)
        all_heard = received_from.all(axis=1)
        for i in range(n):
            if transition_tick[i] == -1 and all_heard[i] and arrived[i]:
                transition_tick[i] = t

    return transition_tick, n_messages


def simulate_inverted(arrival_ticks, channel, horizon_ticks,
                       byzantine_ids=None, byzantine_eta=None,
                       max_plausible_eta_ticks=None):
    """Inverted protocol: en-route drones broadcast EN_ROUTE + ETA on a
    schedule that shrinks as en-route count drops. Each drone latches
    at T_max + δ if no EN_ROUTE arrived in (T_max − τ*, T_max].

    Optional adversarial parameters:
      byzantine_ids:       set of drone ids broadcasting falsified ETAs.
      byzantine_eta:       ETA value (in ticks) the byzantines broadcast,
                           e.g. horizon_ticks * 100 for a stall attack.
      max_plausible_eta_ticks: if set, recipients reject any ETA exceeding
                           current_tick + this value as obviously implausible.
                           Implements the §9.1.5 sanity-bound mitigation.
    """
    n = len(arrival_ticks)
    byz = set(byzantine_ids) if byzantine_ids is not None else set()
    # Each drone's view of max observed ETA (in ticks)
    max_eta_observed = np.zeros(n, dtype=int)
    # Each drone's last EN_ROUTE message receipt tick
    last_en_route_tick = np.full(n, -1, dtype=int)
    # Each drone's broadcast schedule: next tick at which they broadcast
    next_broadcast = np.zeros(n, dtype=int)
    transition_tick = np.full(n, -1, dtype=int)
    n_messages = 0
    delta_ticks = int(DEADLINE_DELTA / TICK_DT)

    for t in range(horizon_ticks):
        # Byzantine drones broadcast EN_ROUTE forever, claiming a falsified
        # ETA. Honest drones broadcast EN_ROUTE only while en-route to their
        # actual leaf; once arrived, they go silent.
        en_route_mask = (arrival_ticks > t)
        for bid in byz:
            en_route_mask[bid] = True  # byzantines never "arrive"
        en_route_count = int(en_route_mask.sum())
        # Each en-route drone broadcasts EN_ROUTE + ETA on its schedule
        for j in range(n):
            if en_route_mask[j] and t >= next_broadcast[j]:
                tau = tau_for_remaining_count(en_route_count, n)
                next_broadcast[j] = t + max(1, int(tau / TICK_DT))
                # ETA carried in this broadcast
                eta_j = byzantine_eta if (j in byz and byzantine_eta is not None) else arrival_ticks[j]
                delivery = channel.deliver(n, sender=j)
                for i in range(n):
                    if i != j and delivery[i]:
                        # Mitigation: recipients reject implausibly far ETAs.
                        if max_plausible_eta_ticks is not None and eta_j > t + max_plausible_eta_ticks:
                            continue
                        if eta_j > max_eta_observed[i]:
                            max_eta_observed[i] = eta_j
                        last_en_route_tick[i] = t
                n_messages += 1
        if hasattr(channel, 'step'):
            channel.step()
        # Each drone (including arrived ones) checks the deadline
        # Arrived drones use their own arrival as a lower bound on max_eta
        for i in range(n):
            if transition_tick[i] != -1:
                continue
            if not en_route_mask[i]:
                # Self-arrival is also evidence of an ETA
                if arrival_ticks[i] > max_eta_observed[i]:
                    max_eta_observed[i] = arrival_ticks[i]
            deadline = max_eta_observed[i] + delta_ticks
            if t >= deadline:
                # Was an EN_ROUTE message received in (deadline - tau*, deadline]?
                window_start = deadline - max(1, int(TAU_NEAR / TICK_DT))
                if last_en_route_tick[i] < window_start:
                    transition_tick[i] = t

    return transition_tick, n_messages


def run_sweep_a():
    print("=" * 74)
    print("SWEEP A — Quiescence detection under packet loss")
    print("=" * 74)
    horizon_ticks = int(HORIZON / TICK_DT)
    rebroadcast_ticks = int(REBROADCAST_INTERVAL / TICK_DT) if REBROADCAST_INTERVAL > 0 else 0

    p_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"\n  N={NUM_DRONES} drones, μ_arrival={PHASE_MU}s ± {PHASE_SIGMA}s, "
          f"horizon={HORIZON}s, {N_SEEDS} seeds")
    print(f"  Naive re-broadcast interval: "
          f"{REBROADCAST_INTERVAL if REBROADCAST_INTERVAL > 0 else 'none'}\n")

    print("  IID channel:")
    print(f"  {'p':>5} {'protocol':>10} {'false_q %':>22} {'missed %':>22} "
          f"{'spread (s)':>22} {'msgs/s':>10}")
    for p in p_values:
        for protocol in ('naive', 'inverted'):
            false_q = []
            missed = []
            spread = []
            msgs_per_sec = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                arrivals = np.clip(
                    rng.normal(PHASE_MU, PHASE_SIGMA, NUM_DRONES),
                    PHASE_MU - 3*PHASE_SIGMA, PHASE_MU + 3*PHASE_SIGMA
                )
                arrival_ticks = (arrivals / TICK_DT).astype(int)
                ch = IIDChannel(p, rng)
                if protocol == 'naive':
                    tr, nmsg = simulate_naive(arrival_ticks, ch, horizon_ticks,
                                                rebroadcast_ticks)
                else:
                    tr, nmsg = simulate_inverted(arrival_ticks, ch, horizon_ticks)
                last_arrival = arrival_ticks.max()
                # False quiescence: drones that transitioned before the
                # slowest drone actually arrived
                fq = np.sum((tr != -1) & (tr < last_arrival)) / NUM_DRONES
                false_q.append(fq)
                # Missed deadline
                miss = np.sum(tr == -1) / NUM_DRONES
                missed.append(miss)
                # Consensus spread
                tr_valid = tr[tr != -1]
                if len(tr_valid) > 1:
                    spread.append((tr_valid.max() - tr_valid.min()) * TICK_DT)
                else:
                    spread.append(0.0)
                msgs_per_sec.append(nmsg / HORIZON)
            fq_lo, fq_hi = bootstrap_ci(false_q)
            m_lo, m_hi = bootstrap_ci(missed)
            s_lo, s_hi = bootstrap_ci(spread)
            print(f"  {p:>5.2f} {protocol:>10} "
                  f"{100*np.mean(false_q):>5.1f} [{100*fq_lo:>4.1f},{100*fq_hi:>4.1f}]  "
                  f"{100*np.mean(missed):>5.1f} [{100*m_lo:>4.1f},{100*m_hi:>4.1f}]  "
                  f"{np.mean(spread):>5.2f} [{s_lo:>4.2f},{s_hi:>4.2f}]  "
                  f"{np.mean(msgs_per_sec):>9.1f}")
        print()

    print("  Bursty (Gilbert-Elliott) channel: p_good=0.05, p_bad=0.95, "
          "q_GB=0.005/tick, q_BG=0.05/tick")
    print(f"  {'protocol':>10} {'false_q %':>22} {'missed %':>22} "
          f"{'spread (s)':>22} {'msgs/s':>10}")
    for protocol in ('naive', 'inverted'):
        false_q = []; missed = []; spread = []; msgs_per_sec = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 1000)
            arrivals = np.clip(
                rng.normal(PHASE_MU, PHASE_SIGMA, NUM_DRONES),
                PHASE_MU - 3*PHASE_SIGMA, PHASE_MU + 3*PHASE_SIGMA
            )
            arrival_ticks = (arrivals / TICK_DT).astype(int)
            ch = GilbertElliott(0.05, 0.95, 0.005, 0.05, rng)
            if protocol == 'naive':
                tr, nmsg = simulate_naive(arrival_ticks, ch, horizon_ticks,
                                            rebroadcast_ticks)
            else:
                tr, nmsg = simulate_inverted(arrival_ticks, ch, horizon_ticks)
            last_arrival = arrival_ticks.max()
            false_q.append(np.sum((tr != -1) & (tr < last_arrival)) / NUM_DRONES)
            missed.append(np.sum(tr == -1) / NUM_DRONES)
            tr_valid = tr[tr != -1]
            spread.append(((tr_valid.max() - tr_valid.min()) * TICK_DT)
                          if len(tr_valid) > 1 else 0.0)
            msgs_per_sec.append(nmsg / HORIZON)
        fq_lo, fq_hi = bootstrap_ci(false_q)
        m_lo, m_hi = bootstrap_ci(missed)
        s_lo, s_hi = bootstrap_ci(spread)
        print(f"  {protocol:>10} "
              f"{100*np.mean(false_q):>5.1f} [{100*fq_lo:>4.1f},{100*fq_hi:>4.1f}]  "
              f"{100*np.mean(missed):>5.1f} [{100*m_lo:>4.1f},{100*m_hi:>4.1f}]  "
              f"{np.mean(spread):>5.2f} [{s_lo:>4.2f},{s_hi:>4.2f}]  "
              f"{np.mean(msgs_per_sec):>9.1f}")
    print()

    # Spatially-correlated loss: K drones in a shadowed cluster experience
    # high loss to/from anyone; the rest experience low loss.
    print("  Shadowed-cluster channel: K drones in cluster at p_inside=0.9, "
          "rest at p_outside=0.1")
    print(f"  (Outside-only links would correspond to p=0.1 baseline above.)")
    print(f"  {'K':>4} {'protocol':>10} {'false_q %':>22} {'missed %':>22} "
          f"{'spread (s)':>22} {'msgs/s':>10}")
    Ks_cluster = [k for k in [5, 10, 20, 50] if k <= NUM_DRONES]
    for K in Ks_cluster:
        for protocol in ('naive', 'inverted'):
            false_q = []; missed = []; spread = []; msgs_per_sec = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed + 3000)
                arrivals = np.clip(
                    rng.normal(PHASE_MU, PHASE_SIGMA, NUM_DRONES),
                    PHASE_MU - 3*PHASE_SIGMA, PHASE_MU + 3*PHASE_SIGMA
                )
                arrival_ticks = (arrivals / TICK_DT).astype(int)
                # Choose K drones for the shadowed cluster (deterministic per seed)
                cluster_ids = rng.choice(NUM_DRONES, size=K, replace=False)
                ch = ShadowedClusterChannel(cluster_ids, p_inside=0.9,
                                              p_outside=0.1, rng=rng)
                if protocol == 'naive':
                    tr, nmsg = simulate_naive(arrival_ticks, ch, horizon_ticks,
                                                rebroadcast_ticks)
                else:
                    tr, nmsg = simulate_inverted(arrival_ticks, ch, horizon_ticks)
                last_arrival = arrival_ticks.max()
                false_q.append(np.sum((tr != -1) & (tr < last_arrival)) / NUM_DRONES)
                missed.append(np.sum(tr == -1) / NUM_DRONES)
                tr_valid = tr[tr != -1]
                spread.append(((tr_valid.max() - tr_valid.min()) * TICK_DT)
                              if len(tr_valid) > 1 else 0.0)
                msgs_per_sec.append(nmsg / HORIZON)
            fq_lo, fq_hi = bootstrap_ci(false_q)
            m_lo, m_hi = bootstrap_ci(missed)
            s_lo, s_hi = bootstrap_ci(spread)
            print(f"  {K:>4d} {protocol:>10} "
                  f"{100*np.mean(false_q):>5.1f} [{100*fq_lo:>4.1f},{100*fq_hi:>4.1f}]  "
                  f"{100*np.mean(missed):>5.1f} [{100*m_lo:>4.1f},{100*m_hi:>4.1f}]  "
                  f"{np.mean(spread):>5.2f} [{s_lo:>4.2f},{s_hi:>4.2f}]  "
                  f"{np.mean(msgs_per_sec):>9.1f}")
        print()


# -----------------------------------------------------------------------------
# Sweep B: mid-flight reconfiguration
# -----------------------------------------------------------------------------
# Reuse the ManifoldNode and assign machinery from bench_witness for
# consistency.

class ManifoldNode:
    def __init__(self, positions, depth=0):
        self.positions = np.array(positions)
        self.center = np.mean(positions, axis=0)
        self.depth = depth
        self.split_axis = None
        self.left = self.right = None
        if len(positions) > 1:
            self._split()
    def _split(self):
        c = self.positions - self.center
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        self.split_axis = Vt[0]
        proj = c @ self.split_axis
        order = np.argsort(proj, kind='stable')
        mid = len(order) // 2
        self.left = ManifoldNode(self.positions[order[:mid]], self.depth + 1)
        self.right = ManifoldNode(self.positions[order[mid:]], self.depth + 1)


def compute_target(my_id, drones, root):
    node = root
    cur = list(drones)
    my_pos = next(np.array(d['pos']) for d in drones if d['id'] == my_id)
    while node.left is not None and len(cur) > 1:
        n = len(cur)
        nl = len(node.left.positions)
        nt = len(node.positions)
        dl = max(0, min(n, int(round(n * nl / nt))))
        if dl == 0:
            node = node.right; continue
        if dl == n:
            node = node.left; continue
        positions = np.array([d['pos'] for d in cur])
        proj = positions @ node.split_axis
        order = np.argsort(proj, kind='stable')
        groups = [
            [cur[order[i]] for i in range(dl)],
            [cur[order[i]] for i in range(dl, n)],
        ]
        subs = [node.left, node.right]
        for i, group in enumerate(groups):
            if any(d['id'] == my_id for d in group):
                node = subs[i]; cur = group; break
    while node.left is not None:
        dl_ = np.linalg.norm(my_pos - node.left.center)
        dr_ = np.linalg.norm(my_pos - node.right.center)
        node = node.left if dl_ <= dr_ else node.right
    if len(node.positions) == 1:
        return node.positions[0]
    return node.center


def assign_all(positions, targets):
    tree = ManifoldNode(targets)
    drones = [{'id': i, 'pos': positions[i].copy()} for i in range(len(positions))]
    leaves = []
    for i in range(len(positions)):
        leaf = compute_target(i, drones, tree)
        idx = int(np.argmin(np.linalg.norm(targets - leaf, axis=1)))
        leaves.append(idx)
    return np.array(leaves)


def make_sphere(n, radius=15, center=(0, 0, 20)):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    cx, cy, cz = center
    return np.column_stack((
        radius * np.cos(theta) * np.sin(phi) + cx,
        radius * np.sin(theta) * np.sin(phi) + cy,
        radius * np.cos(phi) + cz,
    ))


def make_cube_shell(n, size=16, center=(0, 0, 20)):
    cx, cy, cz = center
    half = size / 2
    side = 4
    pts = []
    for face_axis, sign in [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]:
        for i in range(side):
            for j in range(side):
                u = -half + (i + 0.5) * size / side
                v = -half + (j + 0.5) * size / side
                p = [0, 0, 0]
                p[face_axis] = sign * half
                others = [k for k in range(3) if k != face_axis]
                p[others[0]] = u; p[others[1]] = v
                pts.append([p[0] + cx, p[1] + cy, p[2] + cz])
    return np.array(pts[:n])


def run_sweep_b():
    print("=" * 74)
    print("SWEEP B — Mid-flight reconfiguration: Option 1 vs Option 2")
    print("=" * 74)
    print(f"\n  N={NUM_DRONES} drones, M=sphere, M'=cube_shell, N_SEEDS_B={N_SEEDS_B}")
    print(f"  Option 2's clock_skew_ticks parameter: each drone's local snapshot is")
    print(f"  taken k_d ticks late, k_d ~ Uniform(0, clock_skew_ticks). 1 tick = {TICK_DT*1000:.0f}ms.")
    print(f"  When clock_skew_ticks > 0, drones see different mid-transit positions and")
    print(f"  may compute different M' assignments.\n")

    M = make_sphere(NUM_DRONES)
    Mp = make_cube_shell(NUM_DRONES)

    fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    # Clock-skew in ticks: 0 = perfect sync, 1 = 1 tick (40ms), 5 = 5 ticks (200ms).
    skews = [0.0, 1.0, 5.0]

    print(f"  {'option':>10} {'f':>5} {'skew(ticks)':>11} "
          f"{'consensus %':>14} {'path overhead %':>20}")
    for f in fractions:
        for option in (1, 2):
            for skew in (skews if option == 2 else [0.0]):
                consensus = []
                overhead = []
                for seed in range(N_SEEDS_B):
                    rng = np.random.default_rng(seed)
                    starts = rng.uniform(-40, 40, (NUM_DRONES, 3))
                    # Initial assignment to M
                    leaves_M = assign_all(starts, M)
                    targets_M = M[leaves_M]
                    # At fraction f, drones are at: starts + f * (targets_M - starts)
                    mid_pos = starts + f * (targets_M - starts)

                    if option == 1:
                        # All drones use prior_end_state = M[leaves_M].
                        # We test consensus the same way as Option 2: each
                        # drone independently runs assign_all on its own
                        # prior_end_state input, and we verify byte-equality
                        # of the resulting assignment vectors. With identical
                        # inputs and a deterministic ASSIGN this should be
                        # 100%; the test is non-trivial because it would
                        # detect any source of nondeterminism (e.g., dict
                        # iteration order, BLAS sign flips, fp drift).
                        per_drone_assignments = np.zeros((NUM_DRONES, NUM_DRONES), dtype=int)
                        for d in range(NUM_DRONES):
                            per_drone_assignments[d] = assign_all(targets_M, Mp)
                        first = per_drone_assignments[0]
                        all_agree = all((per_drone_assignments[i] == first).all()
                                        for i in range(1, NUM_DRONES))
                        consensus.append(1.0 if all_agree else 0.0)
                        leaves_Mp = first
                        # Path overhead: actual path = starts→mid_pos→Mp[leaves_Mp]
                        actual = (np.linalg.norm(mid_pos - starts, axis=1)
                                  + np.linalg.norm(Mp[leaves_Mp] - mid_pos, axis=1))
                    else:
                        # Option 2: each drone d latches a snapshot k_d ticks late,
                        # k_d ~ Uniform(0, skew). At time of snapshot, drone d sees
                        # mid-transit positions: starts + (f + k_d * tick_dt / T)*(targets_M - starts),
                        # where T = total transit time = PHASE_MU. With skew=0 every drone
                        # latches at the same tick → identical input → consensus. With
                        # skew>0 each drone sees a slightly different f, and since the
                        # PCA bisection is sensitive to point ordering, small input
                        # differences can flip leaf assignments at boundary drones.
                        per_drone_assignments = np.zeros((NUM_DRONES, NUM_DRONES), dtype=int)
                        for d in range(NUM_DRONES):
                            k_d = rng.uniform(0.0, skew)  # ticks late this drone latches
                            f_d = f + (k_d * TICK_DT) / max(1e-6, PHASE_MU)
                            view = starts + f_d * (targets_M - starts)
                            per_drone_assignments[d] = assign_all(view, Mp)
                        # Consensus: do all drones agree?
                        first = per_drone_assignments[0]
                        all_agree = all((per_drone_assignments[i] == first).all()
                                        for i in range(1, NUM_DRONES))
                        consensus.append(1.0 if all_agree else 0.0)
                        # Path overhead using each drone's own assignment
                        leaves_Mp_d = np.array([per_drone_assignments[d][d]
                                                 for d in range(NUM_DRONES)])
                        actual = (np.linalg.norm(mid_pos - starts, axis=1)
                                  + np.linalg.norm(Mp[leaves_Mp_d] - mid_pos, axis=1))

                    # Ideal: starts → final M' leaf directly (using a fresh assignment from starts)
                    leaves_ideal = assign_all(starts, Mp)
                    ideal = np.linalg.norm(Mp[leaves_ideal] - starts, axis=1)
                    overhead.append(100 * (actual.sum() - ideal.sum()) / ideal.sum())

                c_lo, c_hi = bootstrap_ci(consensus)
                o_lo, o_hi = bootstrap_ci(overhead)
                opt_label = f"Opt {option}"
                print(f"  {opt_label:>10} {f:>5.1f} {skew:>11.1f} "
                      f"{100*np.mean(consensus):>5.1f} [{100*c_lo:>5.1f},{100*c_hi:>5.1f}]  "
                      f"{np.mean(overhead):>6.1f} [{o_lo:>5.1f},{o_hi:>5.1f}]")
        print()


# -----------------------------------------------------------------------------
# Sweep C: channel-denial deferral
# -----------------------------------------------------------------------------

def simulate_inverted_with_jam(arrival_ticks, channel, horizon_ticks,
                                  jam_start_tick, jam_width_ticks,
                                  bg_traffic_rate_hz, deferral_silence_s,
                                  use_deferral):
    """Inverted protocol with a jam window plus background telemetry traffic.

    Background traffic: every drone broadcasts a position update every
    1/bg_traffic_rate_hz seconds (this is the *channel liveness* signal).
    During jam, all messages drop with prob 1.

    Deferral: if use_deferral, a drone defers transition if it has not
    received ANY broadcast (en-route OR background) in the last
    deferral_silence_s seconds.
    """
    n = len(arrival_ticks)
    max_eta_observed = np.zeros(n, dtype=int)
    last_en_route_tick = np.full(n, -1, dtype=int)
    last_any_tick = np.full(n, 0, dtype=int)
    next_broadcast = np.zeros(n, dtype=int)
    next_bg = np.zeros(n, dtype=int)
    transition_tick = np.full(n, -1, dtype=int)
    delta_ticks = int(DEADLINE_DELTA / TICK_DT)
    silence_ticks = int(deferral_silence_s / TICK_DT)
    bg_period = max(1, int(1.0 / bg_traffic_rate_hz / TICK_DT))

    for t in range(horizon_ticks):
        in_jam = jam_start_tick <= t < jam_start_tick + jam_width_ticks

        en_route_mask = (arrival_ticks > t)
        en_route_count = int(en_route_mask.sum())

        # EN_ROUTE broadcasts
        for j in range(n):
            if en_route_mask[j] and t >= next_broadcast[j]:
                tau = tau_for_remaining_count(en_route_count, n)
                next_broadcast[j] = t + max(1, int(tau / TICK_DT))
                if not in_jam:
                    delivery = channel.deliver(n, sender=j)
                    for i in range(n):
                        if i != j and delivery[i]:
                            if arrival_ticks[j] > max_eta_observed[i]:
                                max_eta_observed[i] = arrival_ticks[j]
                            last_en_route_tick[i] = t
                            last_any_tick[i] = t

        # Background traffic from every drone
        for j in range(n):
            if t >= next_bg[j]:
                next_bg[j] = t + bg_period
                if not in_jam:
                    delivery = channel.deliver(n, sender=j)
                    for i in range(n):
                        if i != j and delivery[i]:
                            last_any_tick[i] = t

        if hasattr(channel, 'step'):
            channel.step()

        for i in range(n):
            if transition_tick[i] != -1:
                continue
            if not en_route_mask[i]:
                if arrival_ticks[i] > max_eta_observed[i]:
                    max_eta_observed[i] = arrival_ticks[i]
            deadline = max_eta_observed[i] + delta_ticks
            if t >= deadline:
                window_start = deadline - max(1, int(TAU_NEAR / TICK_DT))
                if last_en_route_tick[i] < window_start:
                    if use_deferral and (t - last_any_tick[i]) > silence_ticks:
                        # Defer: channel appears denied, don't transition this tick
                        continue
                    transition_tick[i] = t

    return transition_tick


def run_sweep_c():
    print("=" * 74)
    print("SWEEP C — Channel-denial deferral")
    print("=" * 74)
    # Sweep C uses an extended horizon so we can distinguish
    # "deferred-and-recovered" (transitioned correctly after jam ended) from
    # "deferred-past-horizon" (genuine deadlock). The base bench horizon is
    # 40s; for jams up to 30s past the deadline we need 90s of headroom.
    horizon_seconds = 90.0
    horizon_ticks = int(horizon_seconds / TICK_DT)
    print(f"\n  N={NUM_DRONES}, μ_arrival={PHASE_MU}s ± {PHASE_SIGMA}s, "
          f"{N_SEEDS} seeds, horizon={horizon_seconds}s")
    print(f"  Background traffic: each drone @ 5 Hz position update.")
    print(f"  Deferral threshold: 1.0s of total channel silence.")
    print(f"  Base IID loss p=0.1; jam = 100% loss for window W centered on deadline.\n")

    bg_rate = 5.0
    silence_threshold = 1.0
    base_p = 0.1
    deadline_center_tick = int((PHASE_MU + 3*PHASE_SIGMA + DEADLINE_DELTA) / TICK_DT)

    # Columns:
    #   transitioned %    — fraction that did transition by horizon
    #   false_q %         — transitioned before slowest arrival
    #   deferred-recovered % — never transitioned, but jam ended ≥5s before
    #                         horizon (would have transitioned without deferral
    #                         constraint or if we'd waited longer)
    #   deadlocked %      — never transitioned and jam window also ended without
    #                         recovery (genuine failure mode)
    print(f"  {'jam W':>7} {'mode':>11} {'false_q %':>16} "
          f"{'transitioned %':>18} {'def-recovered %':>18} {'deadlocked %':>16}")

    for W in [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 80.0]:
        for use_def in (False, True):
            false_q = []
            transitioned = []
            def_recovered = []
            deadlocked = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed + 7000)
                arrivals = np.clip(
                    rng.normal(PHASE_MU, PHASE_SIGMA, NUM_DRONES),
                    PHASE_MU - 3*PHASE_SIGMA, PHASE_MU + 3*PHASE_SIGMA
                )
                arrival_ticks = (arrivals / TICK_DT).astype(int)
                ch = IIDChannel(base_p, rng)
                jam_start = max(0, deadline_center_tick - int(W / 2 / TICK_DT))
                jam_width = int(W / TICK_DT)
                jam_end = jam_start + jam_width
                tr = simulate_inverted_with_jam(
                    arrival_ticks, ch, horizon_ticks,
                    jam_start, jam_width, bg_rate, silence_threshold, use_def
                )
                last_arrival = arrival_ticks.max()
                false_q.append(np.sum((tr != -1) & (tr < last_arrival)) / NUM_DRONES)
                transitioned.append(np.sum(tr != -1) / NUM_DRONES)
                # A drone that never transitioned: if the jam ended ≥5s before
                # horizon, the channel had time to recover — so non-transition
                # means deferral over-held; otherwise call it deadlocked.
                untransitioned_mask = (tr == -1)
                untrans_count = np.sum(untransitioned_mask)
                if untrans_count > 0:
                    if (horizon_ticks - jam_end) > int(5.0 / TICK_DT):
                        def_recovered.append(untrans_count / NUM_DRONES)
                        deadlocked.append(0.0)
                    else:
                        def_recovered.append(0.0)
                        deadlocked.append(untrans_count / NUM_DRONES)
                else:
                    def_recovered.append(0.0)
                    deadlocked.append(0.0)
            mode = 'with-defer' if use_def else 'no-defer'
            fq_lo, fq_hi = bootstrap_ci(false_q)
            tr_lo, tr_hi = bootstrap_ci(transitioned)
            r_lo, r_hi = bootstrap_ci(def_recovered)
            d_lo, d_hi = bootstrap_ci(deadlocked)
            print(f"  {W:>5.1f}s {mode:>11} "
                  f"{100*np.mean(false_q):>4.1f} [{100*fq_lo:>3.1f},{100*fq_hi:>3.1f}]  "
                  f"{100*np.mean(transitioned):>6.1f} [{100*tr_lo:>4.1f},{100*tr_hi:>4.1f}]  "
                  f"{100*np.mean(def_recovered):>6.1f} [{100*r_lo:>4.1f},{100*r_hi:>4.1f}]  "
                  f"{100*np.mean(deadlocked):>6.1f} [{100*d_lo:>4.1f},{100*d_hi:>4.1f}]")
        print()


def run_sweep_d():
    """ETA-spoofing attack: K byzantine drones broadcast a falsified ETA
    far in the future. Without mitigation, every honest drone's
    max_eta_observed gets pushed past the horizon and the swarm deadlocks.
    With the §9.1.5 sanity-bound mitigation (recipients reject any ETA >
    current_tick + max_plausible_transit), byzantines are filtered.

    Note: this is the *single most damaging* known attack against the
    quiescence protocol. We measure deadlock rate (no transition by
    horizon) for K ∈ {0, 1, 5, 10, 25} on N=100 drones, with and without
    mitigation. The mitigation is purely local — no consensus required.
    """
    print("=" * 74)
    print("SWEEP D — ETA-spoofing attack (and per-drone sanity-bound defense)")
    print("=" * 74)
    horizon_ticks = int(HORIZON / TICK_DT)
    base_p = 0.1
    # Max plausible ETA: current_tick + 2 * (μ_arrival + 3σ). Generous bound;
    # any ETA more than 70s (= 2 * 35s) in the future is rejected.
    max_plausible_ticks = int(2 * (PHASE_MU + 3*PHASE_SIGMA) / TICK_DT)
    # Falsified ETA: 100× horizon (unambiguously out of range)
    spoof_eta = horizon_ticks * 100

    print(f"\n  N={NUM_DRONES} drones, μ_arrival={PHASE_MU}s ± {PHASE_SIGMA}s, "
          f"horizon={HORIZON}s, IID loss p={base_p}, {N_SEEDS} seeds")
    print(f"  Spoofed ETA = {spoof_eta} ticks (×100 horizon).")
    print(f"  Sanity-bound mitigation: reject ETA > t + {max_plausible_ticks} ticks "
          f"({max_plausible_ticks*TICK_DT:.1f}s).\n")

    print(f"  {'K byz':>6} {'mode':>14} {'deadlocked %':>22} "
          f"{'transitioned %':>22} {'false_q (honest) %':>22}")

    Ks = [k for k in [0, 1, 5, 10, 25] if k <= NUM_DRONES]
    for K in Ks:
        for mode in ('no-defense', 'sanity-bound'):
            deadlocked = []
            transitioned = []
            false_q_honest = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed + 9000)
                arrivals = np.clip(
                    rng.normal(PHASE_MU, PHASE_SIGMA, NUM_DRONES),
                    PHASE_MU - 3*PHASE_SIGMA, PHASE_MU + 3*PHASE_SIGMA
                )
                arrival_ticks = (arrivals / TICK_DT).astype(int)
                ch = IIDChannel(base_p, rng)
                if K > 0:
                    byz_ids = rng.choice(NUM_DRONES, size=K, replace=False)
                else:
                    byz_ids = np.array([], dtype=int)
                byz_set = set(int(x) for x in byz_ids)
                mp = max_plausible_ticks if mode == 'sanity-bound' else None
                tr, _ = simulate_inverted(
                    arrival_ticks, ch, horizon_ticks,
                    byzantine_ids=byz_ids, byzantine_eta=spoof_eta,
                    max_plausible_eta_ticks=mp,
                )
                # Look only at honest drones for the operational metrics
                honest_mask = np.array([i not in byz_set for i in range(NUM_DRONES)])
                honest_tr = tr[honest_mask]
                n_honest = honest_mask.sum()
                deadlocked.append(np.sum(honest_tr == -1) / max(1, n_honest))
                transitioned.append(np.sum(honest_tr != -1) / max(1, n_honest))
                # false_q over honest drones uses honest-only last_arrival
                honest_arrivals = arrival_ticks[honest_mask]
                last_honest_arrival = honest_arrivals.max() if n_honest > 0 else 0
                false_q_honest.append(
                    np.sum((honest_tr != -1) & (honest_tr < last_honest_arrival))
                    / max(1, n_honest)
                )
            d_lo, d_hi = bootstrap_ci(deadlocked)
            t_lo, t_hi = bootstrap_ci(transitioned)
            fq_lo, fq_hi = bootstrap_ci(false_q_honest)
            print(f"  {K:>6d} {mode:>14} "
                  f"{100*np.mean(deadlocked):>5.1f} [{100*d_lo:>4.1f},{100*d_hi:>4.1f}]  "
                  f"{100*np.mean(transitioned):>5.1f} [{100*t_lo:>4.1f},{100*t_hi:>4.1f}]  "
                  f"{100*np.mean(false_q_honest):>5.1f} [{100*fq_lo:>4.1f},{100*fq_hi:>4.1f}]")
        print()


def run_sweep_e():
    """Lemma 9.5(a) empirical fit: failure probability decays as p^k.

    Lemma 9.5(a) claims that for the slowest-arriving drone broadcasting k
    EN_ROUTE messages with per-recipient drop probability p, a single
    recipient fails to receive *any* of those k broadcasts with probability
    exactly p^k. We measure this directly: for each (p, k) cell, count the
    fraction of (recipient, seed) pairs where the recipient received zero
    broadcasts from the slowest drone, and compare to the theoretical p^k.

    k is varied by holding the slowest-drone arrival fixed and changing the
    broadcast schedule (TAU). With the standard schedule, k is determined
    by arrival time and the tau_for_remaining_count function; we instead
    use a fixed broadcast period τ_fixed and force k = ⌈arrival / τ_fixed⌉.
    """
    print("=" * 74)
    print("SWEEP E — Lemma 9.5(a) empirical fit (p^k decay)")
    print("=" * 74)
    print(f"\n  N={NUM_DRONES} drones, slowest drone arrives at fixed t=PHASE_MU+3σ={PHASE_MU+3*PHASE_SIGMA:.1f}s")
    print(f"  k = number of EN_ROUTE broadcasts from slowest drone before its arrival.")
    print(f"  Measured: P(recipient receives 0 of k broadcasts). Theoretical: p^k.\n")

    n = NUM_DRONES
    slowest_arrival_s = PHASE_MU + 3*PHASE_SIGMA
    slowest_arrival_ticks = int(slowest_arrival_s / TICK_DT)

    print(f"  {'p':>5} {'k':>5} {'measured P(0 received)':>25} "
          f"{'theoretical p^k':>18} {'log measured':>14} {'k·log(p)':>12}")

    ps = [0.3, 0.5, 0.7, 0.9]
    # k values: choose tau so that k = ⌈arrival/tau⌉ matches a target list
    target_ks = [1, 2, 5, 10, 20, 50]

    for p in ps:
        for k_target in target_ks:
            # tau (in ticks) such that exactly k_target broadcasts fit before arrival.
            # Each broadcast at t = 0, tau, 2*tau, ..., (k-1)*tau, all < arrival.
            # So we need k_target * tau_ticks ≤ arrival_ticks < (k_target+1) * tau_ticks
            # → tau_ticks = arrival_ticks / k_target (rounded down)
            tau_ticks = max(1, slowest_arrival_ticks // k_target)
            actual_k = slowest_arrival_ticks // tau_ticks
            zero_count = 0
            total = 0
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed + 11000)
                # Slowest drone broadcasts at ticks 0, tau, 2*tau, ..., (actual_k-1)*tau
                # Each broadcast: independent loss to each of (n-1) recipients.
                received_any = np.zeros(n - 1, dtype=bool)
                for b in range(actual_k):
                    delivery = rng.random(n - 1) >= p
                    received_any |= delivery
                zero_count += int(np.sum(~received_any))
                total += (n - 1)
            measured = zero_count / total if total > 0 else 0.0
            theoretical = p ** actual_k
            log_meas = np.log10(max(measured, 1e-12))
            k_log_p = actual_k * np.log10(p)
            print(f"  {p:>5.2f} {actual_k:>5d} {measured:>15.6e}        "
                  f"{theoretical:>15.6e}  {log_meas:>10.3f}  {k_log_p:>10.3f}")
        print()


def main():
    sweep = os.environ.get("SWEEP", "abcde")
    if "a" in sweep:
        run_sweep_a()
    if "b" in sweep:
        run_sweep_b()
    if "c" in sweep:
        run_sweep_c()
    if "d" in sweep:
        run_sweep_d()
    if "e" in sweep:
        run_sweep_e()


if __name__ == '__main__':
    main()
