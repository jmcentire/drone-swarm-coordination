# /// script
# dependencies = ["numpy<3"]
# ###
"""Witness-alarm byzantine detection.

Each drone with a working GPS fix observes neighbors within range R via
visual fiducial / UWB time-of-flight / camera-based pose (with noise σ_obs).
For each observed neighbor B, drone A computes:

  expected_B = A_gps_position + relative_observation
  alarm = |expected_B − broadcast_B| > k × σ_total

where σ_total² = σ_GPS_A² + σ_GPS_B² + σ_obs². If multiple independent
witnesses raise alarms about the same drone, byzantine consensus is
declared and that drone is excluded from ASSIGN.

This catches byzantines via *physical inconsistency* between observed
reality and broadcast claim — a much stronger signal than statistical
anomaly in the broadcast distribution alone (which is what MAD/σ
detection uses).

The test sweeps:
  - k byzantine drones (1, 5, 10, 20)
  - attack model (random direction, coordinated direction)
  - witness count threshold (1, 3, 5)

For each configuration, reports:
  - Detection rate (fraction of byzantines correctly flagged)
  - False positive rate (fraction of honest drones falsely flagged)
  - Cascade after exclusion (% of honest drones with different leaf vs
    clean baseline, after byzantine drones are removed from ASSIGN)
"""

import os
import numpy as np

NUM_DRONES = 100
N_SEEDS = int(os.environ.get("N_SEEDS", "20"))
WORLD = 40.0
GPS_NOISE = float(os.environ.get("GPS_NOISE", "0.5"))   # σ in meters
OBS_NOISE = float(os.environ.get("OBS_NOISE", "0.1"))   # σ relative-position
WITNESS_RANGE = float(os.environ.get("WITNESS_RANGE", "30.0"))  # meters
ALARM_K = float(os.environ.get("ALARM_K", "5.0"))  # σ-multiple for alarm
WITNESS_THRESHOLD = int(os.environ.get("WITNESS_THRESHOLD", "3"))

# Heavy-tailed noise: 10% of GPS/observation samples drawn from a wider
# Gaussian (5× σ). Models real-sensor multipath, NLOS, and detection
# outliers — and stress-tests the witness-alarm false-positive rate.
HEAVY_TAIL_PROB = float(os.environ.get("HEAVY_TAIL_PROB", "0.1"))
HEAVY_TAIL_SCALE = float(os.environ.get("HEAVY_TAIL_SCALE", "5.0"))


def heavy_tailed_normal(rng, shape, sigma):
    base = rng.standard_normal(shape) * sigma
    flat = (shape if isinstance(shape, int) else shape[0])
    is_outlier = rng.random(flat) < HEAVY_TAIL_PROB
    if isinstance(shape, tuple) and len(shape) > 1:
        is_outlier = is_outlier.reshape(-1, *([1] * (len(shape) - 1)))
        is_outlier = np.broadcast_to(is_outlier, shape)
    spike = rng.standard_normal(shape) * sigma * HEAVY_TAIL_SCALE
    return np.where(is_outlier, spike, base)


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


def assign_all(positions, targets, mask=None):
    tree = ManifoldNode(targets)
    if mask is None:
        mask = np.ones(len(positions), dtype=bool)
    drones = [{'id': i, 'pos': positions[i].copy()}
              for i in range(len(positions)) if mask[i]]
    leaves = []
    for i in range(len(positions)):
        if not mask[i]:
            leaves.append(-1)
            continue
        leaf = compute_target(i, drones, tree)
        idx = int(np.argmin(np.linalg.norm(targets - leaf, axis=1)))
        leaves.append(idx)
    return np.array(leaves)


def witness_alarms(true_pos, broadcast_pos, gps_noise, obs_noise,
                    witness_range, alarm_k, rng):
    """Compute the alarm matrix: alarms[a][b] = True iff drone a alarms on
    drone b.

    Drone a:
      - Has its own GPS estimate of its true position: gps_a = true_pos[a] + N(0, σ_gps)
      - Observes neighbor b's relative position: r_ab = (true_pos[b] - true_pos[a]) + N(0, σ_obs)
      - Expected b position: expected_b = gps_a + r_ab
      - Alarm if |expected_b − broadcast_b| > alarm_k × σ_total

    Witness range limits which neighbors a drone can observe (physical
    visibility / UWB range).
    """
    n = len(true_pos)
    alarms = np.zeros((n, n), dtype=bool)
    sigma_total = np.sqrt(2 * gps_noise**2 + obs_noise**2)

    # Each drone's GPS estimate of its own position (heavy-tailed)
    gps_est = true_pos + heavy_tailed_normal(rng, (n, 3), gps_noise)

    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            # Witness range check: drone a can only observe drone b if within R
            true_dist = np.linalg.norm(true_pos[a] - true_pos[b])
            if true_dist > witness_range:
                continue
            # Drone a's noisy relative observation of b (heavy-tailed)
            relative = (true_pos[b] - true_pos[a]) + heavy_tailed_normal(rng, 3, obs_noise)
            expected_b = gps_est[a] + relative
            error = float(np.linalg.norm(expected_b - broadcast_pos[b]))
            if error > alarm_k * sigma_total:
                alarms[a][b] = True
    return alarms


def consensus_from_alarms(alarms, threshold):
    """Drone j is byzantine_consensus=True iff ≥threshold drones alarm on j."""
    n = alarms.shape[0]
    alarm_count = alarms.sum(axis=0)
    return alarm_count >= threshold


def run_attack_with_surplus(k, attack, witness_threshold, n_seeds, targets,
                              surplus=10):
    """Witness detection + patch protocol with surplus.

    Detected byzantines are treated as DEAD. Patch protocol fills their
    leaves from the surplus pool. The architecture handles this as a
    standard cluster-death (Lemma 7): exactly K reassignments per K
    detected byzantines, with 0 unfilled leaves when surplus ≥ K.
    """
    cascades = []
    n_total = NUM_DRONES + surplus
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        true_starts = rng.uniform(-WORLD, WORLD, (n_total, 3))
        byzantine_ids = (rng.choice(NUM_DRONES, size=k, replace=False)
                         if k > 0 else np.array([], dtype=int))
        clean_broadcast = true_starts + heavy_tailed_normal(rng, true_starts.shape, GPS_NOISE)
        baseline_assignment = assign_all(clean_broadcast[:NUM_DRONES],
                                          targets[:NUM_DRONES])

        adv_broadcast = clean_broadcast.copy()
        if attack == 'random':
            for bid in byzantine_ids:
                adv_broadcast[bid] = rng.uniform(-200, 200, 3)
        else:
            shift = rng.standard_normal(3)
            shift /= np.linalg.norm(shift)
            shift *= 100.0
            for bid in byzantine_ids:
                adv_broadcast[bid] = true_starts[bid] + shift

        # Witness alarm
        alarms = witness_alarms(true_starts, adv_broadcast, GPS_NOISE,
                                 OBS_NOISE, WITNESS_RANGE, ALARM_K, rng)
        is_byzantine_consensus = consensus_from_alarms(alarms, witness_threshold)

        # Patch: detected byzantines are DEAD; surplus fills their leaves.
        # Implementation: among the n_total drones, the NUM_DRONES primaries
        # all participate in baseline assignment. Detected byzantines mark
        # their slot as "dead" — the closest surplus drone takes over.
        # Final assignment is over (NUM_DRONES - k_det) honest primaries
        # + k_det promoted surplus = NUM_DRONES drones for NUM_DRONES leaves.

        n_det = int(is_byzantine_consensus.sum())
        # Build a corrected drone set: honest primaries + closest k_det surplus
        honest_mask = ~is_byzantine_consensus[:NUM_DRONES]
        honest_positions = adv_broadcast[:NUM_DRONES][honest_mask]
        # Pick k_det closest-to-need surplus drones from the surplus pool.
        # For simplicity, take the first k_det surplus drones.
        if surplus >= n_det:
            promoted_positions = clean_broadcast[NUM_DRONES:NUM_DRONES + n_det]
            combined = np.vstack([honest_positions, promoted_positions])
        else:
            # Insufficient surplus — leaves go unfilled
            promoted_positions = clean_broadcast[NUM_DRONES:NUM_DRONES + surplus]
            combined = np.vstack([honest_positions, promoted_positions])

        if len(combined) < NUM_DRONES:
            # Pad with NaN — leaves unfilled (graceful degradation)
            pad = np.full((NUM_DRONES - len(combined), 3), np.nan)
            combined = np.vstack([combined, pad])

        # Assign over the corrected set against the original target set
        valid_mask = ~np.any(np.isnan(combined), axis=1)
        adv_corrected = combined.copy()
        adv_corrected[~valid_mask] = 0  # placeholder
        new_assignment = assign_all(adv_corrected, targets[:NUM_DRONES],
                                     mask=valid_mask)

        # Compare to baseline only on drones that exist in both
        # (surplus promotions take a leaf that was originally another drone's;
        #  for the cascade metric, count drones whose leaf differs from baseline)
        # Map honest drones in the corrected set back to their original IDs
        honest_ids_orig = np.where(honest_mask)[0]
        cascade = 0
        for new_idx, orig_id in enumerate(honest_ids_orig):
            if new_assignment[new_idx] != baseline_assignment[orig_id]:
                cascade += 1
        cascades.append(cascade)

    n_honest = NUM_DRONES - k
    return {
        'cascade_mean': float(np.mean(cascades)),
        'cascade_pct': 100 * float(np.mean(cascades)) / max(n_honest, 1),
    }


def run_attack(k, attack, witness_threshold, n_seeds, targets):
    cascades_no_defense = []
    cascades_witness = []
    detected = []
    false_positives = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        true_starts = rng.uniform(-WORLD, WORLD, (NUM_DRONES, 3))
        byzantine_ids = (rng.choice(NUM_DRONES, size=k, replace=False)
                         if k > 0 else np.array([], dtype=int))
        byzantine_set = set(byzantine_ids.tolist())

        # Baseline: clean broadcast (broadcast = true position + GPS noise)
        clean_broadcast = true_starts + heavy_tailed_normal(rng, true_starts.shape, GPS_NOISE)
        baseline_assignment = assign_all(clean_broadcast, targets)

        # Adversarial: byzantines broadcast lies
        adv_broadcast = clean_broadcast.copy()
        if attack == 'random':
            for bid in byzantine_ids:
                adv_broadcast[bid] = rng.uniform(-200, 200, 3)
        elif attack == 'coordinated':
            shift = rng.standard_normal(3)
            shift /= np.linalg.norm(shift)
            shift *= 100.0
            for bid in byzantine_ids:
                adv_broadcast[bid] = true_starts[bid] + shift

        # No-defense cascade
        no_def_assignment = assign_all(adv_broadcast, targets)
        honest_ids = [i for i in range(NUM_DRONES) if i not in byzantine_set]
        no_def_changed = sum(1 for i in honest_ids
                             if no_def_assignment[i] != baseline_assignment[i])
        cascades_no_defense.append(no_def_changed)

        # Witness mechanism: every drone with working GPS observes neighbors
        # and alarms on inconsistencies
        alarms = witness_alarms(
            true_starts, adv_broadcast, GPS_NOISE, OBS_NOISE,
            WITNESS_RANGE, ALARM_K, rng)
        is_byzantine_consensus = consensus_from_alarms(alarms, witness_threshold)

        # Apply: exclude consensused-byzantine from ASSIGN
        mask = ~is_byzantine_consensus
        wit_assignment = assign_all(adv_broadcast, targets, mask=mask)

        wit_changed = sum(1 for i in honest_ids
                          if (mask[i] and wit_assignment[i] != baseline_assignment[i])
                          or (not mask[i]))
        cascades_witness.append(wit_changed)

        # Detection metrics
        correctly_flagged = sum(1 for bid in byzantine_ids if is_byzantine_consensus[bid])
        false_flags = sum(1 for hid in honest_ids if is_byzantine_consensus[hid])
        detected.append(correctly_flagged)
        false_positives.append(false_flags)

    n_honest = NUM_DRONES - k
    return {
        'no_defense_cascade': float(np.mean(cascades_no_defense)),
        'no_defense_pct': 100 * float(np.mean(cascades_no_defense)) / max(n_honest, 1),
        'witness_cascade': float(np.mean(cascades_witness)),
        'witness_pct': 100 * float(np.mean(cascades_witness)) / max(n_honest, 1),
        'detected_mean': float(np.mean(detected)),
        'fp_mean': float(np.mean(false_positives)),
        'detected_raw': np.array(detected),
        'fp_raw': np.array(false_positives),
        'witness_pct_raw': 100 * np.array(cascades_witness) / max(n_honest, 1),
        'no_defense_pct_raw': 100 * np.array(cascades_no_defense) / max(n_honest, 1),
    }


def bootstrap_ci(samples, n_resamples=2000, seed=0):
    samples = np.asarray(samples, dtype=float)
    if len(samples) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.array([
        samples[rng.integers(0, len(samples), len(samples))].mean()
        for _ in range(n_resamples)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    print(f"Witness-alarm benchmark: N={NUM_DRONES}, {N_SEEDS} seeds")
    print(f"GPS noise σ={GPS_NOISE}m, obs noise σ={OBS_NOISE}m, "
          f"witness range R={WITNESS_RANGE}m, alarm threshold k={ALARM_K}σ, "
          f"witness count = {WITNESS_THRESHOLD}\n")

    targets = make_sphere(NUM_DRONES)
    ks = [0, 1, 5, 10, 20]
    attacks = ['random', 'coordinated']

    for attack in attacks:
        print(f"=== {attack.upper()} BYZANTINE ATTACK ===")
        print(f"{'k':>4}  {'no_defense %':>22}  {'witness %':>22}  "
              f"{'witness+patch %':>17}  {'detected':>20}  {'fp':>20}")
        for k in ks:
            r = run_attack(k, attack, WITNESS_THRESHOLD, N_SEEDS, targets)
            r_patch = run_attack_with_surplus(k, attack, WITNESS_THRESHOLD,
                                                N_SEEDS, targets, surplus=10)
            nd_lo, nd_hi = bootstrap_ci(r['no_defense_pct_raw'])
            w_lo, w_hi = bootstrap_ci(r['witness_pct_raw'])
            d_lo, d_hi = bootstrap_ci(r['detected_raw'])
            f_lo, f_hi = bootstrap_ci(r['fp_raw'])
            print(f"{k:>4d}  "
                  f"{r['no_defense_pct']:>5.1f} [{nd_lo:>4.1f},{nd_hi:>4.1f}]  "
                  f"{r['witness_pct']:>5.1f} [{w_lo:>4.1f},{w_hi:>4.1f}]  "
                  f"{r_patch['cascade_pct']:>13.1f}%  "
                  f"{r['detected_mean']:>4.1f} [{d_lo:>4.1f},{d_hi:>4.1f}]  "
                  f"{r['fp_mean']:>4.1f} [{f_lo:>4.1f},{f_hi:>4.1f}]")
        print()

    # Subthreshold-lie sweep: how does detection degrade as the lie
    # magnitude approaches the alarm threshold?
    print("=== SUBTHRESHOLD ATTACK SWEEP (k=10 byzantines, coordinated) ===")
    print(f"Alarm threshold = {ALARM_K}σ × σ_total ≈ "
          f"{ALARM_K * np.sqrt(2*GPS_NOISE**2 + OBS_NOISE**2):.2f}m")
    print(f"{'lie magnitude':>16}  {'no_def':>10}  {'witness':>10}  "
          f"{'detected':>10}  {'fp':>6}")
    lie_magnitudes = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    for lie_mag in lie_magnitudes:
        cascades_no = []
        cascades_wit = []
        det = []
        fp = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            true_starts = rng.uniform(-WORLD, WORLD, (NUM_DRONES, 3))
            byzantine_ids = rng.choice(NUM_DRONES, size=10, replace=False)
            byzantine_set = set(byzantine_ids.tolist())
            clean_broadcast = true_starts + heavy_tailed_normal(rng, true_starts.shape, GPS_NOISE)
            baseline = assign_all(clean_broadcast, targets)
            adv_broadcast = clean_broadcast.copy()
            shift = rng.standard_normal(3)
            shift /= np.linalg.norm(shift)
            shift *= lie_mag
            for bid in byzantine_ids:
                adv_broadcast[bid] = true_starts[bid] + shift
            no_def_assign = assign_all(adv_broadcast, targets)
            honest_ids = [i for i in range(NUM_DRONES) if i not in byzantine_set]
            cascades_no.append(sum(1 for i in honest_ids
                                    if no_def_assign[i] != baseline[i]))
            alarms = witness_alarms(true_starts, adv_broadcast, GPS_NOISE,
                                     OBS_NOISE, WITNESS_RANGE, ALARM_K, rng)
            consensus = consensus_from_alarms(alarms, WITNESS_THRESHOLD)
            mask = ~consensus
            wit_assign = assign_all(adv_broadcast, targets, mask=mask)
            cascades_wit.append(sum(1 for i in honest_ids
                                     if (mask[i] and wit_assign[i] != baseline[i])
                                     or (not mask[i])))
            det.append(sum(1 for bid in byzantine_ids if consensus[bid]))
            fp.append(sum(1 for hid in honest_ids if consensus[hid]))
        d_lo, d_hi = bootstrap_ci(det)
        f_lo, f_hi = bootstrap_ci(fp)
        print(f"{lie_mag:>13.1f}m  "
              f"{np.mean(cascades_no):>9.1f}  "
              f"{np.mean(cascades_wit):>9.1f}  "
              f"{np.mean(det):>4.1f} [{d_lo:>3.1f},{d_hi:>3.1f}]/10  "
              f"{np.mean(fp):>4.1f} [{f_lo:>4.1f},{f_hi:>4.1f}]/90")
    print()


if __name__ == '__main__':
    main()
