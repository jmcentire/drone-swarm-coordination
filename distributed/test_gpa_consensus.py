# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Standalone test of the full consensus pipeline as specified:

  - Each drone has local-only sensing (its k in-comms-range neighbors).
  - Each drone computes a rough local map (ISOMAP/MDS on its own
    subgraph + measured ranges).
  - Each drone broadcasts its local map; gossip propagates so every
    drone ends up with the set of all drones' local maps.
  - Generalized Procrustes Analysis (GPA) aligns all maps to a common
    frame simultaneously: iteratively find rotation+translation per map
    that minimizes residuals against the running mean, update the mean,
    repeat.
  - Per-landmark robust aggregation: for each drone i, gather the
    aligned estimates of i from every contributor that observed i;
    reduce to a single point via geometric median (Weiszfeld's
    algorithm) or trimmed mean. Outlier contributors get dropped.
  - Compare consensus positions to the SECRET truth (held only by the
    test); report per-drone error and aggregate error.

Three scenarios:
  - clean: no faulty sensors
  - one-bad-sensor: one drone's range measurements are biased by +50m
  - multi-bad-sensor: several drones have correlated/uncorrelated biases

Falsifiability: if the pipeline works, the bad-sensor drone's
consensus position is within tolerance of its TRUE position regardless
of how badly the bad drone's OWN measurements deviated. The bad drone
contributes one outlier vote per shared landmark and gets trimmed.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Truth + observation generation
# ---------------------------------------------------------------------------

def make_swarm(n: int, seed: int, spread: float = 15.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-spread, spread, size=(n, 3))


def observer_neighborhood(
    truth: np.ndarray, observer: int, comms_range: float
) -> list[int]:
    """Returns the IDs of drones within comms_range of `observer`,
    excluding the observer itself."""
    n = len(truth)
    diffs = truth - truth[observer]
    dists = np.linalg.norm(diffs, axis=1)
    return [i for i in range(n) if i != observer and dists[i] <= comms_range]


def observer_local_edges(
    truth: np.ndarray,
    observer: int,
    comms_range: float,
    bias: float = 0.0,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int, float]]:
    """The set of range measurements `observer` can make. Returns edges
    (observer, neighbor, measured_range). If `bias` is non-zero, all of
    this observer's measurements are biased by that amount (modeling a
    bad-sensor drone). Noise_sigma adds zero-mean Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng(0)
    edges = []
    for j in range(len(truth)):
        if j == observer:
            continue
        d = float(np.linalg.norm(truth[observer] - truth[j]))
        if d > comms_range:
            continue
        m = d + bias
        if noise_sigma > 0:
            m += float(rng.normal(scale=noise_sigma))
        edges.append((observer, j, max(0.1, m)))
    return edges


# ---------------------------------------------------------------------------
# Per-drone local map (Stage 1 + Stage 2)
# ---------------------------------------------------------------------------

def isomap_embed(n_local: int, edges: list[tuple[int, int, float]]) -> np.ndarray:
    """Classical MDS via ISOMAP over a small local subgraph. n_local is
    the count of local-indexed drones; edges use local indices [0..n_local)."""
    INF = float("inf")
    D = np.full((n_local, n_local), INF)
    np.fill_diagonal(D, 0.0)
    for i, j, d in edges:
        if d < D[i, j]:
            D[i, j] = d
            D[j, i] = d
    # Floyd-Warshall
    for k in range(n_local):
        D = np.minimum(D, D[:, k:k+1] + D[k:k+1, :])
    if not np.all(np.isfinite(D)):
        return np.full((n_local, 3), np.nan)
    J = np.eye(n_local) - np.ones((n_local, n_local)) / n_local
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w = w[order]; V = V[:, order]
    scale = np.sqrt(np.maximum(w[:3], 0))
    return V[:, :3] * scale


def polish_embedding(
    initial: np.ndarray, edges: list[tuple[int, int, float]]
) -> np.ndarray:
    """LM polish: minimize range residuals starting from initial."""
    n = initial.shape[0]
    def residuals(x_flat: np.ndarray) -> np.ndarray:
        x = x_flat.reshape((n, 3))
        res = [float(x[0, 0]), float(x[0, 1]), float(x[0, 2])]  # pin drone-local 0
        for i, j, d in edges:
            res.append(float(np.linalg.norm(x[i] - x[j])) - d)
        return np.array(res, dtype=np.float64)
    result = least_squares(
        residuals, x0=initial.flatten(), max_nfev=500, loss="huber", f_scale=2.0
    )
    return result.x.reshape((n, 3))


def local_map_for_observer(
    truth: np.ndarray,
    observer: int,
    comms_range: float,
    bias_per_observer: dict[int, float] | None = None,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Produce one drone's local map: a dict from drone_id -> position
    in this observer's arbitrary local frame. Includes the observer
    itself and its neighbors. Frame is set by the embedding (no shared
    convention with other drones)."""
    if bias_per_observer is None:
        bias_per_observer = {}
    rng = np.random.default_rng(seed * 1009 + observer * 17)

    # Each drone in observer's neighborhood that also reports edges within
    # the local view: we use observer's OWN measurements (biased if bad).
    neighbors = observer_neighborhood(truth, observer, comms_range)
    local_drone_set = sorted(set([observer] + neighbors))
    if len(local_drone_set) < 4:
        return {}

    # Build local edges: observer's measurements to each of its neighbors,
    # plus measurements between neighbors that observer can OVERHEAR (we
    # model this as: any two drones both within observer's range get an
    # edge weighted by the truth + small overhear-noise, with no bias
    # unless the observer is one of the endpoints).
    edges = []
    obs_bias = bias_per_observer.get(observer, 0.0)
    for nb in neighbors:
        d = float(np.linalg.norm(truth[observer] - truth[nb]))
        m = d + obs_bias + (float(rng.normal(scale=noise_sigma)) if noise_sigma > 0 else 0.0)
        edges.append((observer, nb, max(0.1, m)))
    for i, a in enumerate(neighbors):
        for b in neighbors[i + 1:]:
            d = float(np.linalg.norm(truth[a] - truth[b]))
            if d > comms_range:
                continue
            m = d + (float(rng.normal(scale=noise_sigma)) if noise_sigma > 0 else 0.0)
            edges.append((a, b, max(0.1, m)))

    # Index drones into a local 0..K range for the embedding
    id_to_local = {gid: i for i, gid in enumerate(local_drone_set)}
    local_edges = [(id_to_local[i], id_to_local[j], d) for i, j, d in edges]

    init = isomap_embed(len(local_drone_set), local_edges)
    if np.isnan(init).any():
        return {}
    polished = polish_embedding(init, local_edges)
    return {gid: polished[id_to_local[gid]] for gid in local_drone_set}


# ---------------------------------------------------------------------------
# Generalized Procrustes Analysis (Stage 3 — alignment)
# ---------------------------------------------------------------------------

def procrustes_align(
    source: dict[int, np.ndarray], target: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """Align `source` to `target` frame using shared landmarks. Returns
    aligned source (same keys, but coordinates rotated+translated)."""
    common = sorted(set(source.keys()) & set(target.keys()))
    if len(common) < 3:
        return source  # not enough shared landmarks
    P = np.array([source[c] for c in common])
    T = np.array([target[c] for c in common])
    cP = P.mean(axis=0); cT = T.mean(axis=0)
    H = (P - cP).T @ (T - cT)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(float(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cT - R @ cP
    return {k: R @ source[k] + t for k in source}


def generalized_procrustes(
    local_maps: list[dict[int, np.ndarray]],
    max_iter: int = 30,
    tol: float = 1e-3,
) -> tuple[list[dict[int, np.ndarray]], dict[int, np.ndarray]]:
    """Iteratively align all local maps to a shared frame.

      1. Initialize reference = the local map with the most landmarks.
      2. For each map, align it to the reference using shared landmarks.
      3. Update reference = per-landmark mean of all aligned maps.
      4. Repeat until reference stabilizes.

    Returns (aligned_maps, reference)."""
    if not local_maps:
        return [], {}
    # Pick initial reference: map with most landmarks
    reference = dict(max(local_maps, key=lambda m: len(m)))
    prev_ref: dict[int, np.ndarray] = {}

    for it in range(max_iter):
        aligned = [procrustes_align(m, reference) for m in local_maps]
        # Update reference: per-landmark mean.
        all_ids: set[int] = set()
        for m in aligned:
            all_ids.update(m.keys())
        new_ref: dict[int, np.ndarray] = {}
        for nid in all_ids:
            ests = [m[nid] for m in aligned if nid in m]
            if ests:
                new_ref[nid] = np.mean(ests, axis=0)
        # Convergence check
        if prev_ref:
            common = set(new_ref.keys()) & set(reference.keys())
            if common:
                delta = max(
                    float(np.linalg.norm(new_ref[c] - reference[c])) for c in common
                )
                if delta < tol:
                    reference = new_ref
                    break
        prev_ref = reference
        reference = new_ref
    return aligned, reference


# ---------------------------------------------------------------------------
# Per-landmark robust aggregation (cluster -> single point)
# ---------------------------------------------------------------------------

def geometric_median(points: np.ndarray, max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
    """Weiszfeld's algorithm for the geometric (spatial) median —
    the L1-equivalent of mean, much more robust to outliers."""
    if len(points) == 0:
        return np.zeros(3)
    if len(points) == 1:
        return points[0]
    x = np.mean(points, axis=0)
    for _ in range(max_iter):
        diffs = points - x
        dists = np.linalg.norm(diffs, axis=1)
        nonzero = dists > 1e-9
        if not nonzero.any():
            break
        w = 1.0 / dists[nonzero]
        x_new = np.sum(points[nonzero] * w[:, None], axis=0) / np.sum(w)
        if float(np.linalg.norm(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    return x


def consensus_per_landmark(
    aligned_maps: list[dict[int, np.ndarray]], method: str = "geometric_median"
) -> dict[int, np.ndarray]:
    """For each landmark, gather all aligned estimates and reduce via
    geometric median (or mean, or trimmed mean)."""
    all_ids: set[int] = set()
    for m in aligned_maps:
        all_ids.update(m.keys())
    consensus = {}
    for nid in all_ids:
        ests = np.array([m[nid] for m in aligned_maps if nid in m])
        if method == "mean":
            consensus[nid] = ests.mean(axis=0)
        elif method == "trimmed":
            if len(ests) <= 2:
                consensus[nid] = ests.mean(axis=0)
            else:
                med = np.median(ests, axis=0)
                dists = np.linalg.norm(ests - med, axis=1)
                keep = dists < np.quantile(dists, 0.75)
                consensus[nid] = ests[keep].mean(axis=0)
        else:  # geometric_median
            consensus[nid] = geometric_median(ests)
    return consensus


# ---------------------------------------------------------------------------
# Comparison to truth
# ---------------------------------------------------------------------------

def compare_to_truth(
    consensus: dict[int, np.ndarray], truth: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Procrustes-align consensus to truth frame, return per-drone errors,
    mean, max."""
    ids = sorted(consensus.keys())
    C = np.array([consensus[i] for i in ids])
    T = truth[ids]
    cC = C.mean(axis=0); cT = T.mean(axis=0)
    H = (C - cC).T @ (T - cT)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(float(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    C_aligned = (C - cC) @ R.T + cT
    errors = np.linalg.norm(C_aligned - T, axis=1)
    return errors, float(np.mean(errors)), float(np.max(errors))


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

def run_one_scenario(
    n: int,
    seed: int,
    comms_range: float,
    bad_sensors: dict[int, float] | None = None,
    noise_sigma: float = 0.0,
    aggregation: str = "geometric_median",
) -> dict:
    truth = make_swarm(n=n, seed=seed)
    bad_sensors = bad_sensors or {}
    local_maps = []
    for obs in range(n):
        lm = local_map_for_observer(
            truth, obs, comms_range,
            bias_per_observer=bad_sensors,
            noise_sigma=noise_sigma,
            seed=seed,
        )
        if lm:
            local_maps.append(lm)
    aligned, _ = generalized_procrustes(local_maps)
    consensus = consensus_per_landmark(aligned, method=aggregation)
    errors, mean_err, max_err = compare_to_truth(consensus, truth)
    bad_errors = {
        bid: float(errors[bid]) if bid in consensus else float("nan")
        for bid in bad_sensors
    }
    return {
        "n_local_maps": len(local_maps),
        "mean_err": mean_err,
        "max_err": max_err,
        "bad_drone_errors": bad_errors,
        "per_drone_errors": errors,
    }


def union_graph_median(
    truth: np.ndarray, comms_range: float,
    bad_sensors: dict[int, float] | None = None,
    noise_sigma: float = 0.0, seed: int = 0,
) -> list[tuple[int, int, float]]:
    """Skip per-drone local embeddings: collect ALL drones' raw range
    measurements into a union edge-set. For each (i, j) pair with multiple
    readings, take the median. Returns deduplicated, median-aggregated edges."""
    n = len(truth)
    if bad_sensors is None:
        bad_sensors = {}
    rng = np.random.default_rng(seed)
    bucket: dict[tuple[int, int], list[float]] = {}
    for obs in range(n):
        obs_bias = bad_sensors.get(obs, 0.0)
        edges = observer_local_edges(
            truth, obs, comms_range,
            bias=obs_bias, noise_sigma=noise_sigma, rng=rng,
        )
        for i, j, d in edges:
            key = (min(i, j), max(i, j))
            bucket.setdefault(key, []).append(d)
    median_edges = []
    for (i, j), reads in bucket.items():
        median_edges.append((i, j, float(np.median(reads))))
    return median_edges


def union_graph_with_sigma(
    truth: np.ndarray, comms_range: float,
    bad_sensors: dict[int, float] | None = None,
    noise_sigma: float = 0.0, seed: int = 0,
) -> list[tuple[int, int, float, float]]:
    """Like union_graph_median but each reading carries an a-priori sigma.
    Honest measurements get sigma = noise_sigma (or a small floor).
    Bad-sensor drones don't KNOW they're bad — they report the same
    sigma as honest drones, but their reading has a +50m offset baked in.
    Returns edges as (i, j, range, sigma)."""
    n = len(truth)
    if bad_sensors is None:
        bad_sensors = {}
    rng = np.random.default_rng(seed)
    # We DON'T median per edge here; we keep both readings (with their
    # a-priori sigmas) so the IRLS can weight them individually.
    edges = []
    a_priori_sigma = max(0.1, noise_sigma)
    for obs in range(n):
        obs_bias = bad_sensors.get(obs, 0.0)
        for j in range(n):
            if j == obs:
                continue
            d_true = float(np.linalg.norm(truth[obs] - truth[j]))
            if d_true > comms_range:
                continue
            measured = d_true + obs_bias
            if noise_sigma > 0:
                measured += float(rng.normal(scale=noise_sigma))
            edges.append((obs, j, max(0.1, measured), a_priori_sigma))
    return edges


def irls_embed(
    n: int, edges_with_sigma: list[tuple[int, int, float, float]],
    initial: np.ndarray | None = None,
    n_iters: int = 5,
    huber_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively Reweighted Least Squares for graph embedding.
    Each edge has (i, j, range, sigma). Initial fit weights edges by
    1/sigma². After each iteration, residuals augment the effective
    sigma: sigma_eff² = sigma² + alpha * residual². Inconsistent edges
    get exponentially down-weighted.

    Returns (positions, final_residuals)."""
    if initial is None:
        rng = np.random.default_rng(42)
        initial = rng.uniform(-15, 15, size=(n, 3))
    x = initial.flatten()
    sigmas = np.array([s for *_, s in edges_with_sigma])
    weights = 1.0 / (sigmas ** 2)

    for it in range(n_iters):
        def residuals(x_flat, _w=weights):
            x = x_flat.reshape((n, 3))
            res = [float(x[0, 0]), float(x[0, 1]), float(x[0, 2])]  # pin anchor
            for k, (i, j, d, _) in enumerate(edges_with_sigma):
                actual = float(np.linalg.norm(x[i] - x[j]))
                # weighted residual
                res.append((actual - d) * np.sqrt(_w[k]))
            return np.array(res, dtype=np.float64)

        result = least_squares(
            residuals, x0=x, max_nfev=300, loss="huber", f_scale=huber_scale,
        )
        x = result.x
        # Compute raw residuals (unweighted) for the IRLS update.
        positions = x.reshape((n, 3))
        raw_res = np.array([
            float(np.linalg.norm(positions[i] - positions[j])) - d
            for i, j, d, _ in edges_with_sigma
        ])
        # Update weights: increase sigma_eff² for high-residual edges.
        # alpha=1.0 means residuals contribute equally with a-priori sigma.
        sigmas_eff_sq = sigmas ** 2 + raw_res ** 2
        weights = 1.0 / sigmas_eff_sq

    positions = x.reshape((n, 3))
    final_res = np.array([
        float(np.linalg.norm(positions[i] - positions[j])) - d
        for i, j, d, _ in edges_with_sigma
    ])
    return positions, final_res


def run_irls_pipeline(
    n: int, seed: int, comms_range: float,
    bad_sensors: dict[int, float] | None = None,
    noise_sigma: float = 0.0,
) -> dict:
    truth = make_swarm(n=n, seed=seed)
    edges_ws = union_graph_with_sigma(
        truth, comms_range, bad_sensors=bad_sensors,
        noise_sigma=noise_sigma, seed=seed,
    )
    # Initialize from ISOMAP on median-aggregated graph for a decent seed.
    bucket: dict[tuple[int, int], list[float]] = {}
    for (i, j, d, _) in edges_ws:
        key = (min(i, j), max(i, j))
        bucket.setdefault(key, []).append(d)
    median_edges = [(i, j, float(np.median(reads))) for (i, j), reads in bucket.items()]
    init = isomap_embed(n, median_edges)
    if np.isnan(init).any():
        rng = np.random.default_rng(seed)
        init = rng.uniform(-15, 15, size=(n, 3))
    positions, final_res = irls_embed(n, edges_ws, initial=init)
    consensus = {i: positions[i] for i in range(n)}
    errors, mean_err, max_err = compare_to_truth(consensus, truth)
    bad_errors = {bid: float(errors[bid]) for bid in (bad_sensors or {})}
    return {
        "mean_err": mean_err, "max_err": max_err,
        "bad_drone_errors": bad_errors, "n_edges": len(edges_ws),
        "final_residuals_rms": float(np.sqrt(np.mean(final_res ** 2))),
    }


def irls_with_dr_anchors(
    n: int,
    edges_with_sigma: list[tuple[int, int, float, float]],
    dr_positions: np.ndarray,            # (n, 3) dead-reckoning self-estimates
    dr_sigmas: np.ndarray,                # (n,) per-drone DR uncertainty (m)
    n_iters: int = 5,
    huber_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full final formulation: weighted least squares with both dead-
    reckoning anchor terms and range-edge terms. IRLS reweights both
    edges (by edge residuals) and anchors (by anchor residuals). The
    weighted optimization is:

        min Σ_i (1/σ_DR_i²) ‖x_i - DR_i‖²
          + Σ_(i,j) (1/σ_R_ij²) (‖x_i - x_j‖ - r_ij)²

    The DR terms break rigid-body and reflection ambiguity (no anchor
    drone needed). The edge terms provide local geometric precision.
    Outliers in either source get downweighted by IRLS.

    Returns (positions, edge_residuals, anchor_residuals)."""
    x = dr_positions.copy().flatten()
    edge_sigmas = np.array([s for *_, s in edges_with_sigma])
    anchor_sigmas = dr_sigmas.copy()
    edge_weights = 1.0 / (edge_sigmas ** 2)
    anchor_weights = 1.0 / (anchor_sigmas ** 2)

    for it in range(n_iters):
        def residuals(x_flat, _ew=edge_weights, _aw=anchor_weights):
            xx = x_flat.reshape((n, 3))
            res = []
            # Anchor terms — DR_i with per-drone weight.
            for i in range(n):
                diff = xx[i] - dr_positions[i]
                res.extend([float(diff[k]) * np.sqrt(_aw[i]) for k in range(3)])
            # Edge terms — range residuals with per-edge weight.
            for k, (i, j, d, _) in enumerate(edges_with_sigma):
                res.append((float(np.linalg.norm(xx[i] - xx[j])) - d) * np.sqrt(_ew[k]))
            return np.array(res, dtype=np.float64)

        result = least_squares(
            residuals, x0=x, max_nfev=400,
            loss="huber", f_scale=huber_scale,
        )
        x = result.x

        positions = x.reshape((n, 3))
        # Raw (unweighted) residuals for IRLS update.
        anchor_raw = np.array([
            float(np.linalg.norm(positions[i] - dr_positions[i])) for i in range(n)
        ])
        edge_raw = np.array([
            float(np.linalg.norm(positions[i] - positions[j])) - d
            for i, j, d, _ in edges_with_sigma
        ])
        # Update effective sigmas: σ_eff² = σ_priori² + residual².
        anchor_weights = 1.0 / (anchor_sigmas ** 2 + anchor_raw ** 2)
        edge_weights = 1.0 / (edge_sigmas ** 2 + edge_raw ** 2)

    positions = x.reshape((n, 3))
    final_edge_res = np.array([
        float(np.linalg.norm(positions[i] - positions[j])) - d
        for i, j, d, _ in edges_with_sigma
    ])
    final_anchor_res = np.array([
        float(np.linalg.norm(positions[i] - dr_positions[i])) for i in range(n)
    ])
    return positions, final_edge_res, final_anchor_res


def run_dr_pipeline(
    n: int, seed: int, comms_range: float,
    bad_dr_drones: dict[int, float] | None = None,
    bad_range_drones: dict[int, float] | None = None,
    dr_drift_sigma: float = 1.0,
    range_noise_sigma: float = 0.3,
) -> dict:
    """Dead-reckoning + ranges pipeline. bad_dr_drones: drone_id -> offset
    (m) for drones with biased dead-reckoning (their DR self-estimate is
    truth + offset). bad_range_drones: drone_id -> offset on their range
    measurements."""
    truth = make_swarm(n=n, seed=seed)
    rng = np.random.default_rng(seed * 31)

    # Dead-reckoning: each drone's DR position is truth + small drift,
    # plus large bias if drone is in bad_dr_drones.
    dr_positions = truth + rng.normal(scale=dr_drift_sigma, size=truth.shape)
    bad_dr_drones = bad_dr_drones or {}
    for did, offset in bad_dr_drones.items():
        dr_positions[did] = dr_positions[did] + np.array([offset, 0, 0])
    dr_sigmas = np.full(n, dr_drift_sigma)
    # Bad-DR drones don't know they're bad — they report standard sigma.
    # IRLS will detect via residuals.

    # Range edges: as before, with bad-range drones biased.
    bad_range_drones = bad_range_drones or {}
    edges = union_graph_with_sigma(
        truth, comms_range,
        bad_sensors=bad_range_drones,
        noise_sigma=range_noise_sigma,
        seed=seed,
    )

    positions, edge_res, anchor_res = irls_with_dr_anchors(
        n, edges, dr_positions, dr_sigmas,
    )
    consensus = {i: positions[i] for i in range(n)}
    errors, mean_err, max_err = compare_to_truth(consensus, truth)
    bad_drone_errors = {}
    for did in list(bad_dr_drones.keys()) + list(bad_range_drones.keys()):
        bad_drone_errors[did] = float(errors[did])
    return {
        "mean_err": mean_err, "max_err": max_err,
        "bad_drone_errors": bad_drone_errors,
        "n_edges": len(edges),
        "max_anchor_residual": float(np.max(np.abs(anchor_res))),
        "max_edge_residual": float(np.max(np.abs(edge_res))),
    }


def run_simpler_pipeline(
    n: int, seed: int, comms_range: float,
    bad_sensors: dict[int, float] | None = None,
    noise_sigma: float = 0.0,
) -> dict:
    truth = make_swarm(n=n, seed=seed)
    edges = union_graph_median(
        truth, comms_range, bad_sensors=bad_sensors,
        noise_sigma=noise_sigma, seed=seed,
    )
    init = isomap_embed(n, edges)
    if np.isnan(init).any():
        return {"mean_err": float("inf"), "max_err": float("inf"), "bad_drone_errors": {}}
    polished = polish_embedding(init, edges)
    consensus = {i: polished[i] for i in range(n)}
    errors, mean_err, max_err = compare_to_truth(consensus, truth)
    bad_errors = {bid: float(errors[bid]) for bid in (bad_sensors or {})}
    return {
        "mean_err": mean_err, "max_err": max_err,
        "bad_drone_errors": bad_errors, "n_edges": len(edges),
        "per_drone_errors": errors,
    }


def main() -> int:
    print("=" * 70)
    print("Test: GPA + per-landmark consensus pipeline")
    print("=" * 70)
    n = 20
    comms_range = 18.0
    seeds = [0, 1, 2, 3, 4]

    print(f"\nScenario A — CLEAN (no bad sensors, no noise)")
    clean_errs = []
    for s in seeds:
        r = run_one_scenario(n=n, seed=s, comms_range=comms_range)
        print(f"  seed={s}  n_maps={r['n_local_maps']}  mean_err={r['mean_err']:.3f}m  max_err={r['max_err']:.3f}m")
        clean_errs.append(r["mean_err"])
    print(f"  AVG mean_err across seeds: {np.mean(clean_errs):.3f}m")

    print(f"\nScenario B — ONE BAD SENSOR (drone 0 has +50m range bias)")
    bad_errs = []; bad_drone_errs = []
    for s in seeds:
        r = run_one_scenario(
            n=n, seed=s, comms_range=comms_range,
            bad_sensors={0: 50.0},
            aggregation="geometric_median",
        )
        be = list(r["bad_drone_errors"].values())[0]
        print(f"  seed={s}  mean_err={r['mean_err']:.3f}m  bad_drone_err={be:.3f}m  (target: <5m)")
        bad_errs.append(r["mean_err"])
        bad_drone_errs.append(be)
    print(f"  AVG mean_err: {np.mean(bad_errs):.3f}m, AVG bad-drone-err: {np.mean(bad_drone_errs):.3f}m")

    print(f"\nScenario C — MULTIPLE BAD SENSORS (drones 0,5,10 each +50m bias)")
    multi_errs = []
    for s in seeds:
        r = run_one_scenario(
            n=n, seed=s, comms_range=comms_range,
            bad_sensors={0: 50.0, 5: 50.0, 10: 50.0},
            aggregation="geometric_median",
        )
        avg_bad = np.mean(list(r["bad_drone_errors"].values()))
        print(f"  seed={s}  mean_err={r['mean_err']:.3f}m  avg_bad_drone_err={avg_bad:.3f}m")
        multi_errs.append(r["mean_err"])
    print(f"  AVG mean_err: {np.mean(multi_errs):.3f}m")

    print(f"\nScenario E — UNION-GRAPH MEDIAN PIPELINE (simpler: per-edge median + global ISOMAP+polish)")
    print(f"  E.1 clean (no bias):")
    for s in seeds:
        r = run_simpler_pipeline(n=n, seed=s, comms_range=comms_range)
        print(f"    seed={s}  edges={r['n_edges']}  mean_err={r['mean_err']:.3f}m  max_err={r['max_err']:.3f}m")
    print(f"  E.2 one bad sensor (drone 0 +50m):")
    for s in seeds:
        r = run_simpler_pipeline(
            n=n, seed=s, comms_range=comms_range,
            bad_sensors={0: 50.0},
        )
        be = list(r["bad_drone_errors"].values())[0] if r["bad_drone_errors"] else float("nan")
        print(f"    seed={s}  edges={r['n_edges']}  mean_err={r['mean_err']:.3f}m  bad_drone_err={be:.3f}m")
    print(f"  E.3 three bad sensors:")
    for s in seeds:
        r = run_simpler_pipeline(
            n=n, seed=s, comms_range=comms_range,
            bad_sensors={0: 50.0, 5: 50.0, 10: 50.0},
        )
        avg_bad = np.mean(list(r["bad_drone_errors"].values())) if r["bad_drone_errors"] else float("nan")
        print(f"    seed={s}  edges={r['n_edges']}  mean_err={r['mean_err']:.3f}m  avg_bad_drone_err={avg_bad:.3f}m")

    print(f"\nScenario F — IRLS PIPELINE (a-priori sigma + iterative reweighting)")
    print(f"  F.1 clean (no bias):")
    for s in seeds:
        r = run_irls_pipeline(n=n, seed=s, comms_range=comms_range)
        print(f"    seed={s}  edges={r['n_edges']}  mean_err={r['mean_err']:.3f}m  "
              f"max_err={r['max_err']:.3f}m  rms_residual={r['final_residuals_rms']:.3f}m")
    print(f"  F.2 one bad sensor (drone 0 +50m):")
    for s in seeds:
        r = run_irls_pipeline(
            n=n, seed=s, comms_range=comms_range, bad_sensors={0: 50.0},
        )
        be = list(r["bad_drone_errors"].values())[0] if r["bad_drone_errors"] else float("nan")
        print(f"    seed={s}  mean_err={r['mean_err']:.3f}m  bad_drone_err={be:.3f}m  "
              f"rms_residual={r['final_residuals_rms']:.3f}m")
    print(f"  F.3 three bad sensors:")
    for s in seeds:
        r = run_irls_pipeline(
            n=n, seed=s, comms_range=comms_range,
            bad_sensors={0: 50.0, 5: 50.0, 10: 50.0},
        )
        avg_bad = np.mean(list(r["bad_drone_errors"].values())) if r["bad_drone_errors"] else float("nan")
        print(f"    seed={s}  mean_err={r['mean_err']:.3f}m  avg_bad_drone_err={avg_bad:.3f}m")

    print(f"\nScenario G — DR-ANCHORED IRLS PIPELINE (final formulation)")
    print(f"  All drones share dead-reckoning + range measurements.")
    print(f"  G.1 clean (small DR drift, no bias):")
    for s in seeds:
        r = run_dr_pipeline(n=n, seed=s, comms_range=comms_range)
        print(f"    seed={s}  edges={r['n_edges']}  mean_err={r['mean_err']:.3f}m  "
              f"max_err={r['max_err']:.3f}m  max_anchor_resid={r['max_anchor_residual']:.3f}m")
    print(f"  G.2 one bad-DR drone (drone 0 DR off by +50m):")
    for s in seeds:
        r = run_dr_pipeline(n=n, seed=s, comms_range=comms_range,
                            bad_dr_drones={0: 50.0})
        be = list(r["bad_drone_errors"].values())[0] if r["bad_drone_errors"] else float("nan")
        print(f"    seed={s}  mean_err={r['mean_err']:.3f}m  bad_drone_err={be:.3f}m  "
              f"max_anchor_resid={r['max_anchor_residual']:.3f}m")
    print(f"  G.3 three bad-DR drones (+50m each):")
    for s in seeds:
        r = run_dr_pipeline(n=n, seed=s, comms_range=comms_range,
                            bad_dr_drones={0: 50.0, 5: 50.0, 10: 50.0})
        avg = np.mean(list(r["bad_drone_errors"].values())) if r["bad_drone_errors"] else float("nan")
        print(f"    seed={s}  mean_err={r['mean_err']:.3f}m  avg_bad_drone_err={avg:.3f}m")
    print(f"  G.4 one bad-DR + one bad-range:")
    for s in seeds:
        r = run_dr_pipeline(n=n, seed=s, comms_range=comms_range,
                            bad_dr_drones={0: 50.0},
                            bad_range_drones={5: 50.0})
        avg = np.mean(list(r["bad_drone_errors"].values())) if r["bad_drone_errors"] else float("nan")
        print(f"    seed={s}  mean_err={r['mean_err']:.3f}m  avg_bad_drone_err={avg:.3f}m")

    print(f"\nScenario D — CLEAN with noise (sigma=0.5m on each range measurement)")
    noisy_errs = []
    for s in seeds:
        r = run_one_scenario(
            n=n, seed=s, comms_range=comms_range, noise_sigma=0.5,
        )
        noisy_errs.append(r["mean_err"])
        print(f"  seed={s}  mean_err={r['mean_err']:.3f}m  max_err={r['max_err']:.3f}m")
    print(f"  AVG mean_err: {np.mean(noisy_errs):.3f}m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
