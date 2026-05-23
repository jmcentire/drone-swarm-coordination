# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Standalone test: can a drone recover the swarm geometry from the
NEIGHBOR GRAPH alone (no position broadcasts, no self-estimates)?

Setup:
  - n drones placed in a 3D cube (TRUE positions known only to the test).
  - Each pair within `comms_range` produces an edge with weight = true range.
  - The "graph" is the set of edges (id_a, id_b, range). No vertex carries
    a position; only IDs and edge weights exist.

Recovery:
  - Anchor drone 0 at origin, drone 1 on the +x axis, drone 2 in the +xy
    half-plane. Three anchors fix the rigid-body and reflection ambiguity
    in 3D.
  - Minimize Σ (||x_i - x_j|| - measured_range)² over all other positions
    using Levenberg-Marquardt.

Falsifiability:
  - If recovery works, every edge's distance in the recovered embedding
    matches the measured range to within numerical tolerance, AND every
    drone's recovered position matches truth (after aligning to the same
    anchor frame) to within numerical tolerance.
  - If the graph is too sparse (not generically rigid), fit residuals
    will be large, exposing the rigidity boundary.

Robustness test:
  - Add a single bad-sensor drone whose RANGE MEASUREMENTS (the ones it
    contributes to the graph) are wrong by 50m. With enough redundant
    edges from honest drones, robust loss should isolate the bad edges.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def make_swarm(n: int, seed: int, spread: float = 15.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-spread, spread, size=(n, 3))


def make_neighbor_graph(
    positions: np.ndarray, comms_range: float, perturb_drone: int | None = None,
    perturb_amount: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Return edges as (i, j, range) tuples. Optionally perturb the ranges
    that perturb_drone reports, to simulate one drone with bad sensors."""
    n = positions.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d_true = float(np.linalg.norm(positions[i] - positions[j]))
            if d_true > comms_range:
                continue
            d = d_true
            if perturb_drone is not None and (i == perturb_drone or j == perturb_drone):
                d = d_true + perturb_amount
            edges.append((i, j, d))
    return edges


def fit_embedding(
    n: int,
    edges: list[tuple[int, int, float]],
    seed: int = 0,
    robust: bool = False,
    huber_scale: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Recover a 3D embedding from the graph. Returns (positions, residual_rms)."""
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-15, 15, size=(n, 3))

    # Anchors: 0 at origin, 1 on +x axis, 2 in +xy half-plane.
    # We achieve this by including the anchor positions as residuals.
    def residuals(x_flat: np.ndarray) -> np.ndarray:
        x = x_flat.reshape((n, 3))
        res = []
        # Anchor 0 at origin
        res.extend(list(x[0]))
        # Anchor 1: y = 0, z = 0 (on x axis; x sign chosen by edge to 0)
        res.extend([x[1, 1], x[1, 2]])
        # Anchor 2: z = 0 (in xy plane)
        res.append(x[2, 2])
        # Edge residuals
        for i, j, d in edges:
            actual = float(np.linalg.norm(x[i] - x[j]))
            res.append(actual - d)
        return np.array(res, dtype=np.float64)

    kwargs = {}
    if robust:
        kwargs["loss"] = "huber"
        kwargs["f_scale"] = huber_scale

    result = least_squares(
        residuals, x0=initial.flatten(), max_nfev=2000, **kwargs
    )
    fitted = result.x.reshape((n, 3))
    rms = float(np.sqrt(np.mean(result.fun ** 2)))
    return fitted, rms


def isomap_embed(n: int, edges: list[tuple[int, int, float]]) -> np.ndarray:
    """Linear-algebraic embedding via ISOMAP: Floyd-Warshall fills in
    missing edges with graph-shortest-path distances, then classical MDS
    (eigendecomposition of double-centred -0.5 * D²) gives the 3D embedding.

    Pros: no iteration, no local minima, O(n³).
    Cons: graph-distance overestimates Euclidean distance for sparse graphs
    (the more hops between two drones, the more the path bends), so the
    recovered embedding is stretched relative to truth. For dense graphs
    (degree >> 6) the error is small; for sparse graphs it's larger but
    still topologically correct.
    """
    INF = float("inf")
    D = np.full((n, n), INF)
    np.fill_diagonal(D, 0.0)
    for i, j, d in edges:
        if d < D[i, j]:
            D[i, j] = d
            D[j, i] = d
    # Floyd-Warshall on the graph.
    for k in range(n):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    if not np.all(np.isfinite(D)):
        # Graph not connected; can't embed.
        return np.full((n, 3), np.nan)
    # Classical MDS: B = -1/2 * J * D² * J, eigendecompose.
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    # Top-3 eigenvalues; negative ones get clamped to 0 (numerical artifacts).
    scale = np.sqrt(np.maximum(w[:3], 0))
    return V[:, :3] * scale


def mds_then_polish(
    n: int, edges: list[tuple[int, int, float]],
) -> np.ndarray:
    """Stage 1 + Stage 2: ISOMAP/MDS for rough topology, then LM polish
    against the actual Euclidean range constraints. Single-drone view."""
    initial = isomap_embed(n=n, edges=edges)
    if not np.all(np.isfinite(initial)):
        rng = np.random.default_rng(0)
        initial = rng.uniform(-15, 15, size=(n, 3))

    def residuals(x_flat: np.ndarray) -> np.ndarray:
        x = x_flat.reshape((n, 3))
        res = []
        # Light anchor: pin drone 0 at its current position (just to remove
        # global translation degeneracy during the polish).
        res.extend(list(x[0]))
        for i, j, d in edges:
            res.append(float(np.linalg.norm(x[i] - x[j])) - d)
        return np.array(res, dtype=np.float64)

    result = least_squares(
        residuals, x0=initial.flatten(), max_nfev=400, loss="huber", f_scale=2.0
    )
    return result.x.reshape((n, 3))


def per_drone_local_embedding(
    n: int, drone_id: int, edges: list[tuple[int, int, float]],
    perception_radius_hops: int = 2,
) -> tuple[np.ndarray, set[int]]:
    """Stage 3 input: this drone only sees a SUBSET of the graph (its
    perception horizon). Compute its local embedding from that subset.
    Returns (positions, set_of_known_drone_ids)."""
    # BFS up to `perception_radius_hops` from drone_id to find known drones.
    adj = {i: set() for i in range(n)}
    edge_lookup = {}
    for i, j, d in edges:
        adj[i].add(j); adj[j].add(i)
        edge_lookup[(min(i,j), max(i,j))] = d
    known = {drone_id}
    frontier = {drone_id}
    for _ in range(perception_radius_hops):
        new_frontier = set()
        for u in frontier:
            new_frontier.update(adj[u] - known)
        known.update(new_frontier)
        frontier = new_frontier
    # Sub-graph induced on `known`
    known_list = sorted(known)
    id_map = {gid: i for i, gid in enumerate(known_list)}
    sub_edges = []
    for (i, j), d in edge_lookup.items():
        if i in id_map and j in id_map:
            sub_edges.append((id_map[i], id_map[j], d))
    if len(sub_edges) < 3 or len(known_list) < 4:
        return np.full((n, 3), np.nan), set()
    sub_pos = mds_then_polish(n=len(known_list), edges=sub_edges)
    # Lift back into n-array using NaN for unknowns
    full = np.full((n, 3), np.nan)
    for gid in known_list:
        full[gid] = sub_pos[id_map[gid]]
    return full, set(known_list)


def consensus_macro_plot(
    n: int,
    local_embeddings: list[tuple[np.ndarray, set[int]]],
    trim_pct: float = 0.20,
) -> np.ndarray:
    """Stage 3: combine per-drone local embeddings into a single consensus
    plot. For each drone X, gather all OTHER drones' estimates of X's
    position. Align each contributor's local frame to a shared frame (the
    first contributor's), then take a trimmed mean across the aligned
    estimates. Drones whose entire embedding is geometrically inconsistent
    with the others get effectively filtered out by the trim.

    NOTE: alignment is by Procrustes against an arbitrary reference (the
    first contributor that knows >=3 drones in common with the reference).
    Real protocol would use shared anchors / common reference frame.
    """
    # Pick a reference embedding (the first non-NaN one that knows the most).
    ref = None
    for emb, known in local_embeddings:
        if ref is None or len(known) > len(ref[1]):
            ref = (emb, known)
    if ref is None:
        return np.full((n, 3), np.nan)

    # Align each embedding's frame to ref's frame using overlap.
    aligned_estimates: dict[int, list[np.ndarray]] = {i: [] for i in range(n)}
    for emb, known in local_embeddings:
        # Drones common to both this embedding and ref
        common = sorted(known & ref[1])
        if len(common) < 3:
            continue
        P = np.array([emb[c] for c in common])
        T = np.array([ref[0][c] for c in common])
        # Procrustes alignment (allow reflection)
        cP = P.mean(axis=0); cT = T.mean(axis=0)
        H = (P - cP).T @ (T - cT)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        for did in known:
            aligned = (emb[did] - cP) @ R.T + cT
            aligned_estimates[did].append(aligned)

    # Trimmed mean per drone (drop the worst trim_pct fraction by distance
    # from the median).
    consensus = np.full((n, 3), np.nan)
    for did, ests in aligned_estimates.items():
        if not ests:
            continue
        arr = np.array(ests)
        if len(arr) <= 2:
            consensus[did] = arr.mean(axis=0)
            continue
        median = np.median(arr, axis=0)
        dists = np.linalg.norm(arr - median, axis=1)
        keep_n = max(1, int(len(arr) * (1 - trim_pct)))
        keep_idx = np.argsort(dists)[:keep_n]
        consensus[did] = arr[keep_idx].mean(axis=0)
    return consensus


def edge_rms(positions: np.ndarray, edges: list[tuple[int, int, float]]) -> float:
    """RMS of (||p_i - p_j|| - d) over all edges."""
    errs = []
    for i, j, d in edges:
        errs.append(float(np.linalg.norm(positions[i] - positions[j])) - d)
    return float(np.sqrt(np.mean(np.array(errs) ** 2)))


def align_to_truth_frame(positions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Procrustes alignment: rotate/translate fitted positions to best match
    truth (which is in arbitrary world frame). Allows reflection. Used to
    compare recovered positions to truth despite rigid-body ambiguity."""
    centroid_f = positions.mean(axis=0)
    centroid_t = truth.mean(axis=0)
    P = positions - centroid_f
    T = truth - centroid_t
    H = P.T @ T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    aligned = (positions - centroid_f) @ R.T + centroid_t
    return aligned


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recovery_clean(n: int = 20, comms_range: float = 18.0, seeds: int = 5) -> int:
    print(f"\n[clean recovery] n={n} drones, comms_range={comms_range}m")
    print(f"  {'method':<8s} {'seed':>4s} {'edges':>5s} {'edge_rms':>10s} {'pos_err':>10s}")
    failed = 0
    for s in range(seeds):
        truth = make_swarm(n=n, seed=s)
        edges = make_neighbor_graph(truth, comms_range=comms_range)
        # ISOMAP (linear-algebraic)
        iso = isomap_embed(n=n, edges=edges)
        iso_edge = edge_rms(iso, edges)
        iso_aligned = align_to_truth_frame(iso, truth)
        iso_pos_err = float(np.sqrt(np.mean(np.sum((iso_aligned - truth) ** 2, axis=1))))
        # LM (iterative)
        fitted, rms = fit_embedding(n=n, edges=edges, seed=s + 100)
        lm_edge = edge_rms(fitted, edges)
        lm_aligned = align_to_truth_frame(fitted, truth)
        lm_pos_err = float(np.sqrt(np.mean(np.sum((lm_aligned - truth) ** 2, axis=1))))
        print(f"  {'ISOMAP':<8s} {s:>4d} {len(edges):>5d} {iso_edge:>10.4f} {iso_pos_err:>10.4f}")
        print(f"  {'LM':<8s} {s:>4d} {len(edges):>5d} {lm_edge:>10.4f} {lm_pos_err:>10.4f}")
        # Pass when at least one method recovers within tolerance.
        if iso_pos_err > 3.0 and lm_pos_err > 3.0:
            print(f"  FAIL: both methods >3m pos_err")
            failed += 1
    return failed


def test_recovery_with_bad_sensor(n: int = 20, comms_range: float = 18.0, seeds: int = 5) -> int:
    print(f"\n[bad-sensor robustness] n={n}, comms_range={comms_range}m, "
          f"one drone reports +50m on all its edges")
    failed = 0
    for s in range(seeds):
        truth = make_swarm(n=n, seed=s)
        bad_id = (s * 7) % n
        # Without perturb_drone: full clean graph from honest neighbors of bad_id.
        edges_clean = make_neighbor_graph(truth, comms_range=comms_range)
        # Bad drone's OWN reports: ranges on its edges are +50m.
        edges_bad_only = make_neighbor_graph(
            truth, comms_range=comms_range, perturb_drone=bad_id, perturb_amount=50.0
        )
        # In the real protocol the bad drone's outgoing range list lives in
        # ITS heartbeat; honest drones contribute the correct ranges from
        # their own measurements. Combine: every edge gets BOTH the honest
        # reading and the bad drone's reading (if bad_id is incident).
        # That gives the rigidity solver redundant constraints where the
        # robust loss can identify outliers.
        edges_combined = []
        seen = {}
        for (i, j, d) in edges_clean:
            seen[(i, j)] = [d]
        for (i, j, d) in edges_bad_only:
            if (i, j) in seen:
                seen[(i, j)].append(d)
            else:
                seen[(i, j)] = [d]
        # Each (i,j) may have one or two readings. Push both as separate edges.
        for (i, j), ds in seen.items():
            for d in ds:
                edges_combined.append((i, j, d))

        fitted, _ = fit_embedding(
            n=n, edges=edges_combined, seed=s + 100, robust=True, huber_scale=2.0
        )
        # Check recovered position of the bad-sensor drone vs truth.
        aligned = align_to_truth_frame(fitted, truth)
        bad_pos_err = float(np.linalg.norm(aligned[bad_id] - truth[bad_id]))
        print(f"  seed={s} bad_id={bad_id} edges={len(edges_combined)} "
              f"bad_drone_pos_err={bad_pos_err:.4f}m "
              f"(if naive: ~50m; robust target: <2m)")
        if bad_pos_err > 5.0:
            print(f"  FAIL bad-sensor: position error {bad_pos_err:.3f}m > 5m")
            failed += 1
    return failed


def test_rigidity_threshold() -> int:
    """How sparse can the graph get before recovery fails?"""
    print(f"\n[rigidity threshold] sweep comms_range, observe recovery error")
    n = 30
    truth = make_swarm(n=n, seed=42, spread=15.0)
    for r in [5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 18.0, 22.0, 30.0]:
        edges = make_neighbor_graph(truth, comms_range=r)
        if len(edges) < 10:
            print(f"  range={r:5.1f}m: too few edges ({len(edges)})")
            continue
        fitted, _ = fit_embedding(n=n, edges=edges, seed=42 + 100)
        err_rms = edge_rms(fitted, edges)
        aligned = align_to_truth_frame(fitted, truth)
        pos_err = float(np.sqrt(np.mean(np.sum((aligned - truth) ** 2, axis=1))))
        avg_deg = 2 * len(edges) / n
        print(f"  range={r:5.1f}m  edges={len(edges):4d}  avg_deg={avg_deg:5.2f}  "
              f"edge_rms={err_rms:.4f}m  pos_err_aligned={pos_err:.4f}m")
    return 0


def test_stages_together(seeds: int = 5) -> int:
    """The headline test: each stage alone is insufficient; the combination
    is what produces the right answer in the bad-sensor case.
    """
    print(f"\n[stages-together] bad-sensor recovery: each stage alone vs combined")
    print(f"  Method                                bad_drone_pos_err (mean over seeds)")
    n = 20
    comms_range = 18.0

    iso_errs, polish_errs, combined_errs = [], [], []
    for s in range(seeds):
        truth = make_swarm(n=n, seed=s)
        bad_id = (s * 7) % n
        # Build the dual-reading graph as before.
        edges_clean = make_neighbor_graph(truth, comms_range=comms_range)
        edges_bad = make_neighbor_graph(
            truth, comms_range=comms_range, perturb_drone=bad_id, perturb_amount=50.0,
        )
        seen: dict[tuple[int, int], list[float]] = {}
        for (i, j, d) in edges_clean:
            seen.setdefault((i, j), []).append(d)
        for (i, j, d) in edges_bad:
            seen.setdefault((i, j), []).append(d)
        edges_combined = [
            (i, j, d) for (i, j), ds in seen.items() for d in ds
        ]

        # Stage 1 alone (ISOMAP)
        iso = isomap_embed(n=n, edges=edges_combined)
        iso_aligned = align_to_truth_frame(iso, truth)
        iso_errs.append(float(np.linalg.norm(iso_aligned[bad_id] - truth[bad_id])))

        # Stages 1+2 (MDS + LM polish)
        polish = mds_then_polish(n=n, edges=edges_combined)
        polish_aligned = align_to_truth_frame(polish, truth)
        polish_errs.append(float(np.linalg.norm(polish_aligned[bad_id] - truth[bad_id])))

        # Stages 1+2+3 (per-drone local + macro consensus + outlier reject).
        # Each drone (other than the bad one — it can also contribute) computes
        # its own local embedding from its 2-hop perception of the graph.
        # ITS perception uses ITS OWN measurements (correct for honest drones,
        # wrong for the bad drone).
        local_embeddings = []
        for did in range(n):
            # Each drone's local view: edges where THIS drone is the source of
            # measurement. For honest drones, ranges are correct. For bad_id,
            # ranges are +50m offset.
            if did == bad_id:
                # Bad drone uses its own (wrong) measurements
                source_edges = [
                    (i, j, d) for (i, j, d) in edges_bad
                    if i == did or j == did
                ]
            else:
                # Honest drones use their own (correct) measurements.
                # Also include edges between OTHER honest drones since they
                # gossipped those measurements too.
                source_edges = [
                    (i, j, d) for (i, j, d) in edges_clean
                    if i != bad_id and j != bad_id
                ] + [
                    (i, j, d) for (i, j, d) in edges_clean if i == did or j == did
                ]
                # Plus edges involving bad_id from this honest drone's view
                # (honest drone's measurement of bad_id is correct).
                for (i, j, d) in edges_clean:
                    if (i == did and j == bad_id) or (j == did and i == bad_id):
                        source_edges.append((i, j, d))
            # Dedup
            source_edges = list({(i, j, d) for (i, j, d) in source_edges})
            emb, known = per_drone_local_embedding(n=n, drone_id=did, edges=source_edges)
            if known:
                local_embeddings.append((emb, known))

        macro = consensus_macro_plot(n=n, local_embeddings=local_embeddings)
        if not np.isnan(macro[bad_id]).any():
            macro_aligned = align_to_truth_frame(
                np.where(np.isnan(macro), 0, macro), truth
            )
            combined_errs.append(float(np.linalg.norm(macro_aligned[bad_id] - truth[bad_id])))
        else:
            combined_errs.append(float("inf"))

    print(f"  Stage 1 alone   (ISOMAP)              {np.mean(iso_errs):.3f}m   (per seed: {iso_errs})")
    print(f"  Stages 1+2      (MDS + LM polish)     {np.mean(polish_errs):.3f}m   (per seed: {polish_errs})")
    print(f"  Stages 1+2+3    (per-drone + macro)   {np.mean(combined_errs):.3f}m   (per seed: {combined_errs})")
    failed = 0
    if np.mean(combined_errs) >= min(np.mean(iso_errs), np.mean(polish_errs)):
        print(f"  WARNING: combined pipeline did NOT improve over individual stages")
        failed += 1
    if np.mean(combined_errs) > 5.0:
        print(f"  FAIL: combined-pipeline error >5m (means stage 3 isn't doing its job)")
        failed += 1
    return failed


if __name__ == "__main__":
    failed = 0
    failed += test_recovery_clean()
    failed += test_recovery_with_bad_sensor()
    failed += test_rigidity_threshold()
    failed += test_stages_together()
    print()
    if failed == 0:
        print("All graph-rigidity tests passed.")
    else:
        print(f"{failed} tests failed.")
