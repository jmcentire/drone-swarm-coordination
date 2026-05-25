# /// script
# dependencies = ["numpy<3"]
# ///
"""Vertical acoustic GPS-relay benchmark.

Question:
  If a surface GPS-known group seeds an acoustic localization frame, is it
  better to place relay triads every ~1500 m or a denser chain with one relay
  every ~500 m, given that any 1500 m section can see at least three relays?

This bench models localization only. It does not classify contacts or plan
mine removal. It compares:
  * triad_hops_1500m: three drones at each relay depth.
  * dense_chain_500m: one drone every 500 m on a helical path, so local
    1500 m windows contain multiple non-collinear anchors.

Each geometry is solved two ways:
  * sequential: downstream nodes localize from already-estimated upstream
    anchors, so errors compound hop-by-hop.
  * global: a pose-graph least-squares solve uses all delivered range edges,
    so redundant dense-chain constraints can smooth errors.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_channel import AcousticChannelConfig, MessageProfile, estimate_link, packet_delivery_rate


RELAY_PROFILE = MessageProfile(
    name="relay_localization_ping",
    payload_bytes=64,
    rate_hz=1.0 / 15.0,
    description="Two-way ranging/localization exchange over a 15 s update window.",
    fanout=1,
    safety_factor=2.0,
)


@dataclass(frozen=True)
class RelayScenario:
    name: str
    depth_m: float
    strategy: str
    spacing_m: float
    comms_range_m: float = 1_650.0
    lateral_radius_m: float = 250.0
    n_seeds: int = 100
    range_noise_m: float = 2.0
    range_noise_ppm: float = 300.0
    depth_noise_m: float = 1.0
    gps_noise_m: float = 1.5
    drift_xy_m_per_km: float = 8.0
    extra_packet_loss: float = 0.0
    failed_relay_count: int = 0


@dataclass
class SolverResult:
    ok: bool
    bottom_rmse_m: float
    bottom_max_error_m: float
    all_rmse_m: float
    max_error_m: float
    mean_cov_trace_m2: float
    max_cov_trace_m2: float
    normal_rank: int
    normal_condition: float
    estimated_nodes: int
    failed_bottom_nodes: int


def _surface_triad(radius: float) -> np.ndarray:
    angles = np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    return np.stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros(3),
    ], axis=1)


def _triad_geometry(depth_m: float, spacing_m: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    levels = [0.0]
    z = spacing_m
    while z < depth_m - 1e-9:
        levels.append(z)
        z += spacing_m
    if levels[-1] != depth_m:
        levels.append(depth_m)

    pts: list[np.ndarray] = []
    anchors: list[bool] = []
    for idx, depth in enumerate(levels):
        triad = _surface_triad(radius)
        theta = (idx % 3) * math.pi / 9.0
        rot = np.array([
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        triad = triad @ rot.T
        triad[:, 2] = -depth
        pts.extend(triad)
        anchors.extend([idx == 0] * 3)
    return np.asarray(pts, dtype=np.float64), np.asarray(anchors, dtype=bool)


def _dense_geometry(depth_m: float, spacing_m: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    pts: list[np.ndarray] = [p for p in _surface_triad(radius)]
    anchors: list[bool] = [True, True, True]
    n_steps = int(math.ceil(depth_m / spacing_m))
    for step in range(1, n_steps + 1):
        depth = min(depth_m, step * spacing_m)
        angle = (step - 1) * 2.0 * math.pi / 3.0
        pts.append(np.array([radius * math.cos(angle), radius * math.sin(angle), -depth]))
        anchors.append(False)
    return np.asarray(pts, dtype=np.float64), np.asarray(anchors, dtype=bool)


def _geometry(spec: RelayScenario) -> tuple[np.ndarray, np.ndarray]:
    if spec.strategy == "triad_hops":
        return _triad_geometry(spec.depth_m, spec.spacing_m, spec.lateral_radius_m)
    if spec.strategy == "dense_chain":
        return _dense_geometry(spec.depth_m, spec.spacing_m, spec.lateral_radius_m)
    raise ValueError(f"unknown strategy: {spec.strategy}")


def _apply_drift(nominal: np.ndarray, anchors: np.ndarray, spec: RelayScenario, rng: np.random.Generator) -> np.ndarray:
    true = nominal.copy()
    for i in range(len(true)):
        if anchors[i]:
            continue
        depth_km = abs(true[i, 2]) / 1_000.0
        sigma = spec.drift_xy_m_per_km * max(depth_km, 0.1)
        true[i, :2] += rng.normal(0.0, sigma, size=2)
    return true


def _failed_nodes(anchors: np.ndarray, spec: RelayScenario, rng: np.random.Generator) -> set[int]:
    if spec.failed_relay_count <= 0:
        return set()
    candidates = np.where(~anchors)[0]
    if len(candidates) == 0:
        return set()
    n = min(spec.failed_relay_count, len(candidates))
    return set(int(x) for x in rng.choice(candidates, size=n, replace=False))


def _range_edges(
    true_pos: np.ndarray,
    spec: RelayScenario,
    failed: set[int],
    channel: AcousticChannelConfig,
    rng: np.random.Generator,
) -> list[tuple[int, int, float, float, float]]:
    edges: list[tuple[int, int, float, float, float]] = []
    n = len(true_pos)
    for i in range(n):
        if i in failed:
            continue
        for j in range(i + 1, n):
            if j in failed:
                continue
            d = float(np.linalg.norm(true_pos[i] - true_pos[j]))
            if d > spec.comms_range_m:
                continue
            pdr, _ = packet_delivery_rate(d, RELAY_PROFILE.payload_bytes, channel)
            p_deliver = max(0.0, min(1.0, pdr * (1.0 - spec.extra_packet_loss)))
            if rng.random() > p_deliver:
                continue
            sigma = spec.range_noise_m + d * spec.range_noise_ppm * 1e-6
            measured = d + rng.normal(0.0, sigma)
            edges.append((i, j, measured, sigma, p_deliver))
    return edges


def _bottom_nodes(pos: np.ndarray) -> np.ndarray:
    z_min = float(np.min(pos[:, 2]))
    return np.where(np.isclose(pos[:, 2], z_min))[0]


def _solve_global(
    nominal: np.ndarray,
    true_pos: np.ndarray,
    anchors: np.ndarray,
    anchor_xy: np.ndarray,
    z_est: np.ndarray,
    failed: set[int],
    edges: list[tuple[int, int, float, float, float]],
) -> SolverResult:
    active = np.array([i not in failed for i in range(len(nominal))], dtype=bool)
    unknown = np.where(active & ~anchors)[0]
    idx = {node: k for k, node in enumerate(unknown)}
    x = nominal[unknown, :2].reshape(-1).copy()
    if len(unknown) == 0:
        return _score_solution(nominal[:, :2], true_pos, z_est, active, anchors, unknown, np.zeros((0, 0)))

    for _ in range(30):
        residuals: list[float] = []
        jac_rows: list[np.ndarray] = []
        for i, j, measured, sigma, _ in edges:
            if not active[i] or not active[j]:
                continue
            pi = _xy_for_node(i, x, idx, anchor_xy, nominal)
            pj = _xy_for_node(j, x, idx, anchor_xy, nominal)
            dz = z_est[i] - z_est[j]
            diff = np.array([pi[0] - pj[0], pi[1] - pj[1], dz])
            pred = max(float(np.linalg.norm(diff)), 1e-9)
            w = 1.0 / max(sigma, 1e-6)
            residuals.append((pred - measured) * w)
            row = np.zeros(2 * len(unknown))
            grad = diff[:2] / pred * w
            if i in idx:
                row[2 * idx[i]:2 * idx[i] + 2] += grad
            if j in idx:
                row[2 * idx[j]:2 * idx[j] + 2] -= grad
            jac_rows.append(row)
        for node in unknown:
            prior_sigma = max(50.0, 2.0 * abs(nominal[node, 2]) / 1_000.0 * 35.0)
            cur = _xy_for_node(node, x, idx, anchor_xy, nominal)
            for axis in range(2):
                row = np.zeros(2 * len(unknown))
                row[2 * idx[node] + axis] = 1.0 / prior_sigma
                residuals.append((cur[axis] - nominal[node, axis]) / prior_sigma)
                jac_rows.append(row)
        if len(jac_rows) < len(x):
            break
        jmat = np.vstack(jac_rows)
        rvec = np.asarray(residuals)
        delta, *_ = np.linalg.lstsq(jmat, -rvec, rcond=None)
        x += delta
        if float(np.linalg.norm(delta)) < 1e-5:
            break

    xy = nominal[:, :2].copy()
    xy[anchors] = anchor_xy[anchors]
    xy[unknown] = x.reshape((-1, 2))
    normal = np.zeros((2 * len(unknown), 2 * len(unknown)))
    if len(unknown):
        rows = []
        for i, j, _, sigma, _ in edges:
            if not active[i] or not active[j]:
                continue
            pi = xy[i]
            pj = xy[j]
            dz = z_est[i] - z_est[j]
            diff = np.array([pi[0] - pj[0], pi[1] - pj[1], dz])
            pred = max(float(np.linalg.norm(diff)), 1e-9)
            row = np.zeros(2 * len(unknown))
            grad = diff[:2] / pred / max(sigma, 1e-6)
            if i in idx:
                row[2 * idx[i]:2 * idx[i] + 2] += grad
            if j in idx:
                row[2 * idx[j]:2 * idx[j] + 2] -= grad
            rows.append(row)
        for node in unknown:
            prior_sigma = max(50.0, 2.0 * abs(nominal[node, 2]) / 1_000.0 * 35.0)
            for axis in range(2):
                row = np.zeros(2 * len(unknown))
                row[2 * idx[node] + axis] = 1.0 / prior_sigma
                rows.append(row)
        if rows:
            jmat = np.vstack(rows)
            normal = jmat.T @ jmat
    result = _score_solution(xy, true_pos, z_est, active, anchors, unknown, normal)
    if len(unknown) and result.normal_rank < 2 * len(unknown):
        result.ok = False
        result.bottom_rmse_m = float("inf")
        result.bottom_max_error_m = float("inf")
        result.all_rmse_m = float("inf")
        result.max_error_m = float("inf")
    return result


def _xy_for_node(
    node: int,
    x: np.ndarray,
    idx: dict[int, int],
    anchor_xy: np.ndarray,
    nominal: np.ndarray,
) -> np.ndarray:
    if node in idx:
        return x[2 * idx[node]:2 * idx[node] + 2]
    if node < len(anchor_xy):
        return anchor_xy[node]
    return nominal[node, :2]


def _solve_sequential(
    nominal: np.ndarray,
    true_pos: np.ndarray,
    anchors: np.ndarray,
    anchor_xy: np.ndarray,
    z_est: np.ndarray,
    failed: set[int],
    edges: list[tuple[int, int, float, float, float]],
) -> SolverResult:
    active = np.array([i not in failed for i in range(len(nominal))], dtype=bool)
    known = np.zeros(len(nominal), dtype=bool)
    known[anchors & active] = True
    xy = nominal[:, :2].copy()
    xy[anchors] = anchor_xy[anchors]
    edge_map: dict[int, list[tuple[int, float, float]]] = {i: [] for i in range(len(nominal))}
    for i, j, measured, sigma, _ in edges:
        edge_map[i].append((j, measured, sigma))
        edge_map[j].append((i, measured, sigma))

    order = sorted([i for i in range(len(nominal)) if active[i] and not anchors[i]], key=lambda i: abs(nominal[i, 2]))
    for node in order:
        refs = [(j, measured, sigma) for j, measured, sigma in edge_map[node] if known[j]]
        if len(refs) < 3:
            continue
        est = xy[node].copy()
        for _ in range(20):
            residuals = []
            jac_rows = []
            for ref, measured, sigma in refs:
                diff = np.array([est[0] - xy[ref, 0], est[1] - xy[ref, 1], z_est[node] - z_est[ref]])
                pred = max(float(np.linalg.norm(diff)), 1e-9)
                w = 1.0 / max(sigma, 1e-6)
                residuals.append((pred - measured) * w)
                jac_rows.append(diff[:2] / pred * w)
            jmat = np.vstack(jac_rows)
            rvec = np.asarray(residuals)
            delta, *_ = np.linalg.lstsq(jmat, -rvec, rcond=None)
            est += delta
            if float(np.linalg.norm(delta)) < 1e-5:
                break
        xy[node] = est
        known[node] = True

    unknown = np.where(active & ~anchors)[0]
    normal = np.eye(max(1, 2 * len(unknown))) * 1e-9
    return _score_solution(xy, true_pos, z_est, known, anchors, unknown, normal)


def _score_solution(
    xy: np.ndarray,
    true_pos: np.ndarray,
    z_est: np.ndarray,
    estimated: np.ndarray,
    anchors: np.ndarray,
    unknown: np.ndarray,
    normal: np.ndarray,
) -> SolverResult:
    est_pos = np.column_stack([xy, z_est])
    bottom = _bottom_nodes(true_pos)
    bottom_estimated = np.array([i for i in bottom if estimated[i]], dtype=int)
    failed_bottom = int(len(bottom) - len(bottom_estimated))
    active_est = np.where(estimated)[0]
    if len(active_est):
        errs = np.linalg.norm(est_pos[active_est] - true_pos[active_est], axis=1)
        all_rmse = float(np.sqrt(np.mean(errs ** 2)))
        max_err = float(np.max(errs))
    else:
        all_rmse = float("inf")
        max_err = float("inf")
    if len(bottom_estimated):
        b_errs = np.linalg.norm(est_pos[bottom_estimated] - true_pos[bottom_estimated], axis=1)
        bottom_rmse = float(np.sqrt(np.mean(b_errs ** 2)))
        bottom_max = float(np.max(b_errs))
    else:
        bottom_rmse = float("inf")
        bottom_max = float("inf")

    rank = 0
    cond = float("inf")
    traces: list[float] = []
    if normal.size > 1:
        s = np.linalg.svd(normal, compute_uv=False)
        if len(s):
            rank = int(np.sum(s > max(s[0] * 1e-9, 1e-12)))
            if rank == len(s) and s[-1] > 0.0:
                cond = float(s[0] / s[-1])
            cov = np.linalg.pinv(normal, rcond=1e-9)
            for k, node in enumerate(unknown):
                if estimated[node]:
                    block = cov[2 * k:2 * k + 2, 2 * k:2 * k + 2]
                    traces.append(float(np.trace(block)))
    mean_trace = float(np.mean(traces)) if traces else float("inf")
    max_trace = float(np.max(traces)) if traces else float("inf")
    return SolverResult(
        ok=failed_bottom == 0 and math.isfinite(bottom_rmse),
        bottom_rmse_m=bottom_rmse,
        bottom_max_error_m=bottom_max,
        all_rmse_m=all_rmse,
        max_error_m=max_err,
        mean_cov_trace_m2=mean_trace,
        max_cov_trace_m2=max_trace,
        normal_rank=rank,
        normal_condition=cond,
        estimated_nodes=int(np.sum(estimated)),
        failed_bottom_nodes=failed_bottom,
    )


def _percentiles(values: list[float]) -> list[float]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if len(finite) == 0:
        return [float("inf"), float("inf"), float("inf")]
    return [float(np.mean(finite)), float(np.percentile(finite, 5)), float(np.percentile(finite, 95))]


def _run_seed(spec: RelayScenario, seed: int, channel: AcousticChannelConfig) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    nominal, anchors = _geometry(spec)
    true_pos = _apply_drift(nominal, anchors, spec, rng)
    failed = _failed_nodes(anchors, spec, rng)
    anchor_xy = nominal[:, :2].copy()
    anchor_xy[anchors] = true_pos[anchors, :2] + rng.normal(0.0, spec.gps_noise_m, size=(int(np.sum(anchors)), 2))
    z_est = true_pos[:, 2] + rng.normal(0.0, spec.depth_noise_m, size=len(true_pos))
    z_est[anchors] = true_pos[anchors, 2] + rng.normal(0.0, spec.gps_noise_m, size=int(np.sum(anchors)))
    edges = _range_edges(true_pos, spec, failed, channel, rng)
    sequential = _solve_sequential(nominal, true_pos, anchors, anchor_xy, z_est, failed, edges)
    global_solve = _solve_global(nominal, true_pos, anchors, anchor_xy, z_est, failed, edges)
    link_est = estimate_link(min(spec.spacing_m, spec.comms_range_m), RELAY_PROFILE, channel)
    possible_edges = 0
    for i in range(len(true_pos)):
        if i in failed:
            continue
        for j in range(i + 1, len(true_pos)):
            if j in failed:
                continue
            if float(np.linalg.norm(true_pos[i] - true_pos[j])) <= spec.comms_range_m:
                possible_edges += 1
    return {
        "seed": seed,
        "n_drones": int(len(true_pos) - len(failed)),
        "n_failed_relays": len(failed),
        "possible_edges": possible_edges,
        "delivered_edges": len(edges),
        "edge_delivery_ratio": len(edges) / possible_edges if possible_edges else 0.0,
        "nominal_link_pdr_at_spacing": link_est.packet_delivery_rate,
        "nominal_link_retry_at_spacing": link_est.retry_burden,
        "nominal_link_occupancy_at_spacing": link_est.channel_occupancy,
        "sequential": asdict(sequential),
        "global": asdict(global_solve),
    }


def _summarize(spec: RelayScenario, seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scenario": asdict(spec),
        "n": len(seed_rows),
        "n_drones": _percentiles([r["n_drones"] for r in seed_rows]),
        "delivered_edges": _percentiles([r["delivered_edges"] for r in seed_rows]),
        "edge_delivery_ratio": _percentiles([r["edge_delivery_ratio"] for r in seed_rows]),
        "nominal_link_pdr_at_spacing": _percentiles([r["nominal_link_pdr_at_spacing"] for r in seed_rows]),
        "nominal_link_retry_at_spacing": _percentiles([r["nominal_link_retry_at_spacing"] for r in seed_rows]),
    }
    for solver in ("sequential", "global"):
        vals = [r[solver] for r in seed_rows]
        out[solver] = {
            "success_rate": sum(1 for v in vals if v["ok"]) / len(vals),
            "bottom_rmse_m": _percentiles([v["bottom_rmse_m"] for v in vals]),
            "bottom_max_error_m": _percentiles([v["bottom_max_error_m"] for v in vals]),
            "all_rmse_m": _percentiles([v["all_rmse_m"] for v in vals]),
            "max_error_m": _percentiles([v["max_error_m"] for v in vals]),
            "mean_cov_trace_m2": _percentiles([v["mean_cov_trace_m2"] for v in vals]),
            "max_cov_trace_m2": _percentiles([v["max_cov_trace_m2"] for v in vals]),
            "estimated_nodes": _percentiles([v["estimated_nodes"] for v in vals]),
            "failed_bottom_nodes_mean": float(np.mean([v["failed_bottom_nodes"] for v in vals])),
        }
    return out


def _comparison_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float], dict[str, dict[str, Any]]] = {}
    for row in rows:
        name = row["scenario"]["name"]
        regime = name.rsplit("_", 2)[0]
        depth = float(row["scenario"]["depth_m"])
        grouped.setdefault((regime, depth), {})[row["scenario"]["strategy"]] = row
    for (regime, depth), strategies in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        triad = strategies.get("triad_hops")
        dense = strategies.get("dense_chain")
        if triad is None or dense is None:
            continue
        triad_global = triad["global"]
        dense_global = dense["global"]
        triad_seq = triad["sequential"]
        dense_seq = dense["sequential"]
        if triad_global["success_rate"] != dense_global["success_rate"]:
            global_winner = "triad_hops" if triad_global["success_rate"] > dense_global["success_rate"] else "dense_chain"
        else:
            global_winner = (
                "triad_hops"
                if triad_global["bottom_rmse_m"][0] <= dense_global["bottom_rmse_m"][0]
                else "dense_chain"
            )
        if triad_seq["success_rate"] != dense_seq["success_rate"]:
            sequential_winner = "triad_hops" if triad_seq["success_rate"] > dense_seq["success_rate"] else "dense_chain"
        else:
            sequential_winner = (
                "triad_hops"
                if triad_seq["bottom_rmse_m"][0] <= dense_seq["bottom_rmse_m"][0]
                else "dense_chain"
            )
        pairs.append({
            "regime": regime,
            "depth_m": depth,
            "global_winner": global_winner,
            "sequential_winner": sequential_winner,
            "triad_global_success": triad_global["success_rate"],
            "dense_global_success": dense_global["success_rate"],
            "triad_global_bottom_rmse_m": triad_global["bottom_rmse_m"][0],
            "dense_global_bottom_rmse_m": dense_global["bottom_rmse_m"][0],
            "triad_sequential_success": triad_seq["success_rate"],
            "dense_sequential_success": dense_seq["success_rate"],
            "triad_sequential_bottom_rmse_m": triad_seq["bottom_rmse_m"][0],
            "dense_sequential_bottom_rmse_m": dense_seq["bottom_rmse_m"][0],
            "triad_drones": triad["n_drones"][0],
            "dense_drones": dense["n_drones"][0],
        })
    return pairs


def _make_scenarios(depths: list[float], seeds: int) -> list[RelayScenario]:
    scenarios: list[RelayScenario] = []
    regimes = [
        ("nominal", 2.0, 300.0, 8.0, 0.0, 0),
        ("noisy_drift", 6.0, 800.0, 35.0, 0.05, 0),
        ("one_failed_relay", 3.0, 500.0, 15.0, 0.02, 1),
    ]
    for depth in depths:
        for regime, noise_m, ppm, drift, extra_loss, failures in regimes:
            scenarios.append(RelayScenario(
                name=f"{regime}_triad_{int(depth)}m",
                depth_m=depth,
                strategy="triad_hops",
                spacing_m=1_500.0,
                n_seeds=seeds,
                range_noise_m=noise_m,
                range_noise_ppm=ppm,
                drift_xy_m_per_km=drift,
                extra_packet_loss=extra_loss,
                failed_relay_count=failures,
            ))
            scenarios.append(RelayScenario(
                name=f"{regime}_dense_{int(depth)}m",
                depth_m=depth,
                strategy="dense_chain",
                spacing_m=500.0,
                n_seeds=seeds,
                range_noise_m=noise_m,
                range_noise_ppm=ppm,
                drift_xy_m_per_km=drift,
                extra_packet_loss=extra_loss,
                failed_relay_count=failures,
            ))
    return scenarios


def _parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--depths", type=_parse_float_list, default=[1_000.0, 3_000.0, 6_000.0])
    ap.add_argument("--output", default="underwater/bench_vertical_relay_results.json")
    args = ap.parse_args()

    channel = AcousticChannelConfig()
    scenarios = _make_scenarios(sorted(args.depths), args.seeds)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for idx, spec in enumerate(scenarios, start=1):
        print(f"\n[{idx}/{len(scenarios)}] {spec.name} seeds={spec.n_seeds} jobs={args.jobs}", flush=True)
        ts = time.perf_counter()
        seed_ids = list(range(spec.n_seeds))
        if args.jobs > 1:
            run_one = partial(_run_seed, spec, channel=channel)
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
                seed_rows = list(ex.map(run_one, seed_ids))
        else:
            seed_rows = [_run_seed(spec, s, channel) for s in seed_ids]
        summary = _summarize(spec, seed_rows)
        summary["wall_time_s"] = time.perf_counter() - ts
        rows.append(summary)
        print(
            f"  seq bottom rmse={summary['sequential']['bottom_rmse_m'][0]:.2f}m "
            f"global bottom rmse={summary['global']['bottom_rmse_m'][0]:.2f}m "
            f"edges={summary['delivered_edges'][0]:.1f} "
            f"drones={summary['n_drones'][0]:.1f}",
            flush=True,
        )

    artifact = {
        "method": {
            "question": (
                "Compare sparse triad relay hops against a dense 500m relay chain "
                "for propagating a surface GPS coordinate frame down a water column."
            ),
            "channel_config": asdict(channel),
            "relay_profile": asdict(RELAY_PROFILE),
            "solvers": {
                "sequential": "downstream trilateration from already-estimated upstream anchors",
                "global": "least-squares pose graph over all delivered range constraints with known depth priors",
            },
            "metrics": [
                "bottom_rmse_m",
                "all_rmse_m",
                "covariance trace from normal matrix",
                "rank/conditioning diagnostics",
                "delivered range-edge ratio",
                "drone count",
            ],
            "metric_notes": (
                "success_rate requires every bottom-depth node to be estimated. "
                "bottom_rmse_m is computed over bottom nodes that were estimated, "
                "so failure cases must be interpreted with success_rate and "
                "failed_bottom_nodes_mean."
            ),
        },
        "rows": rows,
        "comparison_summary": _comparison_summary(rows),
        "wall_time_s": time.perf_counter() - t0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nTotal wall time: {artifact['wall_time_s']:.2f}s")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
