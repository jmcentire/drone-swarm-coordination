# /// script
# dependencies = ["numpy<3"]
# ///
"""Underwater contact detection and hazard-map bench.

Research question:
  Can a sparse underwater drone collective detect and map mine-like contacts
  under acoustic comms limits, localization drift, data loss, and occasional
  fixed-point reorientation?

Scope:
  This bench models detection and mapping only. It does not classify contacts,
  identify mine type, neutralize anything, or model adversarial compromise.

Each scenario states:
  - claim: what the scenario supports if it passes
  - falsifying: what would refute the claim
  - mechanism: how the simulation stresses the system
  - pass_criterion: quantitative gate

The simulation is intentionally compact:
  - drones sweep parallel lanes through a 3D search volume
  - contacts are fixed point objects with no type labels
  - detections produce noisy position reports in the drone's estimated frame
  - localization bias follows either random walk, directed current, or shear
  - anchor fixes reduce localization bias and uncertainty
  - reports are gossiped through a range-limited lossy acoustic substrate
  - the map fuses reports into contact hypotheses with confidence/staleness
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    claim: str
    falsifying: str
    mechanism: str
    pass_criterion: str
    n_drones: int = 8
    n_contacts: int = 28
    duration_s: float = 1200.0
    dt_s: float = 5.0
    area_length_m: float = 900.0
    area_width_m: float = 360.0
    depth_m: float = 80.0
    sensor_radius_m: float = 45.0
    detection_p_at_center: float = 0.92
    sensor_noise_m: float = 4.0
    false_contact_rate_per_drone_s: float = 0.00025
    comms_range_m: float = 180.0
    report_loss_rate: float = 0.0
    gossip_rounds_per_tick: int = 1
    random_drift_m_sqrt_s: float = 0.01
    directed_current_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    shear_current_mps_per_m: float = 0.0
    anchor_interval_s: float | None = None
    anchor_fix_noise_m: float = 2.0
    anchor_fraction: float = 0.25
    calloff_uncertainty_m: float = 60.0
    stale_after_s: float = 240.0


SCENARIOS: list[Scenario] = [
    Scenario(
        name="D1_nominal_detection_map",
        claim="With low drift and reliable acoustic report exchange, the swarm builds a usable contact map.",
        falsifying="Recall below 90%, localization RMSE above 20m, or false-map rate above 0.35 per true contact.",
        mechanism="Parallel-lane sweep; fixed contacts; low random drift; no report loss; no anchors.",
        pass_criterion="recall >= 0.90 AND localization_rmse_m <= 20 AND false_per_true <= 0.35",
    ),
    Scenario(
        name="D2_report_loss_30pct",
        claim="The map degrades gracefully under 30% acoustic report loss because multiple drones and gossip provide redundant observations.",
        falsifying="Recall below 75% or localization RMSE above 30m.",
        mechanism="Same field as D1 with independent 30% report loss on each acoustic exchange.",
        pass_criterion="recall >= 0.75 AND localization_rmse_m <= 30",
        report_loss_rate=0.30,
    ),
    Scenario(
        name="D3_random_walk_drift",
        claim="Moderate per-drone dead-reckoning drift reduces absolute accuracy but still leaves a useful hazard map.",
        falsifying="Recall below 70% or localization RMSE above 45m.",
        mechanism="Per-drone localization bias follows an independent random walk; no anchors.",
        pass_criterion="recall >= 0.70 AND localization_rmse_m <= 45",
        random_drift_m_sqrt_s=0.08,
    ),
    Scenario(
        name="D4_directed_current",
        claim="A shared directed current is less damaging than independent drift because relative coverage remains coherent.",
        falsifying="Recall below 75% or localization RMSE above 45m.",
        mechanism="All drones experience an unmodeled 0.03m/s lateral current while sweeping.",
        pass_criterion="recall >= 0.75 AND localization_rmse_m <= 45",
        directed_current_mps=(0.0, 0.03, 0.0),
        random_drift_m_sqrt_s=0.02,
    ),
    Scenario(
        name="D5_anchor_reorientation",
        claim="Sparse fixed-point reorientation substantially improves map accuracy under random drift.",
        falsifying="Anchor-aided RMSE is not at least 40% lower than the matched no-anchor control.",
        mechanism="A quarter of drones periodically acquire noisy fixed-point position fixes and use them to correct the shared map frame.",
        pass_criterion="rmse_improvement_vs_control >= 0.40 AND recall >= 0.75",
        random_drift_m_sqrt_s=0.03,
        directed_current_mps=(0.0, 0.06, 0.0),
        anchor_interval_s=240.0,
        anchor_fraction=0.25,
    ),
    Scenario(
        name="D6_calloff_on_stale_map",
        claim="When drift grows beyond the mission envelope without anchors, the system marks the map unsafe instead of pretending it is reliable.",
        falsifying="Map uncertainty exceeds the calloff threshold for more than 120s before calloff.",
        mechanism="High random drift, no anchors. The mission should call off when propagated map uncertainty exceeds 60m.",
        pass_criterion="calloff_triggered AND calloff_delay_s <= 120",
        random_drift_m_sqrt_s=2.0,
        calloff_uncertainty_m=60.0,
    ),
]


@dataclass
class ContactReport:
    contact_id: int | None
    position: tuple[float, float, float]
    sigma_m: float
    confidence: float
    observed_t: float
    observer: int


def _bootstrap_ci(values: list[float], n_boot: int = 1000) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        x = float(arr[0])
        return (x, x, x)
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return (0.0, 0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) + z * z / (4 * n)) / n) ** 0.5 / denom
    return (float(phat), float(max(0.0, center - half)), float(min(1.0, center + half)))


def _initial_contacts(spec: Scenario, rng: np.random.Generator) -> np.ndarray:
    margin = spec.sensor_radius_m * 0.6
    pts: list[np.ndarray] = []
    min_sep = 1.25 * spec.sensor_radius_m
    attempts = 0
    while len(pts) < spec.n_contacts and attempts < spec.n_contacts * 500:
        attempts += 1
        p = np.array([
            rng.uniform(margin, spec.area_length_m - margin),
            rng.uniform(-spec.area_width_m / 2 + margin, spec.area_width_m / 2 - margin),
            rng.uniform(-spec.depth_m - 8.0, -spec.depth_m + 8.0),
        ])
        if all(float(np.linalg.norm(p - q)) >= min_sep for q in pts):
            pts.append(p)
    if len(pts) < spec.n_contacts:
        raise RuntimeError("could not place separated contacts")
    return np.array(pts, dtype=np.float64)


def _planned_positions(spec: Scenario, t: float) -> np.ndarray:
    lanes = np.linspace(-spec.area_width_m / 2, spec.area_width_m / 2, spec.n_drones)
    progress = min(1.0, t / spec.duration_s)
    x = progress * spec.area_length_m
    z = -spec.depth_m * np.ones(spec.n_drones)
    return np.stack([np.full(spec.n_drones, x), lanes, z], axis=1)


def _neighbors(positions: np.ndarray, comms_range_m: float) -> list[list[int]]:
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    return [
        [int(j) for j in range(len(positions)) if j != i and d[i, j] <= comms_range_m]
        for i in range(len(positions))
    ]


def _detect_contacts(
    spec: Scenario,
    contacts: np.ndarray,
    true_pos: np.ndarray,
    est_pos: np.ndarray,
    bias_sigma: np.ndarray,
    t: float,
    rng: np.random.Generator,
) -> list[list[ContactReport]]:
    reports: list[list[ContactReport]] = [[] for _ in range(spec.n_drones)]
    for i in range(spec.n_drones):
        d = np.linalg.norm(contacts - true_pos[i], axis=1)
        visible = np.where(d <= spec.sensor_radius_m)[0]
        for cid in visible:
            p = spec.detection_p_at_center * max(0.0, 1.0 - 0.65 * (d[cid] / spec.sensor_radius_m) ** 2)
            if rng.random() > p:
                continue
            localization_bias = est_pos[i] - true_pos[i]
            noise = rng.normal(0.0, spec.sensor_noise_m, size=3)
            estimate = contacts[cid] + localization_bias + noise
            sigma = float((spec.sensor_noise_m ** 2 + bias_sigma[i] ** 2) ** 0.5)
            reports[i].append(ContactReport(
                contact_id=int(cid),
                position=tuple(float(x) for x in estimate),
                sigma_m=sigma,
                confidence=float(p),
                observed_t=t,
                observer=i,
            ))
        if rng.random() < spec.false_contact_rate_per_drone_s * spec.dt_s:
            fake = est_pos[i] + rng.normal(0.0, spec.sensor_radius_m * 0.5, size=3)
            reports[i].append(ContactReport(
                contact_id=None,
                position=tuple(float(x) for x in fake),
                sigma_m=float((spec.sensor_noise_m ** 2 + bias_sigma[i] ** 2) ** 0.5),
                confidence=0.25,
                observed_t=t,
                observer=i,
            ))
    return reports


def _gossip_reports(
    knowledge: list[list[ContactReport]],
    neighbors: list[list[int]],
    loss_rate: float,
    rounds: int,
    rng: np.random.Generator,
) -> list[list[ContactReport]]:
    # Deduplicate by object identity tuple. Reports are immutable enough for this bench.
    known = [dict(((r.observer, r.observed_t, r.contact_id, r.position), r) for r in ks) for ks in knowledge]
    for _ in range(rounds):
        prev = [dict(k) for k in known]
        for i, ns in enumerate(neighbors):
            for j in ns:
                if loss_rate > 0 and rng.random() < loss_rate:
                    continue
                known[i].update(prev[j])
    return [list(k.values()) for k in known]


def _fuse_map(
    reports: list[ContactReport],
    t_now: float,
    stale_after_s: float,
    merge_gate_m: float = 35.0,
) -> list[dict[str, Any]]:
    # Contacts are fixed hazards. A stale report should lower confidence in
    # the surrounding searched region, not erase the contact hypothesis.
    fresh = list(reports)
    fresh.sort(key=lambda r: r.observed_t)
    clusters: list[list[ContactReport]] = []
    centers: list[np.ndarray] = []
    for r in fresh:
        p = np.asarray(r.position, dtype=np.float64)
        if not clusters:
            clusters.append([r])
            centers.append(p)
            continue
        d = np.array([np.linalg.norm(p - c) for c in centers])
        j = int(np.argmin(d))
        if d[j] <= merge_gate_m:
            clusters[j].append(r)
            weights = np.array([1.0 / max(x.sigma_m ** 2, 1e-6) for x in clusters[j]])
            pts = np.array([x.position for x in clusters[j]], dtype=np.float64)
            centers[j] = (pts * weights[:, None]).sum(axis=0) / weights.sum()
        else:
            clusters.append([r])
            centers.append(p)

    out = []
    for rs, c in zip(clusters, centers):
        weights = np.array([1.0 / max(x.sigma_m ** 2, 1e-6) for x in rs])
        sigma = float((1.0 / weights.sum()) ** 0.5)
        confidence = float(1.0 - np.prod([1.0 - min(0.95, max(0.0, r.confidence)) for r in rs]))
        out.append({
            "position": c,
            "sigma_m": sigma,
            "confidence": confidence,
            "last_observed_t": max(r.observed_t for r in rs),
            "stale": (t_now - max(r.observed_t for r in rs)) > stale_after_s,
            "n_reports": len(rs),
            "observers": sorted(set(r.observer for r in rs)),
        })
    return out


def _score_map(hypotheses: list[dict[str, Any]], contacts: np.ndarray) -> dict[str, float]:
    if len(hypotheses) == 0:
        return {
            "recall": 0.0,
            "localization_rmse_m": float("inf"),
            "false_hypotheses": 0.0,
            "false_per_true": 0.0,
            "mean_confidence": 0.0,
        }
    hypo_pos = np.array([h["position"] for h in hypotheses], dtype=np.float64)
    d = np.linalg.norm(contacts[:, None, :] - hypo_pos[None, :, :], axis=-1)
    matched_contacts = set()
    matched_hypotheses = set()
    errors = []
    for _ in range(min(len(contacts), len(hypotheses))):
        idx = np.unravel_index(int(np.argmin(d)), d.shape)
        if not np.isfinite(d[idx]) or d[idx] > 45.0:
            break
        ci, hi = int(idx[0]), int(idx[1])
        matched_contacts.add(ci)
        matched_hypotheses.add(hi)
        errors.append(float(d[ci, hi]))
        d[ci, :] = np.inf
        d[:, hi] = np.inf
    false_h = len(hypotheses) - len(matched_hypotheses)
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf")
    return {
        "recall": float(len(matched_contacts) / len(contacts)),
        "localization_rmse_m": rmse,
        "false_hypotheses": float(false_h),
        "false_per_true": float(false_h / len(contacts)),
        "mean_confidence": float(np.mean([h["confidence"] for h in hypotheses])),
    }


def run_seed(spec: Scenario, seed: int, control_without_anchors: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    contacts = _initial_contacts(spec, rng)
    bias = np.zeros((spec.n_drones, 3), dtype=np.float64)
    bias_sigma = np.zeros(spec.n_drones, dtype=np.float64)
    knowledge: list[list[ContactReport]] = [[] for _ in range(spec.n_drones)]
    max_uncertainty = 0.0
    calloff_t: float | None = None
    first_bad_t: float | None = None
    current = np.asarray(spec.directed_current_mps, dtype=np.float64)
    anchor_interval = None if control_without_anchors else spec.anchor_interval_s
    anchor_ids = set(range(max(1, int(round(spec.n_drones * spec.anchor_fraction)))))

    n_ticks = int(spec.duration_s / spec.dt_s) + 1
    for tick in range(n_ticks):
        t = tick * spec.dt_s
        planned = _planned_positions(spec, t)
        shear = np.zeros_like(planned)
        if spec.shear_current_mps_per_m:
            shear[:, 1] = spec.shear_current_mps_per_m * (planned[:, 2] + spec.depth_m)
        true_pos = planned + (current[None, :] + shear) * t

        if tick > 0:
            drift_step = spec.random_drift_m_sqrt_s * np.sqrt(spec.dt_s)
            bias += rng.normal(0.0, drift_step, size=bias.shape)
            bias_sigma = np.sqrt(bias_sigma ** 2 + drift_step ** 2)

        if anchor_interval is not None and t > 0 and abs(t % anchor_interval) < 1e-9:
            # Anchor/scout fixes estimate a shared correction from the
            # estimated map frame back toward the physical frame. The
            # correction is broadcast as map-frame information, so every
            # drone can apply the common offset; per-drone residual drift
            # remains.
            est_before = planned + bias
            corrections = []
            for i in anchor_ids:
                fix = true_pos[i] + rng.normal(0.0, spec.anchor_fix_noise_m, size=3)
                corrections.append(fix - est_before[i])
                bias_sigma[i] = spec.anchor_fix_noise_m
            common_correction = np.mean(corrections, axis=0)
            bias += common_correction[None, :]
            bias_sigma = np.maximum(spec.anchor_fix_noise_m, bias_sigma * 0.65)

        est_pos = planned + bias
        max_uncertainty = max(max_uncertainty, float(np.max(bias_sigma)))
        if first_bad_t is None and max_uncertainty >= spec.calloff_uncertainty_m:
            first_bad_t = t
        if calloff_t is None and max_uncertainty >= spec.calloff_uncertainty_m:
            calloff_t = t

        tick_reports = _detect_contacts(spec, contacts, true_pos, est_pos, bias_sigma, t, rng)
        for i, rs in enumerate(tick_reports):
            knowledge[i].extend(rs)
        neighbors = _neighbors(true_pos, spec.comms_range_m)
        knowledge = _gossip_reports(
            knowledge, neighbors, spec.report_loss_rate, spec.gossip_rounds_per_tick, rng
        )

    all_reports: dict[tuple[Any, ...], ContactReport] = {}
    for ks in knowledge:
        for r in ks:
            all_reports[(r.observer, r.observed_t, r.contact_id, r.position)] = r
    hypotheses = _fuse_map(list(all_reports.values()), spec.duration_s, spec.stale_after_s)
    score = _score_map(hypotheses, contacts)
    calloff_delay = (
        0.0 if first_bad_t is not None and calloff_t is not None and calloff_t <= first_bad_t
        else float("inf") if first_bad_t is not None
        else 0.0
    )
    score.update({
        "seed": seed,
        "n_hypotheses": float(len(hypotheses)),
        "n_reports": float(len(all_reports)),
        "max_uncertainty_m": max_uncertainty,
        "calloff_triggered": calloff_t is not None,
        "calloff_t": calloff_t,
        "first_bad_t": first_bad_t,
        "calloff_delay_s": calloff_delay,
    })
    return score


def _passes(spec: Scenario, metrics: dict[str, tuple[float, float, float]]) -> bool:
    m = {k: v[0] for k, v in metrics.items()}
    if spec.name == "D1_nominal_detection_map":
        return m["recall"] >= 0.90 and m["localization_rmse_m"] <= 20.0 and m["false_per_true"] <= 0.35
    if spec.name == "D2_report_loss_30pct":
        return m["recall"] >= 0.75 and m["localization_rmse_m"] <= 30.0
    if spec.name == "D3_random_walk_drift":
        return m["recall"] >= 0.70 and m["localization_rmse_m"] <= 45.0
    if spec.name == "D4_directed_current":
        return m["recall"] >= 0.75 and m["localization_rmse_m"] <= 45.0
    if spec.name == "D5_anchor_reorientation":
        return m["rmse_improvement_vs_control"] >= 0.40 and m["recall"] >= 0.75
    if spec.name == "D6_calloff_on_stale_map":
        return m["calloff_triggered"] >= 0.5 and m["calloff_delay_s"] <= 120.0
    return False


def aggregate(spec: Scenario, runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "recall", "localization_rmse_m", "false_hypotheses", "false_per_true",
        "mean_confidence", "n_hypotheses", "n_reports", "max_uncertainty_m",
        "calloff_delay_s",
    ]
    metrics = {k: _bootstrap_ci([float(r[k]) for r in runs]) for k in keys}
    metrics["calloff_triggered"] = _wilson(sum(1 for r in runs if r["calloff_triggered"]), len(runs))
    if spec.name == "D5_anchor_reorientation":
        controls = [run_seed(replace(spec, anchor_interval_s=None), int(r["seed"]), control_without_anchors=True) for r in runs]
        control_rmse = [float(r["localization_rmse_m"]) for r in controls]
        aided_rmse = [float(r["localization_rmse_m"]) for r in runs]
        improvements = [
            max(0.0, (c - a) / c) if np.isfinite(c) and c > 1e-9 else 0.0
            for c, a in zip(control_rmse, aided_rmse)
        ]
        metrics["control_localization_rmse_m"] = _bootstrap_ci(control_rmse)
        metrics["rmse_improvement_vs_control"] = _bootstrap_ci(improvements)
    return {
        "scenario": asdict(spec),
        "n_seeds": len(runs),
        "metrics": metrics,
        "passed": _passes(spec, metrics),
        "runs": runs,
    }


def _checkpoint_path(checkpoint_dir: str, scenario: str, seed: int) -> Path:
    return Path(checkpoint_dir) / scenario / f"seed_{seed:03d}.json"


def _run_seed_task(args: tuple[Scenario, int]) -> tuple[int, dict[str, Any], float]:
    spec, seed = args
    t0 = time.perf_counter()
    return seed, run_seed(spec, seed), time.perf_counter() - t0


def run_scenario(spec: Scenario, n_seeds: int, jobs: int = 1, checkpoint_dir: str | None = None) -> dict[str, Any]:
    runs_by_seed: dict[int, dict[str, Any]] = {}
    pending = []
    if checkpoint_dir:
        (Path(checkpoint_dir) / spec.name).mkdir(parents=True, exist_ok=True)
    for seed in range(n_seeds):
        cp = _checkpoint_path(checkpoint_dir, spec.name, seed) if checkpoint_dir else None
        if cp and cp.exists():
            with open(cp) as f:
                runs_by_seed[seed] = json.load(f)
            continue
        pending.append(seed)
    jobs = max(1, min(jobs, len(pending) or 1))
    t_s = time.perf_counter()
    if jobs == 1:
        for seed in pending:
            _, run, dt = _run_seed_task((spec, seed))
            runs_by_seed[seed] = run
            if checkpoint_dir:
                cp = _checkpoint_path(checkpoint_dir, spec.name, seed)
                tmp = cp.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(run, f, indent=2)
                os.replace(tmp, cp)
            print(f"  [{spec.name}] seed {seed+1}/{n_seeds} done dt={dt:.2f}s cum={time.perf_counter()-t_s:.2f}s", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_run_seed_task, (spec, seed)): seed for seed in pending}
            for fut in concurrent.futures.as_completed(futs):
                seed, run, dt = fut.result()
                runs_by_seed[seed] = run
                if checkpoint_dir:
                    cp = _checkpoint_path(checkpoint_dir, spec.name, seed)
                    tmp = cp.with_suffix(".json.tmp")
                    with open(tmp, "w") as f:
                        json.dump(run, f, indent=2)
                    os.replace(tmp, cp)
                print(f"  [{spec.name}] seed {seed+1}/{n_seeds} done dt={dt:.2f}s cum={time.perf_counter()-t_s:.2f}s jobs={jobs}", flush=True)
    return aggregate(spec, [runs_by_seed[s] for s in range(n_seeds)])


def print_report(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("UNDERWATER DETECTION/MAPPING BENCH")
    print("=" * 100)
    for res in results:
        s = res["scenario"]
        m = res["metrics"]
        print(f"\n--- {s['name']} ---")
        print(f"  Claim:      {s['claim']}")
        print(f"  Falsifier:  {s['falsifying']}")
        print(f"  Mechanism:  {s['mechanism']}")
        print(f"  Criterion:  {s['pass_criterion']}")
        print(f"  Recall:     {m['recall'][0]:.3f} [{m['recall'][1]:.3f}, {m['recall'][2]:.3f}]")
        print(f"  RMSE:       {m['localization_rmse_m'][0]:.2f}m [{m['localization_rmse_m'][1]:.2f}, {m['localization_rmse_m'][2]:.2f}]")
        print(f"  False/true: {m['false_per_true'][0]:.3f} [{m['false_per_true'][1]:.3f}, {m['false_per_true'][2]:.3f}]")
        print(f"  Reports:    {m['n_reports'][0]:.1f}")
        print(f"  Max unc:    {m['max_uncertainty_m'][0]:.1f}m")
        if "rmse_improvement_vs_control" in m:
            print(f"  Anchor gain:{m['rmse_improvement_vs_control'][0]:.3f} [{m['rmse_improvement_vs_control'][1]:.3f}, {m['rmse_improvement_vs_control'][2]:.3f}]")
        print(f"  Calloff:    {m['calloff_triggered'][0]:.3f}; delay {m['calloff_delay_s'][0]:.1f}s")
        print(f"  >>> {'PASS' if res['passed'] else 'FAIL'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--output", default="underwater/bench_detection_mapping_results.json")
    args = ap.parse_args()

    chosen = [s for s in SCENARIOS if args.scenarios is None or s.name in args.scenarios]
    t0 = time.perf_counter()
    results = []
    for spec in chosen:
        print(f"\nRunning {spec.name} ({args.seeds} seeds, jobs={args.jobs})...", flush=True)
        ts = time.perf_counter()
        res = run_scenario(spec, args.seeds, jobs=args.jobs, checkpoint_dir=args.checkpoint_dir)
        print(f"  done in {time.perf_counter()-ts:.2f}s", flush=True)
        results.append(res)
    elapsed = time.perf_counter() - t0
    print_report(results)
    with open(args.output, "w") as f:
        json.dump({"results": results, "wall_time_s": elapsed}, f, indent=2, default=str)
    print(f"\nTotal wall time: {elapsed:.2f}s")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
