# /// script
# dependencies = ["numpy<3"]
# ///
"""Communication-spectrum bench for underwater detection mapping.

This is the operating-envelope companion to bench_detection_mapping.py. The
D1-D7 bench tests named claims; this bench sweeps acoustic neighbor range,
per-exchange report capacity, and neighbor-exchange loss to find where
distributed mapping stops behaving like central log fusion and where it falls
back toward local-only performance.

Output is a compact JSON artifact with one aggregate row per comms setting plus
cliff summaries by loss/budget slice. Raw seed checkpoints are optional and are
kept outside the committed result file.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from bench_detection_mapping import Scenario, run_scenario


BASE_SPEC = Scenario(
    name="SPECTRUM_BASE",
    claim="Distributed mapping degrades gracefully as acoustic exchange becomes range-limited, lossy, or message-budgeted.",
    falsifying="Recall cliffs abruptly without warning, or bounded comms collapses to local-only performance over plausible acoustic settings.",
    mechanism="Nominal detection survey with acoustic range, exchange budget, and exchange loss swept independently.",
    pass_criterion="analysis artifact; see cliff_summary and per-cell metrics",
)


PARAMETER_MODEL = {
    "distance_axis": {
        "field": "comms_range_m",
        "unit": "meters",
        "meaning": (
            "Maximum acoustic neighbor-exchange radius. This is a graph-connectivity "
            "parameter, not a physical modem range claim."
        ),
    },
    "information_capacity_axis": {
        "field": "max_reports_per_exchange",
        "unit": "contact reports per neighbor exchange per 5 s simulation tick",
        "meaning": (
            "Per-link payload budget for sharing mapped contacts. It approximates "
            "bandwidth, packet size, and scheduling limits without assigning bytes "
            "or modem rate."
        ),
    },
    "loss_axis": {
        "field": "report_loss_rate",
        "unit": "probability",
        "meaning": (
            "Independent probability that a neighbor exchange contributes no shared "
            "reports during a tick. It approximates packet loss, missed scheduling "
            "slots, or short acoustic outages."
        ),
    },
    "held_constant_detection_mapping_parameters": {
        "sensor_noise_m": BASE_SPEC.sensor_noise_m,
        "false_contact_rate_per_drone_s": BASE_SPEC.false_contact_rate_per_drone_s,
        "random_drift_m_sqrt_s": BASE_SPEC.random_drift_m_sqrt_s,
        "directed_current_mps": BASE_SPEC.directed_current_mps,
        "shear_current_mps_per_m": BASE_SPEC.shear_current_mps_per_m,
        "duration_s": BASE_SPEC.duration_s,
        "dt_s": BASE_SPEC.dt_s,
    },
    "not_yet_modeled": [
        "bit-level payload corruption",
        "explicit bytes-per-second modem rates",
        "half-duplex MAC scheduling",
        "propagation latency",
        "asymmetric links",
        "frequency-dependent attenuation",
        "multipath",
        "Doppler",
        "ambient-noise-derived SNR or decoding probability",
    ],
    "interpretation": (
        "The sweep is a coordination-envelope test. It is useful for identifying "
        "connectivity and information-capacity cliffs before replacing the abstract "
        "loss/range knobs with a modem- and channel-specific acoustic model."
    ),
}

PAPER_DERIVED_TARGETS = {
    "source": "OCEANS26B_0177_MS-2.pdf",
    "title": "Feasibility of Dual-Layer Acoustic Communication Architectures for Supervisory Control of Heterogeneous AUV Swarms",
    "features_to_track": {
        "dual_layer_architecture": (
            "Long-range human-on-the-loop administrative channel separated from "
            "short-range peer-to-peer swarm coordination."
        ),
        "admin_channel_frequency_hz": 28_000,
        "uplink_payload_bytes": 32,
        "uplink_update_rate_hz": 1,
        "admin_channel_viable_range_m": 2_000,
        "missed_command_retry_delay_s": 15,
        "traffic_asymmetry": (
            "Sparse low-rate downlink commands and denser continuous uplink state."
        ),
        "channel_constraints_called_out": [
            "frequency-dependent attenuation",
            "multipath",
            "Doppler",
            "ambient noise",
            "low bandwidth",
            "approximately 1500 m/s propagation speed",
            "half-duplex acoustic channel access",
            "packet scheduling",
        ],
    },
    "sufficient_claim": (
        "For the paper's admin channel, 32-byte 1 Hz uplink telemetry over a 28 kHz "
        "acoustic channel was reported viable and uncongested up to 2 km in simulation. "
        "That is a useful sufficiency target for supervisory telemetry, not proof that "
        "peer gossip, detection-map exchange, or formation control are sufficient at 2 km."
    ),
    "missing_for_physical_model": [
        "modem data rate or packet airtime",
        "MAC schedule and number of vehicles sharing the half-duplex channel",
        "source level, receive threshold, and SNR-to-packet-error mapping",
        "environmental noise distribution",
        "water depth and sound-speed profile",
        "multipath and Doppler model parameters",
        "range-dependent packet loss or latency curves",
        "topology and mobility assumptions for the peer-coordination layer",
    ],
    "benchmark_use": (
        "Use 2 km/32-byte/1 Hz/15 s as paper-tracked admin-channel targets. Keep this "
        "spectrum bench's range/capacity/loss axes as abstract peer-gossip envelope "
        "parameters until modem-specific packet timing and channel error models are available."
    ),
    "zenodo_check": {
        "checked_queries": [
            "underwater_acoustic_simulation",
            "Feasibility of Dual-Layer Acoustic Communication Architectures",
            "bw1l50n",
            "SeaLink OEM",
            "Brandon Wilson SeaLink",
        ],
        "matching_archive_found": False,
        "notes": (
            "Zenodo did not expose an exact archive for the local paper or the referenced "
            "underwater_acoustic_simulation repository during this run."
        ),
    },
    "external_modeling_sources_to_consider": [
        {
            "name": "UnetStack underwater network simulator handbook",
            "url": "https://unetstack.net/handbook/unet-handbook.html",
            "use": (
                "Practical simulator defaults for protocol and acoustic channels, including "
                "communication/detection/interference ranges and BasicAcousticChannel "
                "parameters such as carrier frequency, bandwidth, spreading, salinity, "
                "noise level, and water depth."
            ),
        },
        {
            "name": "Lucani, Medard, Stojanovic underwater acoustic network channel models",
            "url": "https://arxiv.org/abs/0809.0070",
            "use": (
                "Analytical distance/capacity/power framing for underwater acoustic links; "
                "good source for replacing report-count budgets with capacity-vs-distance "
                "requirements."
            ),
        },
        {
            "name": "Network-coding hybrid ARQ protocol for underwater acoustic sensor networks",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5038722/",
            "use": (
                "Packet loss/error modeling under path loss, ambient noise, and fading; "
                "useful for replacing independent exchange loss with distance/frequency "
                "dependent packet error probability."
            ),
        },
    ],
}

TIMING_NOTE = (
    "wall_time_s is the current invocation elapsed time. When checkpoints are "
    "replayed, per-row wall_time_s values measure checkpoint load/reduce time, "
    "not the original simulation cost. full_run_wall_time_s preserves the original "
    "non-replay local run when provided."
)


def _parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer value")
    return values


def _metric_mean(metrics: dict[str, Any], key: str) -> float:
    value = metrics[key]
    if isinstance(value, list):
        return float(value[0])
    return float(value)


def _ci(metrics: dict[str, Any], key: str) -> list[float]:
    value = metrics[key]
    if isinstance(value, tuple):
        return [float(x) for x in value]
    if isinstance(value, list):
        return [float(x) for x in value]
    x = float(value)
    return [x, x, x]


def _row_from_result(result: dict[str, Any], wall_time_s: float) -> dict[str, Any]:
    scenario = result["scenario"]
    metrics = result["metrics"]
    return {
        "name": scenario["name"],
        "range_m": float(scenario["comms_range_m"]),
        "max_reports_per_exchange": int(scenario["max_reports_per_exchange"]),
        "loss_rate": float(scenario["report_loss_rate"]),
        "n_seeds": int(result["n_seeds"]),
        "passed": bool(result["passed"]),
        "wall_time_s": wall_time_s,
        "recall": _ci(metrics, "recall"),
        "localization_rmse_m": _ci(metrics, "localization_rmse_m"),
        "false_per_true": _ci(metrics, "false_per_true"),
        "local_only_recall": _ci(metrics, "local_only_recall"),
        "single_drone_recall": _ci(metrics, "single_drone_recall"),
        "centralized_recall": _ci(metrics, "centralized_recall"),
        "centralized_recall_delta_vs_distributed": _ci(metrics, "centralized_recall_delta_vs_distributed"),
        "operational_recall_gap": _ci(metrics, "operational_recall_gap"),
        "max_operational_recall_gap": _ci(metrics, "max_operational_recall_gap"),
        "operational_capture_ratio": _ci(metrics, "operational_capture_ratio"),
        "gossip_recall_gain_vs_local": _ci(metrics, "gossip_recall_gain_vs_local"),
        "reports_per_drone": _ci(metrics, "n_reports"),
    }


def _classify(row: dict[str, Any]) -> str:
    recall = row["recall"][0]
    local = row["local_only_recall"][0]
    delta = row["centralized_recall_delta_vs_distributed"][0]
    if recall < max(0.35, local + 0.10):
        return "local_collapse"
    if recall < 0.75:
        return "mission_cliff"
    if delta < 0.02:
        return "central_equivalent"
    if delta < 0.10:
        return "near_central"
    return "bounded_distributed"


def _summarize_cliffs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    groups = sorted({(r["loss_rate"], r["max_reports_per_exchange"]) for r in rows})
    for loss, budget in groups:
        slice_rows = sorted(
            [r for r in rows if r["loss_rate"] == loss and r["max_reports_per_exchange"] == budget],
            key=lambda r: r["range_m"],
        )
        viable = [
            r for r in slice_rows
            if r["recall"][0] >= 0.75 and r["gossip_recall_gain_vs_local"][0] >= 0.50
        ]
        bounded = [r for r in viable if _classify(r) == "bounded_distributed"]
        central_equiv = [
            r for r in slice_rows
            if r["recall"][0] >= 0.90 and r["centralized_recall_delta_vs_distributed"][0] < 0.02
        ]
        best = max(slice_rows, key=lambda r: (r["recall"][0], -r["range_m"]))
        first_viable = min(viable, key=lambda r: r["range_m"]) if viable else None
        first_bounded = min(bounded, key=lambda r: r["range_m"]) if bounded else None
        first_central = min(central_equiv, key=lambda r: r["range_m"]) if central_equiv else None
        cliff = None
        for prev, cur in zip(slice_rows, slice_rows[1:]):
            gain = cur["recall"][0] - prev["recall"][0]
            if gain >= 0.15:
                cliff = {
                    "from_range_m": prev["range_m"],
                    "to_range_m": cur["range_m"],
                    "recall_gain": gain,
                }
                break
        summary.append({
            "loss_rate": loss,
            "max_reports_per_exchange": budget,
            "first_viable_range_m": None if first_viable is None else first_viable["range_m"],
            "first_bounded_distributed_range_m": None if first_bounded is None else first_bounded["range_m"],
            "first_central_equivalent_range_m": None if first_central is None else first_central["range_m"],
            "best_range_m": best["range_m"],
            "best_recall": best["recall"][0],
            "best_central_delta": best["centralized_recall_delta_vs_distributed"][0],
            "cliff": cliff,
        })
    return summary


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\nCOMMS SPECTRUM")
    print("range  budget  loss  recall  central_delta  op_gap  capture  local  class")
    print("-" * 83)
    for row in sorted(rows, key=lambda r: (r["loss_rate"], r["max_reports_per_exchange"], r["range_m"])):
        print(
            f"{row['range_m']:>5.0f}"
            f"  {row['max_reports_per_exchange']:>6}"
            f"  {row['loss_rate']:>4.2f}"
            f"  {row['recall'][0]:>6.3f}"
            f"  {row['centralized_recall_delta_vs_distributed'][0]:>13.3f}"
            f"  {row['operational_recall_gap'][0]:>6.3f}"
            f"  {row['operational_capture_ratio'][0]:>7.3f}"
            f"  {row['local_only_recall'][0]:>5.3f}"
            f"  {_classify(row)}"
        )


def _scenario_name(range_m: float, budget: int, loss: float) -> str:
    loss_pct = int(round(loss * 100))
    range_i = int(round(range_m))
    return f"C_range{range_i:03d}_budget{budget}_loss{loss_pct:02d}"


def _make_spec(range_m: float, budget: int, loss: float) -> Scenario:
    return replace(
        BASE_SPEC,
        name=_scenario_name(range_m, budget, loss),
        comms_range_m=float(range_m),
        max_reports_per_exchange=int(budget),
        report_loss_rate=float(loss),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--ranges", type=_parse_float_list, default=[30.0, 45.0, 60.0, 75.0, 90.0, 120.0, 150.0, 180.0])
    ap.add_argument("--budgets", type=_parse_int_list, default=[1, 2, 4])
    ap.add_argument("--losses", type=_parse_float_list, default=[0.0, 0.3, 0.5, 0.7])
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--output", default="underwater/bench_comms_spectrum_results.json")
    ap.add_argument("--full-run-wall-time-s", type=float, default=None)
    args = ap.parse_args()

    ranges = sorted(args.ranges)
    budgets = sorted(args.budgets)
    losses = sorted(args.losses)
    specs = [_make_spec(r, b, loss) for loss in losses for b in budgets for r in ranges]

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for idx, spec in enumerate(specs, start=1):
        print(
            f"\n[{idx}/{len(specs)}] {spec.name} "
            f"(seeds={args.seeds}, jobs={args.jobs}, range={spec.comms_range_m:g}, "
            f"budget={spec.max_reports_per_exchange}, loss={spec.report_loss_rate:g})",
            flush=True,
        )
        ts = time.perf_counter()
        result = run_scenario(spec, args.seeds, jobs=args.jobs, checkpoint_dir=args.checkpoint_dir)
        dt = time.perf_counter() - ts
        row = _row_from_result(result, dt)
        row["class"] = _classify(row)
        rows.append(row)
        print(
            f"  recall={row['recall'][0]:.3f} central_delta="
            f"{row['centralized_recall_delta_vs_distributed'][0]:.3f} "
            f"op_gap={row['operational_recall_gap'][0]:.3f} "
            f"class={row['class']} done in {dt:.2f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    cliff_summary = _summarize_cliffs(rows)
    _print_table(rows)
    artifact = {
        "method": {
            "base_scenario": asdict(BASE_SPEC),
            "ranges_m": ranges,
            "max_reports_per_exchange": budgets,
            "loss_rates": losses,
            "n_cells": len(rows),
            "n_seeds_per_cell": args.seeds,
            "parameter_model": PARAMETER_MODEL,
            "paper_derived_targets": PAPER_DERIVED_TARGETS,
            "classification": {
                "central_equivalent": "recall >= 0.90 and centralized_recall_delta_vs_distributed < 0.02",
                "bounded_distributed": "recall >= 0.75 and central delta >= 0.10",
                "near_central": "recall >= 0.75 and 0.02 <= central delta < 0.10",
                "mission_cliff": "recall < 0.75 but remains above local collapse",
                "local_collapse": "recall < max(0.35, local_only_recall + 0.10)",
            },
        },
        "rows": rows,
        "cliff_summary": cliff_summary,
        "wall_time_s": elapsed,
        "full_run_wall_time_s": args.full_run_wall_time_s,
        "timing_note": TIMING_NOTE,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nTotal wall time: {elapsed:.2f}s")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
