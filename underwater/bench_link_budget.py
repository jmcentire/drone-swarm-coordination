# /// script
# dependencies = []
# ///
"""Link-budget and throughput benchmark for underwater swarm comms.

This bench answers the question that the abstract comms spectrum cannot:
whether a proposed distance band has enough acoustic link budget and scheduled
throughput for the messages the swarm wants to exchange.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from acoustic_channel import AcousticChannelConfig, MessageProfile, estimate_rows


SOURCES = [
    {
        "name": "Local OCEANS paper target",
        "detail": (
            "OCEANS26B_0177_MS-2.pdf reports a 28 kHz admin channel, 32-byte "
            "uplink payload, 1 Hz telemetry, viability up to 2 km, and 15 s "
            "retry penalty. It is used here as an admin-channel target, not as "
            "a peer-gossip channel validation."
        ),
    },
    {
        "name": "UnetStack BasicAcousticChannel",
        "url": "https://unetstack.net/handbook/unet-handbook_modems_and_channel_models.html",
        "detail": (
            "Uses Urick-style average transmission loss plus BPSK fading model; "
            "parameters include carrier frequency, bandwidth, spreading, noise "
            "level, salinity, temperature, and water depth."
        ),
    },
    {
        "name": "Lucani, Medard, Stojanovic 2008",
        "url": "https://arxiv.org/abs/0809.0070",
        "detail": (
            "Provides distance/capacity/power framing for underwater acoustic "
            "links with path loss dependent on distance and frequency."
        ),
    },
    {
        "name": "Network-coding hybrid ARQ underwater acoustic sensor network model",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5038722/",
        "detail": (
            "Uses Urick/Thorp path loss, ambient noise, fading, BPSK BER, and "
            "packet error probability from packet length."
        ),
    },
]


DEFAULT_PROFILES = [
    MessageProfile(
        name="oceans_admin_uplink",
        payload_bytes=32,
        rate_hz=1.0,
        fanout=1,
        safety_factor=1.0,
        description="Paper-tracked supervisory telemetry target: one 32-byte state payload per second.",
    ),
    MessageProfile(
        name="peer_contact_report",
        payload_bytes=56,
        rate_hz=0.2,
        fanout=1,
        safety_factor=2.0,
        description="One contact/map report every 5 s with duplicate/metadata allowance.",
    ),
    MessageProfile(
        name="peer_edge_quality",
        payload_bytes=32,
        rate_hz=0.2,
        fanout=1,
        safety_factor=1.5,
        description="Per-neighbor link-quality summary every 5 s.",
    ),
    MessageProfile(
        name="peer_map_bundle",
        payload_bytes=160,
        rate_hz=0.2,
        fanout=1,
        safety_factor=2.0,
        description="Bundled contact reports plus edge-quality data every 5 s.",
    ),
    MessageProfile(
        name="shared_channel_8_drone_map_bundle",
        payload_bytes=160,
        rate_hz=0.2,
        fanout=8,
        safety_factor=2.0,
        description="Eight drones sharing one half-duplex acoustic channel for map bundles.",
    ),
    MessageProfile(
        name="recovery_alert",
        payload_bytes=64,
        rate_hz=1.0 / 15.0,
        fanout=8,
        safety_factor=3.0,
        description="Repeated recovery/rally alert over a 15 s retry window.",
    ),
]


def _parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return values


def _strong_band(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    bands: dict[str, dict[str, float | None]] = {}
    profiles = sorted({r["profile_name"] for r in rows})
    for profile in profiles:
        profile_rows = sorted([r for r in rows if r["profile_name"] == profile], key=lambda r: r["distance_m"])
        strong = [r for r in profile_rows if r["classification"] == "strong"]
        marginal_or_better = [r for r in profile_rows if r["classification"] in {"strong", "marginal"}]
        bands[profile] = {
            "max_strong_distance_m": None if not strong else strong[-1]["distance_m"],
            "max_marginal_or_better_distance_m": None if not marginal_or_better else marginal_or_better[-1]["distance_m"],
            "first_failed_distance_m": next((r["distance_m"] for r in profile_rows if r["classification"] == "failed"), None),
        }
    return bands


def _planning_recommendations(rows: list[dict]) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    for r in sorted(rows, key=lambda x: (x["profile_name"], x["distance_m"])):
        if r["classification"] == "strong":
            action = "normal"
            reason = "link has enough packet delivery and scheduled capacity for this profile"
        elif r["classification"] == "marginal":
            action = "tighten_or_thin"
            reason = (
                "link is usable but should trigger formation tightening, relay insertion, "
                "lower message rate, or summary-only payloads before mission planning assumes it is stable"
            )
        else:
            action = "do_not_plan_direct_exchange"
            reason = (
                "direct exchange is outside modeled delivery/capacity bounds; plan a relay, "
                "rally/tighten, or use delayed/post-mission recovery instead"
            )
        recommendations.append({
            "profile_name": r["profile_name"],
            "distance_m": r["distance_m"],
            "classification": r["classification"],
            "action": action,
            "reason": reason,
            "observables_to_monitor": [
                "packet_delivery_rate",
                "retry_burden",
                "channel_occupancy",
                "snr_db",
                "one_way_delay_s",
            ],
        })
    return recommendations


def _print_table(rows: list[dict]) -> None:
    print("\nLINK BUDGET")
    print("dist  profile                         snr   pdr   occ   retry  goodput  class")
    print("-" * 89)
    for r in sorted(rows, key=lambda x: (x["profile_name"], x["distance_m"])):
        print(
            f"{r['distance_m']:>4.0f}  "
            f"{r['profile_name']:<30.30} "
            f"{r['snr_db']:>5.1f} "
            f"{r['packet_delivery_rate']:>5.3f} "
            f"{r['channel_occupancy']:>5.2f} "
            f"{r['retry_burden']:>6.2f} "
            f"{r['effective_goodput_bps']:>8.0f} "
            f"{r['classification']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--distances",
        type=_parse_float_list,
        default=[50, 100, 180, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000],
        help="comma-separated distances in meters",
    )
    ap.add_argument("--output", default="underwater/bench_link_budget_results.json")
    ap.add_argument("--source-level-db", type=float, default=190.0)
    ap.add_argument("--bandwidth-hz", type=float, default=4096.0)
    ap.add_argument("--carrier-frequency-hz", type=float, default=28000.0)
    ap.add_argument("--noise-psd-db", type=float, default=60.0)
    ap.add_argument("--mac-efficiency", type=float, default=0.35)
    args = ap.parse_args()

    cfg = replace(
        AcousticChannelConfig(),
        source_level_db_re_upa_at_1m=args.source_level_db,
        bandwidth_hz=args.bandwidth_hz,
        carrier_frequency_hz=args.carrier_frequency_hz,
        noise_psd_db_re_upa_per_sqrt_hz=args.noise_psd_db,
        mac_efficiency=args.mac_efficiency,
    )
    distances = sorted(args.distances)
    rows = estimate_rows(distances, DEFAULT_PROFILES, cfg)
    _print_table(rows)

    artifact = {
        "method": {
            "purpose": (
                "Estimate whether modeled acoustic distances have sufficient "
                "packet delivery and scheduled throughput for swarm data needs."
            ),
            "channel_config": asdict(cfg),
            "message_profiles": [asdict(p) for p in DEFAULT_PROFILES],
            "classification": {
                "strong": (
                    "packet_delivery_rate >= strong_pdr, channel_occupancy <= "
                    "strong_occupancy, and retry_burden <= strong_retry_limit"
                ),
                "marginal": (
                    "packet_delivery_rate >= marginal_pdr, channel_occupancy <= "
                    "marginal_occupancy, and retry_burden <= marginal_retry_limit"
                ),
                "failed": "outside marginal bounds",
            },
            "model_limits": [
                "not a site-specific sound-speed profile or bathymetry model",
                "does not model multipath geometry explicitly",
                "does not model Doppler spread explicitly yet",
                "does not allocate a full MAC schedule across multiple links yet",
                "default source level is calibrated so the OCEANS admin target is plausible near 2 km",
            ],
            "sources": SOURCES,
        },
        "rows": rows,
        "strong_band_summary": _strong_band(rows),
        "planning_recommendations": _planning_recommendations(rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nResults written to {output}")


if __name__ == "__main__":
    main()
