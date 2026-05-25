# /// script
# dependencies = []
# ///
"""Acoustic link-budget helpers for underwater swarm benchmarks.

This module gives the swarm benches a physical accounting layer between
"neighbor within range" and "information arrived." It is deliberately modest:
the defaults are calibrated engineering assumptions, not a modem datasheet or a
site-specific propagation model.

Model structure:
  distance + carrier frequency -> Urick/Thorp transmission loss
  source level - loss - noise -> SNR
  SNR + packet bits -> packet delivery estimate
  channel rate + packet delivery -> effective throughput and retry burden

Use this to decide whether a scenario is in a strong, marginal, or failed link
band before interpreting distributed coordination results.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AcousticChannelConfig:
    """Parameters for a compact underwater acoustic link-budget model."""

    name: str = "calibrated_28khz_admin_reference"
    carrier_frequency_hz: float = 28_000.0
    bandwidth_hz: float = 4_096.0
    spreading_factor: float = 1.5
    source_level_db_re_upa_at_1m: float = 190.0
    noise_psd_db_re_upa_per_sqrt_hz: float = 60.0
    directivity_index_db: float = 0.0
    coding_gain_db: float = 3.0
    implementation_loss_db: float = 3.0
    packet_overhead_bytes: int = 16
    mac_efficiency: float = 0.35
    sound_speed_mps: float = 1_500.0
    strong_pdr: float = 0.90
    marginal_pdr: float = 0.50
    strong_occupancy: float = 0.40
    marginal_occupancy: float = 0.80
    strong_retry_limit: float = 2.0
    marginal_retry_limit: float = 5.0


@dataclass(frozen=True)
class MessageProfile:
    """Application-layer traffic demand for one link or shared channel."""

    name: str
    payload_bytes: int
    rate_hz: float
    description: str
    fanout: int = 1
    safety_factor: float = 1.0

    @property
    def demand_bps(self) -> float:
        return float(self.payload_bytes * 8) * self.rate_hz * self.fanout * self.safety_factor


@dataclass(frozen=True)
class LinkEstimate:
    """Computed link quality and capacity for one distance/profile pair."""

    distance_m: float
    profile_name: str
    payload_bytes: int
    rate_hz: float
    fanout: int
    application_demand_bps: float
    gross_capacity_bps: float
    scheduled_capacity_bps: float
    effective_goodput_bps: float
    channel_occupancy: float
    one_way_delay_s: float
    retry_burden: float
    packet_delivery_rate: float
    packet_error_rate: float
    bit_error_rate: float
    snr_db: float
    transmission_loss_db: float
    absorption_db_per_km: float
    classification: str


def thorp_absorption_db_per_km(frequency_hz: float) -> float:
    """Return Thorp absorption in dB/km for frequency in Hz.

    The common empirical form expects kHz.
    """

    f = max(frequency_hz / 1_000.0, 1e-9)
    f2 = f * f
    return float(0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4_100.0 + f2) + 2.75e-4 * f2 + 0.003)


def transmission_loss_db(distance_m: float, cfg: AcousticChannelConfig) -> float:
    """Compute spreading + absorption loss in dB.

    Uses practical spreading k=1.5 by default: TL = 10 k log10(r) + alpha r_km.
    """

    r_m = max(float(distance_m), 1.0)
    r_km = r_m / 1_000.0
    spreading = 10.0 * cfg.spreading_factor * math.log10(r_m)
    absorption = thorp_absorption_db_per_km(cfg.carrier_frequency_hz) * r_km
    return float(spreading + absorption)


def noise_level_db(cfg: AcousticChannelConfig) -> float:
    """Noise over the modeled bandwidth in dB re uPa."""

    return float(cfg.noise_psd_db_re_upa_per_sqrt_hz + 10.0 * math.log10(max(cfg.bandwidth_hz, 1.0)))


def snr_db(distance_m: float, cfg: AcousticChannelConfig) -> float:
    """Estimate receive SNR in dB."""

    return float(
        cfg.source_level_db_re_upa_at_1m
        - transmission_loss_db(distance_m, cfg)
        - noise_level_db(cfg)
        + cfg.directivity_index_db
        + cfg.coding_gain_db
        - cfg.implementation_loss_db
    )


def bpsk_rayleigh_bit_error_rate(snr_db_value: float) -> float:
    """Average BPSK BER under Rayleigh fading.

    This is intentionally conservative compared with a tuned modem link. It is
    useful as a monotonic loss curve for benchmark stress.
    """

    gamma = 10.0 ** (snr_db_value / 10.0)
    if gamma <= 0.0:
        return 0.5
    ber = 0.5 * (1.0 - math.sqrt(gamma / (1.0 + gamma)))
    return float(min(0.5, max(0.0, ber)))


def packet_delivery_rate(distance_m: float, payload_bytes: int, cfg: AcousticChannelConfig) -> tuple[float, float]:
    """Return (packet_delivery_rate, bit_error_rate)."""

    bits = max(1, (payload_bytes + cfg.packet_overhead_bytes) * 8)
    ber = bpsk_rayleigh_bit_error_rate(snr_db(distance_m, cfg))
    pdr = (1.0 - ber) ** bits
    return float(min(1.0, max(0.0, pdr))), ber


def gross_capacity_bps(distance_m: float, cfg: AcousticChannelConfig) -> float:
    """Shannon-style capacity proxy over the modeled bandwidth."""

    gamma = 10.0 ** (snr_db(distance_m, cfg) / 10.0)
    return float(cfg.bandwidth_hz * math.log2(1.0 + max(0.0, gamma)))


def estimate_link(distance_m: float, profile: MessageProfile, cfg: AcousticChannelConfig) -> LinkEstimate:
    """Estimate link health and throughput for a message profile."""

    pdr, ber = packet_delivery_rate(distance_m, profile.payload_bytes, cfg)
    gross = gross_capacity_bps(distance_m, cfg)
    scheduled = gross * cfg.mac_efficiency
    demand = profile.demand_bps
    effective = scheduled * pdr
    occupancy = demand / scheduled if scheduled > 0.0 else math.inf
    retry = 1.0 / max(pdr, 1e-9)
    delay = distance_m / cfg.sound_speed_mps
    loss = transmission_loss_db(distance_m, cfg)
    absorption = thorp_absorption_db_per_km(cfg.carrier_frequency_hz)
    cls = classify_link(pdr, occupancy, retry, cfg)
    return LinkEstimate(
        distance_m=float(distance_m),
        profile_name=profile.name,
        payload_bytes=int(profile.payload_bytes),
        rate_hz=float(profile.rate_hz),
        fanout=int(profile.fanout),
        application_demand_bps=demand,
        gross_capacity_bps=gross,
        scheduled_capacity_bps=scheduled,
        effective_goodput_bps=effective,
        channel_occupancy=occupancy,
        one_way_delay_s=delay,
        retry_burden=retry,
        packet_delivery_rate=pdr,
        packet_error_rate=1.0 - pdr,
        bit_error_rate=ber,
        snr_db=snr_db(distance_m, cfg),
        transmission_loss_db=loss,
        absorption_db_per_km=absorption,
        classification=cls,
    )


def classify_link(pdr: float, occupancy: float, retry_burden: float, cfg: AcousticChannelConfig) -> str:
    """Classify whether a link is strong enough for planning assumptions."""

    if (
        pdr >= cfg.strong_pdr
        and occupancy <= cfg.strong_occupancy
        and retry_burden <= cfg.strong_retry_limit
    ):
        return "strong"
    if (
        pdr >= cfg.marginal_pdr
        and occupancy <= cfg.marginal_occupancy
        and retry_burden <= cfg.marginal_retry_limit
    ):
        return "marginal"
    return "failed"


def estimate_rows(
    distances_m: list[float],
    profiles: list[MessageProfile],
    cfg: AcousticChannelConfig,
) -> list[dict]:
    """Return JSON-ready rows for all distance/profile pairs."""

    rows = []
    for distance_m in distances_m:
        for profile in profiles:
            rows.append(asdict(estimate_link(distance_m, profile, cfg)))
    return rows
