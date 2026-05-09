# Phase B1 — Envelope Ranges for Bench Channel Parameters

**Purpose.** This document grounds the bench's channel-parameter sweeps in cited
real-world measurements, so the loss-rate, latency, mission-duration,
blackout-duration, asymmetry, and replan-cadence choices that flow into Phase C
bench design are defensible. Where the open literature is silent, the gap is
called out; ranges are not invented to fill silence.

**Mission classes.** The bench targets a small number of comms regimes:
**UAV-mesh** (802.11-class peer-to-peer between airborne nodes), **RF-LOS**
(narrow-band tactical VHF/UHF, line-of-sight), **RF-NLOS** (same, with terrain
or urban shadowing), **satellite-uplink** (Iridium / Inmarsat / BGAN),
**acoustic-underwater** (AUV / glider modems), and **mmWave** (5G NR / 60 GHz)
as a stress class. These map onto the architectures Phase C will benchmark.

---

## Section 1 — Parameter ranges from the literature

### 1.1 Per-broadcast packet loss rate

| Mission class | Low | Typical | High | Source |
|---|---|---|---|---|
| 802.11b mesh, urban rooftop | ~0% (a few links) | broad mid-range; 30-50% common | ~100% (a few links) | Aguayo, Bicket, Biswas, Judd, Morris (2004), *Link-level Measurements from an 802.11b Mesh Network*, ACM SIGCOMM CCR — finding: distribution of per-link loss is **uniform across 0-100%**; there is no clean "good/bad" threshold; intermediate-loss links dominate and are attributed to multipath fading rather than range. |
| 802.11s UAV mesh, urban (sim) | ~7.7% (300 nodes) | — | ~14.3% (900 nodes) | Reported PDR of 92.3% at 300 nodes / 85.7% at 900 nodes for an 802.11s-enabled UAV swarm in urban simulation (loss = 1 - PDR). Treat as a soft data point: simulation, not field. |
| 802.11b multi-hop end-to-end | 0% | varies; routes degrade as hop count grows | ~10% of node pairs find no working route | Bicket, Aguayo, Biswas, Morris (2005), *Architecture and Evaluation of an Unplanned 802.11b Mesh Network*, MobiCom — ~10% of pairs failed to find any working multi-hop route in the Roofnet trace. |
| Acoustic underwater (AUV experimental) | varies with range | 30-50% packet loss is realistic at non-trivial ranges | 60%+ in shadow zones / long range | Open-literature consensus from underwater-acoustic survey work (Stojanovic & Preisig 2009 *IEEE Comms Mag*; Sozer, Stojanovic, Proakis 2000 *IEEE J. Ocean. Eng.*); experimental AUV networks have reported per-link loss in the 32.8-61.5% band depending on range and environment. |
| Iridium (legacy SBD / voice) | — | dominated by latency, not loss; "five-nines"-class availability not formally claimed | call drops on failed inter-satellite handoff | DTIC ADA464192 (McMahon & Rathburn, *Measuring Latency in Iridium Satellite Constellation Data Services*); Iridium constellation handoff every ~50 s (spotbeam) and ~7 min (satellite). |
| Inmarsat BGAN | — | "99.9% network availability" claim (no quantitative loss SLA in public docs) | rare deep fades | Inmarsat BGAN brochure / developer pages cite 99.9% satellite-and-ground availability; **no public per-packet loss SLA was located**. |
| Tactical VHF/UHF | low when SINR > ~6-23 dBm above floor | sharp threshold behavior | >50% as SINR collapses | Properties of Mobile Tactical Radio Networks on VHF Bands (ResearchGate 228845813) and military radio handbooks (USMC MCRP 3-40.3C) report SINR-driven thresholds and ~40 dB link margins designed in to keep loss low when nodes are connected; precise loss-rate distributions are not in open lit. |
| 5G mmWave under human/body blockage | nominal ~0% | event-driven outage | 20-56 dB attenuation during a blockage event (effectively total loss) | NSF par.10112905 (Empirical Effects of Dynamic Human-Body Blockage in 60 GHz); 28 GHz human-blockage measurements report 42-56 dB additional attenuation. |

**Headline finding (UAV-mesh row).** The Roofnet result — that per-link loss is
**uniformly distributed across 0-100%** rather than bimodal — is the most
load-bearing fact for bench design. It means the bench cannot validly assume a
"good link / bad link" dichotomy; sweep points must cover the full interior.

### 1.2 Correlated burst-loss duration

| Cause | Typical duration | Source |
|---|---|---|
| 5G mmWave, single human blocker | ~0.2 s of >20 dB attenuation; full outage interval ~300-1000 ms | Empirical Effects of Dynamic Human-Body Blockage in 60 GHz Wireless Systems (NSF par.10112905); MDPI Sensors 20/14/3880 modeling. |
| Iridium inter-satellite handoff | ~0.25 s when handoff succeeds; full call drop if no satellite in view | Iridium constellation overview (kt.agh.edu.pl / Iridium IEEE 710110). |
| Iridium spotbeam handoff | every ~50 s; usually transparent | same. |
| Solar flare / shortwave fadeout | tens of minutes to >1 hr | NOAA SWPC *Solar Flares (Radio Blackouts)*. |
| Post-storm MUF depression (HF) | mean 5.5 h moderate, 8.5 h severe | Saito et al., *Occurrence rate and duration of space weather impacts on HF radio communication used by aviation*, J. Space Weather and Space Climate (2022). |
| Polar Cap Absorption (PCA) | days to weeks | NOAA SWPC. |
| Ionospheric scintillation (equatorial, post-sunset) | minutes-of-fading per event; activity persists ~6-10 h post-sunset; individual fades < ~15 s | Stanford SCPNT scintillation white paper; *Inside GNSS* equatorial summary. |
| Acoustic shadow zone (thermocline-driven) | persistent during stratification; SNR varies by ~10 dB on multi-hour scales; full disconnection possible for nodes deep in zone | Underwater acoustics literature (Wikipedia / arc.id.au); arxiv 2501.02256 (acoustic RIS for shadow zones) cites multi-hour SNR variability. |
| Acoustic shadow zone, spatial | tens of km wide where no signal propagates | arxiv 2501.02256. |
| UAV RF shadow (canyon / urban / mountain) | seconds to ~30 s during fly-through | **Direct quantitative measurement was not located in open lit.** Drone return-to-home timeouts ("a few seconds") are the closest public number. Phase C will treat this as a **documented assumption**, not a citation. |
| Tactical jamming (benign-loss model) | minutes to indefinite while emitter is active | Open-lit jamming surveys (arxiv 2403.19868) describe duration as a function of attacker dwell, not a measured number. Treat as a documented assumption. |

### 1.3 End-to-end broadcast latency

| Mission class | Low | Typical | High | Source |
|---|---|---|---|---|
| LAN / 802.11 mesh, single hop | sub-ms | 1-10 ms | 10s of ms under contention | implicit in Roofnet papers; throughput-dominated, not latency-dominated. |
| 802.11 mesh, multi-hop (3 hops) | — | tens of ms | 100+ ms | Bicket et al. 2005 (avg 627 kbps over 3-hop). |
| Iridium legacy SBD / data | ~980 ms | 1.3-1.8 s RTT | >2 s | DTIC ADA464192; static avg RTT 1686 ms, dynamic 1812 ms. |
| Iridium Certus | 40-50 ms network | ~500 ms (aviation terminal end-to-end) | — | SKYTRAC / Iridium Certus documentation. |
| Inmarsat BGAN | ~900 ms | 900-1700 ms | 2000+ ms (congestion) | Riverbed BGAN doc; Inmarsat BGAN brochure. |
| Acoustic underwater | ~0.67 ms/km propagation | propagation-bound; modems add tens of ms; "tens or hundreds of ms" delay spread | seconds for multi-km links | Stojanovic & Preisig 2009 IEEE Comms Mag; speed-of-sound 1500 m/s. |

### 1.4 Mission duration

| Mission class | Low | Typical | High | Source |
|---|---|---|---|---|
| Coast Guard SAR (single case) | minutes (false alarm) | hours | ~days for offshore-vessel cases | USCG CG-SAR-1 SAR Facts & Reports; aggregate FY2020 = 16,845 cases, time-per-case not directly published in open data we could pull. **The exact distribution of per-case durations was not located**; the bench will use "hours-to-days" as a documented bracket, not a precise figure. |
| UAV ISR sortie, Group 1 small | ~1 h | ~3 h | ~6.5 h (Puma LE Group 2) | AeroVironment Puma AE / Puma LE specs. |
| UAV ISR sortie, tactical Group 2-3 | 8 h | 8-12 h | — | Defense Advancement *Group 2 UAS* survey. |
| UAV ISR, MQ-9 Reaper class | 14 h | 23-30 h | 27-42 h (ER variant, ISR-only) | Wikipedia MQ-9; Air & Space Forces Magazine; GlobalMilitary.net (cross-source consensus). |
| AUV (propelled) | hours | 1 day | several days | Hydro International AUV survey (treat as soft). |
| AUV glider (Slocum, Seaglider) | weeks | months | 18 months max endurance (Slocum) | WHOI Slocum glider page; Teledyne Webb. |

### 1.5 Asymmetric loss patterns

| Channel | What's asymmetric | Source |
|---|---|---|
| 802.11 sensor / mesh | "A substantial percentage of links are asymmetric, many are even unidirectional"; ETX metric was developed precisely because of this | ACM TOSN *On link asymmetry and one-way estimation in wireless sensor networks* (Sang, Arora, Zhang 2010, ACM 1689242). |
| 802.11 outdoor (UAV-to-ground) | Loss from UAV to ground vs ground to UAV diverges with altitude / antenna pattern | Cheng, Hsiao, Kung, Vlah (2006), *Performance Measurement of 802.11a Wireless Links from UAV to Ground Nodes*, ICCCN — measurements exist but **specific asymmetry numbers were not extractable from the PDF in this audit pass**; cited as evidence the phenomenon is documented, not as a concrete percentage. |
| Acoustic underwater | Depth-dependent propagation paths produce one-way signal availability; thermocline can leave one node in surface duct, another in shadow | Wikipedia Underwater acoustics; arxiv 2501.02256. |
| 802.11ay directional mmWave | Standards-level acknowledgment that one station may transmit but not receive | IEEE 802.11ay primer (engineering.wustl.edu). |

**Open-lit gap.** None of the sources we accessed gives a clean "fraction of
nodes that are deaf to a given broadcast" distribution for a real swarm. The
qualitative finding is robust ("substantial fraction asymmetric"); the
quantitative one is not. Section 3 calls this out.

### 1.6 Iteration rate / replan cadence

| System | Cadence | Source |
|---|---|---|
| SAROPS (USCG operational) | Trajectory predictions at fixed intervals (e.g., every 5 min for particles); the human-driven planner cycle is **per search-asset deployment**, not a closed-loop autonomous re-plan. Environmental data refresh is per-EDS-update (hours). | Wikipedia SAROPS; Kratzke, Stone IEEE 5712114 *Search and Rescue Optimal Planning System*; *Advances in Search and Rescue at Sea* Davidson et al. 2012. |
| Autonomous UAV swarm replanning | Sub-second to seconds; specific Hz figures not standardized in the open-lit surveys | Multiple FANET / swarm surveys (Springer s44147-025-00582-3; Science Robotics abm5954). **No single survey gives a canonical "Hz" number**; values are platform-specific. |

---

## Section 2 — Recommended bench sweep ranges (load-bearing artifact)

These are the discrete sweep sets the bench should cover. Each point ties back
to a mission class; the rationale column says which Section 1 row justifies it.

### 2.1 Per-broadcast packet loss rate

**Recommended sweep: p ∈ {0%, 5%, 15%, 30%, 50%, 70%, 90%}.**

| Point | Covers | Rationale |
|---|---|---|
| 0% | clean baseline | sanity / regression check; not a real-world claim. |
| 5% | benign UAV-mesh, near-LOS; satellite "good day" | within Roofnet's lower-loss cluster; below 802.11s urban-swarm typical (7-14%). |
| 15% | typical UAV mesh, mixed urban / open | spans Roofnet intermediate band; matches 802.11s urban-swarm 900-node sim. |
| 30% | stressed UAV mesh; lower edge of acoustic AUV band | matches lower bound of the 32.8-61.5% AUV-experimental band. |
| 50% | acoustic-AUV mid-band; shadowed RF-NLOS | center of AUV-experimental band; tests architecture at "half the broadcasts heard". |
| 70% | acoustic shadow zone, severe RF-NLOS | upper band of AUV experimental loss; brutally adversarial benign loss. |
| 90% | failure-mode probe | tests graceful-degradation claim under near-total comms collapse; not a claim about a specific real channel. |

The choice of 7 points (not finer) reflects the Roofnet finding that **link
loss is uniformly distributed across 0-100% with no clean threshold**. Sampling
densely near 0% or near 100% would be unfaithful to that distribution.

### 2.2 Correlated burst-loss duration

**Recommended sweep: burst_duration ∈ {0 s, 1 s, 5 s, 30 s, 300 s, 1800 s}**
combined with **off-burst loss ∈ {p from 2.1}** under a Gilbert-Elliott
two-state model.

| Point | Covers | Rationale |
|---|---|---|
| 0 s | independent loss baseline | matches Roofnet finding that *most* per-link losses are non-bursty. |
| 1 s | mmWave blockage, satellite handoff glitch | mmWave outage 0.3-1 s; Iridium spotbeam handoff hiccup. |
| 5 s | typical UAV-canyon flythrough (assumed) | **documented assumption**, see Section 3. |
| 30 s | extended canyon / urban shadow (assumed); jamming dwell | upper end of return-to-home thresholds; matches typical jamming-dwell anecdotes. |
| 300 s (5 min) | acoustic shadow-zone transit; HF fadeout shoulder | matches lower edge of HF post-flare fadeouts. |
| 1800 s (30 min) | severe ionospheric event | within Saito et al.'s mean-event window for moderate post-storm MUF depression (mean 5.5 h, so 30 min is conservative). |

GE-model burst-length default: **average burst length 1.5-2.5 packets** when
not running an explicit burst-duration sweep, per Hasslinger & Hohlfeld (2008,
ResearchGate 221440836). This is the "off-the-shelf" GE setting and matches
Internet measurement.

### 2.3 End-to-end broadcast latency

**Recommended sweep: latency ∈ {1 ms, 50 ms, 500 ms, 1500 ms, 5000 ms}.**

| Point | Covers | Rationale |
|---|---|---|
| 1 ms | LAN-class single-hop | swarm internal mesh, low contention. |
| 50 ms | multi-hop mesh; Iridium Certus network-only | matches mid-band 802.11 multi-hop and Certus 40-50 ms. |
| 500 ms | Certus aviation end-to-end; short acoustic link | matches SKYTRAC IMS-350 figure; ~1 km acoustic link, modem overhead. |
| 1500 ms | legacy Iridium / Inmarsat BGAN | center of 1.3-1.8 s legacy Iridium and 0.9-1.7 s BGAN bands. |
| 5000 ms | acoustic multi-km; congested BGAN | acoustic 1500 m/s × multi-km plus protocol; BGAN peak >2 s. |

### 2.4 Mission duration

**Recommended sweep: T_mission ∈ {600 s, 3600 s, 14400 s, 86400 s, 604800 s}**
i.e. **{10 min, 1 h, 4 h, 1 day, 1 week}**.

| Point | Covers | Rationale |
|---|---|---|
| 10 min | unit / regression test | not a mission claim. |
| 1 h | Group 1 UAV partial sortie | within Puma AE 3-h envelope. |
| 4 h | typical Group 2 UAV sortie; long Coast Guard small-boat case | within tactical 8-12 h envelope. |
| 1 day | MQ-9-class ISR; multi-day Coast Guard offshore SAR | within 23-30 h MQ-9 endurance. |
| 1 week | AUV glider deployment; multi-day SAR | within Slocum / Seaglider weeks-to-months envelope. |

### 2.5 Asymmetric loss

**Recommended sweep: deaf_fraction ∈ {0, 0.1, 0.25, 0.5}** of broadcasters who
**don't hear** a given epoch's traffic, drawn IID per epoch.

| Point | Rationale |
|---|---|
| 0 | symmetric baseline. |
| 0.1 | mild antenna-pointing / fading effect — **soft assumption** (see Section 3). |
| 0.25 | one quadrant of swarm in shadow. |
| 0.5 | half the swarm partitioned (geometry, depth band). |

**Caveat.** The literature firmly establishes that asymmetric/unidirectional
links exist in significant fractions, but does not give a canonical
"deaf-fraction distribution" for swarm broadcasts. The above is a
*defensible-but-assumed* sweep; Section 3 records this.

### 2.6 Replan cadence

**Recommended sweep: replan_period ∈ {0.1 s, 1 s, 10 s, 300 s}.**

| Point | Covers | Rationale |
|---|---|---|
| 0.1 s (10 Hz) | aggressive autonomous swarm | ceiling-of-feasible from FANET surveys. |
| 1 s | typical autonomous swarm | mid-band; matches sub-second-to-seconds claim in surveys. |
| 10 s | conservative autonomous; SAROPS particle-trajectory step | matches SAROPS 5-min particle interval order-of-magnitude floor for autonomy. |
| 300 s (5 min) | SAROPS particle update; environmental refresh | matches SAROPS trajectory-prediction interval. |

The bench does **not** sweep all the way out to "human-in-the-loop hours"
because the architecture's claim is autonomous; that operating point is the
human-comparison baseline, not an autonomous design point.

---

## Section 3 — Gaps (what the open lit does not give us)

The bench will need to document each of these as an assumption rather than a
citation:

1. **UAV RF shadow / canyon flythrough duration.** Drone manufacturers cite
   "a few seconds" RTH timeouts; no peer-reviewed measurement of canyon-transit
   blackout-duration distribution was located in this pass. **Bench assumption:
   1-30 s, log-uniform.** Tag in Phase C: `assumption.canyon_blackout`.

2. **Tactical-jamming dwell duration.** Open-lit treats this as
   attacker-dependent; no benign field-measured distribution. **Bench
   assumption: same range as canyon plus a heavy-tail (up to 30 min) component
   for sustained jamming.** Tag: `assumption.jamming_dwell`.

3. **Inmarsat BGAN per-packet loss SLA.** Inmarsat publishes 99.9%
   *availability*, not a loss-rate-conditional-on-availability SLA. **Bench
   assumption: when available, treat BGAN packet-loss as ≤1% steady-state, with
   rare deep fades modeled by the GE bad-state.** Tag: `assumption.bgan_loss`.

4. **Coast Guard per-case mission duration distribution.** Aggregate annual
   counts are public; per-case duration histograms are not in the documents we
   accessed. **Bench assumption: log-uniform 1 h to 3 days.** Tag:
   `assumption.uscg_case_duration`.

5. **Per-swarm deaf-fraction distribution.** Asymmetry is qualitatively
   established; the exact fraction-of-deaf-receivers distribution is not.
   **Bench assumption: sweep across {0, 0.1, 0.25, 0.5} as a parametric study,
   not a calibrated field number.** Tag: `assumption.deaf_fraction`.

6. **Autonomous-swarm replan-cadence canonical value.** Surveys agree on
   "sub-second to seconds" but no canonical Hz. **Bench assumption: cadence is
   itself a sweep variable** (Section 2.6); we do not commit to one value.

7. **Roofnet quantitative loss histogram.** The 2004/2005 papers state the
   uniform-across-0-to-100% finding, but extracting numerical histogram bins
   requires the original PDF, which the WebFetch path could not parse. The
   qualitative finding is sourced; the bench's mid-band sampling (Section 2.1)
   is consistent with it but not pixel-matched.

---

## Section 4 — Cross-variable correlations

The bench must not sweep meaningless combinations. This section records which
variable combinations are physically coupled in real channels, so the sweep
matrix in Phase C can prune nonsense corners.

| Channel | Latency | Loss | Burst structure | Mission duration | Note |
|---|---|---|---|---|---|
| UAV mesh (802.11) | 1-100 ms | 0-90% per link, uniform | mostly non-bursty (Roofnet) | 1 h - 30 h | Latency stays low even when loss is high; bursts are the *exception*, not the rule. |
| RF-LOS (tactical VHF/UHF) | 1-50 ms | low until SINR threshold, then collapse | threshold-driven bursts (terrain) | 1 h - 30 h | Sharp on/off; intermediate band is narrow. |
| RF-NLOS (urban / canyon) | same as RF-LOS | shadow-driven blackouts | bursty by construction | 1 h - 30 h | Couple burst-duration to motion model. |
| Satellite (legacy Iridium) | 1300-1800 ms | low average; handoff glitches | sub-second handoff bursts; rare longer drops | hours-days | High-latency / low-loss / occasional-burst regime. **Not** the same as acoustic. |
| Satellite (Certus) | 40-500 ms | low | as above | as above | Modern LEO improves latency but keeps the handoff burst structure. |
| Inmarsat BGAN | 900-2000 ms | low (assumption) | rare deep fades | hours-days | "GEO with weather" regime. |
| Acoustic underwater | 100s of ms - seconds | 30-60% | spatially-correlated shadow zones; hours-scale SNR variation | hours - **months** | Couples high latency, high loss, long mission duration, and depth-driven asymmetry. The bench's "acoustic-AUV-glider" point is the only place where week-scale missions meet 50% loss. |
| 5G mmWave | low | event-driven | sub-second outages from blockers | minutes - hours | Bursty by physics; latency is fine when not blocked. |

**Implications for the sweep matrix.**

- **Do not** combine LAN-class latency (1 ms) with acoustic-class loss (50%)
  unless explicitly modeling a hypothetical / pathological setup; that point is
  a mathematical stress test, not a real channel.
- **Do** combine satellite-class latency (1500 ms) with low loss (≤5%) and
  short bursts (≤1 s) as the "Iridium legacy operating point."
- **Do** combine acoustic-class latency (500-5000 ms) with 30-70% loss, long
  bursts (300+ s), and week-scale mission duration as the "AUV glider
  operating point."
- **Do** combine 802.11-class latency (1-50 ms) with the full 0-90% loss range
  but mostly **non-bursty** loss as the "UAV mesh operating point" — the
  Roofnet finding rules out long-burst structure as the dominant mode.
- The mmWave column should be modeled as **low loss + bursty** with bursts in
  the 0.3-1 s band; latency stays low.

The bench will mark each cell of the sweep matrix with one of these mission
classes; cells with no class assignment are explicitly stress tests, not claims
about real channels.

---

## Sources cited (for re-checking and citation in Phase C papers)

- Aguayo, D., Bicket, J., Biswas, S., Judd, G., Morris, R. (2004).
  *Link-level Measurements from an 802.11b Mesh Network*. ACM SIGCOMM CCR.
  https://web.stanford.edu/class/cs244/papers/roofnet-sigcomm04.pdf
- Bicket, J., Aguayo, D., Biswas, S., Morris, R. (2005). *Architecture and
  Evaluation of an Unplanned 802.11b Mesh Network*. MobiCom.
  https://pdos.csail.mit.edu/papers/roofnet:mobicom05/roofnet-mobicom05.pdf
- Stojanovic, M., Preisig, J. (2009). *Underwater acoustic communication
  channels: Propagation models and statistical characterization*. IEEE
  Communications Magazine 47(1):84-89.
  https://ieeexplore.ieee.org/document/4752682/
- Sozer, E. M., Stojanovic, M., Proakis, J. G. (2000). *Underwater Acoustic
  Networks*. IEEE J. Oceanic Engineering 25(1):72-83.
- McMahon, M., Rathburn, R. *Measuring Latency in Iridium Satellite
  Constellation Data Services*. DTIC ADA464192.
  https://apps.dtic.mil/sti/pdfs/ADA464192.pdf
- Saito, S. et al. (2022). *Occurrence rate and duration of space weather
  impacts on high-frequency radio communication used by aviation*. J. Space
  Weather and Space Climate.
  https://www.swsc-journal.org/articles/swsc/full_html/2022/01/swsc220003/swsc220003.html
- NOAA SWPC, *Solar Flares (Radio Blackouts)*.
  https://www.swpc.noaa.gov/phenomena/solar-flares-radio-blackouts
- Stanford SCPNT GPS Lab, *Effect of Ionospheric Scintillations on GNSS — A
  White Paper*.
  https://web.stanford.edu/group/scpnt/gpslab/website_files/sbas-ion_wg/sbas_iono_scintillations_white_paper.pdf
- Kratzke, T., Stone, L. *Search and Rescue Optimal Planning System*. IEEE
  Conf. Pub. 5712114; Metron preprint
  https://www.metsci.com/wp-content/uploads/2019/08/Search-and-Rescue-Optimal-Planning-System.pdf
- USCG CG-SAR-1, *SAR Facts & Reports*.
  https://www.dco.uscg.mil/Our-Organization/Assistant-Commandant-for-Response-Policy-CG-5R/Office-of-Incident-Management-Preparedness-CG-5RI/US-Coast-Guard-Office-of-Search-and-Rescue-CG-SAR/CG-SAR-1/SAR-Facts-Reports/
- AeroVironment Puma AE / Puma LE specifications.
  https://www.avinc.com/uas/puma-ae , https://www.avinc.com/uas/puma-le
- General Atomics MQ-9A specifications and Wikipedia entry.
  https://en.wikipedia.org/wiki/General_Atomics_MQ-9_Reaper
- WHOI Slocum Glider page.
  https://www.whoi.edu/what-we-do/explore/underwater-vehicles/auvs/slocum-glider/
- Hasslinger, G., Hohlfeld, O. (2008). *The Gilbert-Elliott Model for Packet
  Loss in Real Time Services on the Internet*. MMB.
  https://people.computing.clemson.edu/~jmarty/projects/lowLatencyNetworking/papers/APPFEC/GEModelForLossinTheRTInternet.pdf
- Inmarsat BGAN brochure / developer pages.
  https://developer.inmarsat.com/technology/bgan/
- *Empirical Effects of Dynamic Human-Body Blockage in 60 GHz*, NSF par.10112905.
  https://par.nsf.gov/servlets/purl/10112905
- Sang, L., Arora, A., Zhang, H. (2010). *On link asymmetry and one-way
  estimation in wireless sensor networks*. ACM TOSN.
  https://dl.acm.org/doi/10.1145/1689239.1689242
- Cheng, C.-M., Hsiao, P.-H., Kung, H. T., Vlah, D. (2006). *Performance
  Measurement of 802.11a Wireless Links from UAV to Ground Nodes*. ICCCN.
  https://www.eecs.harvard.edu/~htk/publication/2006-icccn-cheng-hsiao-kung-vlah.pdf
- *Covering Underwater Shadow Zones using Acoustic Reconfigurable Intelligent
  Surfaces*. https://arxiv.org/html/2501.02256v1

**Access barriers encountered:** several IEEE / ACM / NSF PDFs returned binary
content that the WebFetch text-extraction path could not parse (Roofnet
SIGCOMM 2004 PDF, Roofnet MobiCom 2005 PDF, Buffalo INFOCOM 2017 paper,
Harvard Cheng et al. 2006, MIT/WHOI handbook chap 5, Metron SAROPS
preprint). For each, the qualitative findings extracted via search-result
snippets are cited; the raw histograms / tables are flagged as gaps in
Section 3 and would be the next thing to extract by hand if Phase C needs
finer numbers.

---

*Word count target was 2000-3000. This document is sized to fit; Section 2's
sweep tables are the load-bearing artifact for Phase C.*
