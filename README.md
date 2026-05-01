# Decentralized Swarm Coordination

A four-layer decentralized coordination architecture for drone swarms,
built on a single primitive: broadcast-as-shared-state with locally-
deterministic per-drone computation against a PCA tree of the target
manifold.

The four layers — assignment, recovery, priority allocation, and
localization — are independently provable, empirically validated across
N=10 to N=10,000, and compose without interference.

**Citable archives**:
- Paper: [10.5281/zenodo.19954717](https://doi.org/10.5281/zenodo.19954717)
- Software (this repository): [10.5281/zenodo.19954678](https://doi.org/10.5281/zenodo.19954678)

See `paper.pdf` (compiled from `paper.tex`) for the full writeup, or
`WRITEUP.md` and `PROOFS.md` for the source.

## Quick demo

```bash
# Live four-phase formation: sphere → torus → cube → star
python3 simulator.py

# Render to mp4 (requires ffmpeg)
SAVE_VIDEO=1 VIDEO_PATH=swarm.mp4 python3 simulator.py
```

The recorded demo is `swarm.mp4`.

## Reproducing the empirical results

Each script is self-contained with inline `# /// script` dependency
declarations; run with `python3 <file>` if numpy/scipy/matplotlib are
installed, or `uv run <file>` to fetch dependencies automatically.

```bash
# Layer 1: Hierarchical assignment vs Hungarian (scaling sweep)
python3 bench_assignment.py

# Layer 2: Loss recovery (single death, cluster, surplus, tiered)
python3 bench_loss.py
SURPLUS=10 python3 bench_loss.py
KEY_SURPLUS=10 FILLER_SURPLUS=5 KEY_COUNT=10 python3 bench_loss.py
LOSS_RATE=0.15 MAX_TICKS=6000 python3 bench_attrition.py

# Layer 2 — patch optimality (greedy vs Hungarian cluster)
python3 bench_patch_optimality.py

# Layer 4: Localization (drift dynamics across GPS regimes)
python3 bench_localization.py

# Layer 4 — full fiducial-selection + cooperative-localization protocol
python3 bench_layer4.py

# Decentralized comparator: hierarchical vs CBBA (Choi-Brunet-How 2009)
python3 bench_cbba.py

# Adversarial threat: byzantine cascade and statistical defenses
python3 bench_adversarial.py

# Witness-alarm detection (the operational defense)
python3 bench_witness.py

# Floating-point determinism stress test
python3 bench_determinism.py

# Streaming/mocap-style time-varying manifolds
CYCLE=30 python3 bench_streaming.py

# Empirical fit comparison for the optimality gap (Conjecture 4)
python3 bench_conjecture4.py

# Generate the five paper figures
python3 make_figures.py
```

## Headline results

All headline numbers carry bootstrap 95% CIs (resampled from per-seed
runs); see source scripts for the per-experiment seed counts.

| Claim | Number | Source |
|---|---|---|
| Assignment gap from Hungarian | 1.43% at N=10,000 | `bench_assignment.py` |
| Hierarchical vs CBBA messages | 100 vs 1350 (14× fewer); 3.0% vs 7.7% gap | `bench_cbba.py` |
| Empirical fit form | `1.27 + 14.12/√N` (best of 5 candidates by AIC/BIC) | `bench_conjecture4.py` |
| Single-death reassignment | 0.9% with surplus + patch | `bench_loss.py` |
| Cluster recovery (S ≥ K) | exactly K reassignments, 0 unfilled | `bench_loss.py` |
| Tiered redundancy benefit | 56% reduction in flight cost when threat correlates with priority | `bench_loss.py` |
| Layer 4 formation tolerance (heavy-tail, t=78s) | 0.082m [0.072, 0.092] vs 0.252 GPS-only vs 0.409 INS-only; 30 seeds × 2000 ticks | `bench_layer4.py` |
| Witness-alarm detection | ~100% above 5σ threshold under Gaussian noise; 14% FP under heavy-tail at 5σ, 1.3% at 10σ | `bench_witness.py` |
| Floating-point determinism | robust to perturbations up to 1mm; FP differences are 1e-15m | `bench_determinism.py` |

## File layout

```
drone_swarm/
├── simulator.py             — live four-phase formation, video render
├── bench_assignment.py      — Hungarian comparison, scaling sweep
├── bench_loss.py            — recovery, cluster, shadow, tiered
├── bench_patch_optimality.py — greedy vs Hungarian cluster
├── bench_attrition.py       — Poisson loss process, graceful degradation
├── bench_localization.py    — INS noise, GPS regimes, drift
├── bench_layer4.py          — fiducial selection + cooperative localization
├── bench_cbba.py            — CBBA decentralized auction comparator
├── bench_witness.py         — byzantine detection via mutual observation
├── bench_adversarial.py     — k-byzantine cascade, MAD/σ outlier rejection
├── bench_determinism.py     — FP perturbation stress test
├── bench_streaming.py       — time-varying manifolds (mocap-style)
├── bench_conjecture4.py     — empirical fit-form comparison
├── make_figures.py          — generates the five paper figures
├── paper.tex / paper.pdf    — compiled paper (single document)
├── WRITEUP.md               — paper source (markdown)
├── PROOFS.md                — formal lemmas, theorems, proofs
├── figures/                 — 5 PNG figures
├── swarm.mp4                — rendered four-phase demo
└── README.md                — this file
```

## License

Reserved.
