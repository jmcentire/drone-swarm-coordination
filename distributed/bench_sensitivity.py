# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Parameter sensitivity sweep.

For the headline question — leader consensus in the protocol — sweep the
substrate parameters that matter most for low-comm orchestration:

  * comms_range_m: how close drones must be to communicate
  * sound_speed: propagation delay (m/tick)
  * loss_rate: per-message drop probability

Each sweep point: 10 seeds, report mean + bootstrap CI on
final leader_consensus.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from agent import Agent
from bench_distributed import _initial_positions, _make_manifold
from stats import bootstrap_ci
from world import World, WorldConfig


def run_one(n_drones: int, comms_range_m: float, sound_speed: float,
            loss_rate: float, n_ticks: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    starts = _initial_positions(n_drones, rng)
    cfg = WorldConfig(
        n_drones=n_drones, comms_range_m=comms_range_m,
        sound_speed_m_per_tick=sound_speed, loss_rate=loss_rate,
        max_ticks=n_ticks, log_events=False,
    )
    w = World(cfg, seed=seed)
    for i in range(n_drones):
        a = Agent(drone_id=i, priority=i, position=starts[i].copy())
        w.attach_agent(i, a)
    manifold = _make_manifold(n_drones, rng)
    w.issue_global_command(manifold, heading=np.array([1.0, 0, 0]))
    w.run()
    return {
        "leader_consensus": (
            w.metrics.leader_consensus_frac[-1]
            if w.metrics.leader_consensus_frac else 0.0
        ),
        "form_err_mean": (
            w.metrics.formation_error_mean[-1]
            if w.metrics.formation_error_mean else 0.0
        ),
        "coverage": w.metrics.coverage_frac[-1] if w.metrics.coverage_frac else 0.0,
        "comms_delivery_rate": w.comms.summary()["delivery_rate"],
    }


def sweep(name: str, axis: str, values: list, n_seeds: int = 10,
           defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    base = {
        "n_drones": 30, "comms_range_m": 12.0, "sound_speed": 150.0,
        "loss_rate": 0.0, "n_ticks": 200,
    }
    base.update(defaults)
    rows = []
    for v in values:
        cfg = dict(base)
        cfg[axis] = v
        results = [run_one(seed=s, **cfg) for s in range(n_seeds)]
        lc = bootstrap_ci([r["leader_consensus"] for r in results])
        fe = bootstrap_ci([r["form_err_mean"] for r in results])
        cd = bootstrap_ci([r["comms_delivery_rate"] for r in results])
        rows.append({"value": v, "lc": lc, "fe": fe, "cd": cd, "n_seeds": n_seeds})
    return {"name": name, "axis": axis, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--output", default="/Users/jmcentire/Code/drone_swarm/distributed/bench_sensitivity_results.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    sweeps = []

    print("Sweep: comms range")
    s = sweep("Comms range (m)", "comms_range_m", [6.0, 8.0, 12.0, 20.0, 30.0],
              n_seeds=args.seeds)
    sweeps.append(s)
    for r in s["rows"]:
        print(f"  range={r['value']:5.1f}m  lc={r['lc'][0]:.3f}[{r['lc'][1]:.3f},{r['lc'][2]:.3f}]"
              f"  fe={r['fe'][0]:.3f}m  cd={r['cd'][0]:.3f}")

    print("\nSweep: loss rate")
    s = sweep("Per-message loss rate", "loss_rate", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
              n_seeds=args.seeds)
    sweeps.append(s)
    for r in s["rows"]:
        print(f"  loss={r['value']:.2f}  lc={r['lc'][0]:.3f}[{r['lc'][1]:.3f},{r['lc'][2]:.3f}]"
              f"  fe={r['fe'][0]:.3f}m  cd={r['cd'][0]:.3f}")

    print("\nSweep: sound speed (m/tick)")
    s = sweep("Sound speed (m/tick)", "sound_speed", [30.0, 60.0, 150.0, 500.0, 1500.0],
              n_seeds=args.seeds)
    sweeps.append(s)
    for r in s["rows"]:
        print(f"  ss={r['value']:5.0f}  lc={r['lc'][0]:.3f}[{r['lc'][1]:.3f},{r['lc'][2]:.3f}]"
              f"  fe={r['fe'][0]:.3f}m  cd={r['cd'][0]:.3f}")

    print("\nSweep: n_drones")
    s = sweep("Drone count", "n_drones", [10, 20, 30, 50],
              n_seeds=args.seeds, defaults={"n_ticks": 250})
    sweeps.append(s)
    for r in s["rows"]:
        print(f"  N={r['value']:3d}  lc={r['lc'][0]:.3f}[{r['lc'][1]:.3f},{r['lc'][2]:.3f}]"
              f"  fe={r['fe'][0]:.3f}m  cd={r['cd'][0]:.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\nWall time: {elapsed:.1f}s")

    with open(args.output, "w") as f:
        json.dump({"sweeps": sweeps, "wall_time_s": elapsed}, f, indent=2, default=str)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
