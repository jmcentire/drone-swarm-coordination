# /// script
# dependencies = ["numpy<3"]
# ///
"""Pretty-print bench_search_results.json for inclusion in NOTE_SEARCH.md."""

import json
import sys
import numpy as np


def fmt_ci(arr):
    a = np.asarray([x for x in arr if x is not None and not (isinstance(x, float) and np.isnan(x))])
    if not len(a):
        return "—"
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(a), size=(1000, len(a)))
    means = a[idxs].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return f"{a.mean():.1f} [{lo:.1f}, {hi:.1f}]"


def main(path="bench_search_results.json"):
    with open(path) as f:
        data = json.load(f)
    runs = data["runs"]
    comms = data.get("comms_runs", [])
    summary = data.get("summary", [])

    print("# Primary scenarios — find rate, iters when found, distance\n")
    by_scen = {}
    for r in runs:
        by_scen.setdefault(r["scenario"], {}).setdefault(r["algorithm"], []).append(r)

    algo_order = ["bayesian", "bayesian_eig", "sarops_class", "bayesian_bc",
                  "lawnmower", "lawnmower_drift", "random", "oracle"]
    for scen, algos in sorted(by_scen.items()):
        print(f"## {scen}")
        print("| Algorithm | found | iters when found | distance | final conf |")
        print("|---|---|---|---|---|")
        for algo in algo_order:
            rs = algos.get(algo, [])
            if not rs:
                continue
            found = sum(1 for r in rs if r["found"])
            iters_w = [r["iterations"] for r in rs if r["found"]]
            dists = [r["distance"] for r in rs]
            confs = [r.get("final_confidence", 0.0) for r in rs]
            iters_str = fmt_ci(iters_w)
            dist_str = fmt_ci(dists)
            conf_avg = float(np.mean([c for c in confs if not np.isnan(c)]) if confs else 0.0)
            print(f"| {algo} | {found}/{len(rs)} | {iters_str} | {dist_str} | {conf_avg:.3f} |")
        print()

    print("\n# EIG vs coverage-mass-argmax (per-scenario lift)\n")
    print("| Scenario | bayesian iters | bayesian_eig iters | lift |")
    print("|---|---|---|---|")
    for scen in by_scen:
        b = by_scen[scen].get("bayesian", [])
        e = by_scen[scen].get("bayesian_eig", [])
        b_iters = [r["iterations"] for r in b if r["found"]]
        e_iters = [r["iterations"] for r in e if r["found"]]
        if not b_iters or not e_iters:
            continue
        b_mean = np.mean(b_iters)
        e_mean = np.mean(e_iters)
        lift = (b_mean - e_mean) / b_mean * 100
        print(f"| {scen} | {b_mean:.1f} | {e_mean:.1f} | {lift:+.1f}% |")

    if comms:
        print("\n# Comms-drop stress test (legacy IID drop_rate sweep)\n")
        by_dr = {}
        for r in comms:
            key = (r["scenario"], r["algorithm"], r["drop_rate"])
            by_dr.setdefault(key, []).append(r)
        for scen in sorted(set(k[0] for k in by_dr)):
            for algo in sorted(set(k[1] for k in by_dr if k[0] == scen)):
                print(f"## {scen} / {algo}")
                print("| drop_rate | found | iters when found | unanimity |")
                print("|---|---|---|---|")
                for dr in sorted(set(k[2] for k in by_dr
                                     if k[0] == scen and k[1] == algo)):
                    rs = by_dr[(scen, algo, dr)]
                    found = sum(1 for r in rs if r["found"])
                    iters_w = [r["iterations"] for r in rs if r["found"]]
                    unan = [r["unanimity_rate"] for r in rs]
                    iters_str = fmt_ci(iters_w)
                    unan_str = fmt_ci(unan)
                    print(f"| {dr:.2f} | {found}/{len(rs)} | {iters_str} | {unan_str} |")
                print()

    channel_runs = data.get("channel_runs", [])
    if channel_runs:
        print("\n# Channel-sweep stress test (Phase C-1: GE + asymmetric)\n")
        by_ch = {}
        for r in channel_runs:
            key = (r["scenario"], r["algorithm"], r.get("channel_label", ""))
            by_ch.setdefault(key, []).append(r)
        for scen in sorted(set(k[0] for k in by_ch)):
            for algo in sorted(set(k[1] for k in by_ch if k[0] == scen)):
                print(f"## {scen} / {algo}")
                print("| channel | found | iters when found | unanimity |")
                print("|---|---|---|---|")
                # Order: IID first, then GE, then asymmetric
                labels = sorted(set(k[2] for k in by_ch if k[0] == scen and k[1] == algo))
                ordered = ([l for l in labels if l.startswith("iid")]
                           + [l for l in labels if l.startswith("ge")]
                           + [l for l in labels if l.startswith("asym")])
                for lbl in ordered:
                    rs = by_ch[(scen, algo, lbl)]
                    found = sum(1 for r in rs if r["found"])
                    iters_w = [r["iterations"] for r in rs if r["found"]]
                    unan = [r["unanimity_rate"] for r in rs]
                    iters_str = fmt_ci(iters_w)
                    unan_str = fmt_ci(unan)
                    print(f"| {lbl} | {found}/{len(rs)} | {iters_str} | {unan_str} |")
                print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench_search_results.json")
