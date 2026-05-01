# /// script
# dependencies = ["numpy<3", "matplotlib"]
# ///
"""Generate the headline figures for the paper.

Produces PNG plots in `figures/`:
  fig1_gap_vs_n.png     — optimality gap shrinks as O(1/log N)
  fig2_recovery.png      — reassignment fraction by cluster size and surplus
  fig3_attrition.png     — formation occupancy under sustained Poisson loss
  fig4_localization.png  — formation drift across GPS / IMU regimes
  fig5_priority.png      — uniform vs shadow vs tiered max-extra-distance
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def fig1_gap_vs_n():
    Ns = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    gap = np.array([5.54, 5.05, 3.02, 2.20, 1.70, 1.52, 1.43])
    gap_sd = np.array([4.30, 2.04, 0.73, 0.29, 0.04, 0.04, 0.01])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(Ns, gap, yerr=gap_sd, fmt='o-', color='steelblue',
                linewidth=2, markersize=8, capsize=4, label='Empirical')

    fit_M5 = 1.27 + 14.12 / np.sqrt(Ns)
    fit_M1 = 12.25 / np.log(Ns)
    ax.plot(Ns, fit_M5, '--', color='crimson', linewidth=1.5,
            label=r'$1.27 + 14.12/\sqrt{N}$ (best fit)')
    ax.plot(Ns, fit_M1, ':', color='gray', linewidth=1.0,
            label=r'$12.25/\ln(N)$ (alternative)')

    ax.set_xscale('log')
    ax.set_xlabel('Swarm size N', fontsize=12)
    ax.set_ylabel('Gap above Hungarian optimum (%)', fontsize=12)
    ax.set_title('Hierarchical assignment gap: best-fit form is a + b/√N',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_gap_vs_n.png"), dpi=120)
    plt.close(fig)
    print("Saved fig1_gap_vs_n.png")


def fig2_recovery():
    cluster = np.array([1, 3, 5, 10])
    rerun = np.array([23.0, 33.0, 45.5, 52.0])
    rerun_sd = np.array([15.2, 9.2, 6.1, 5.9])
    patch = np.array([0.9, 2.8, 4.8, 9.0])
    patch_sd = np.array([0.0, 0.0, 0.0, 0.6])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(cluster, rerun, yerr=rerun_sd, fmt='s-', color='crimson',
                linewidth=2, markersize=10, capsize=5,
                label='Rerun (no surplus)')
    ax.errorbar(cluster, patch, yerr=patch_sd, fmt='o-', color='forestgreen',
                linewidth=2, markersize=10, capsize=5,
                label='Patch (S=10 surplus)')
    ax.set_xlabel('Cluster size K', fontsize=12)
    ax.set_ylabel('Reassignment fraction (%)', fontsize=12)
    ax.set_title('Recovery: patch is linear in cluster size, rerun is bimodal',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(cluster)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_recovery.png"), dpi=120)
    plt.close(fig)
    print("Saved fig2_recovery.png")


def fig3_attrition():
    t = np.arange(0, 240, 24)
    no_surplus = np.array([100.0, 97.8, 94.0, 88.4, 85.6, 82.6, 79.6, 74.8, 70.6, 66.8])
    s10 = np.array([100.0, 100.0, 100.0, 98.4, 95.6, 92.6, 89.6, 84.8, 80.6, 76.8])
    s30 = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.0, 96.0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, no_surplus, 'o-', color='crimson', linewidth=2, markersize=8,
            label='No surplus')
    ax.plot(t, s10, 's-', color='steelblue', linewidth=2, markersize=8,
            label='Uniform surplus = 10')
    ax.plot(t, s30, '^-', color='forestgreen', linewidth=2, markersize=8,
            label='Uniform surplus = 30')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Occupied primary leaves', fontsize=12)
    ax.set_title('Sustained attrition (λ = 0.15 deaths/s, N = 100):\n'
                 'surplus extends time-to-first-gap, then formation degrades '
                 'one-per-loss',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim(60, 102)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_attrition.png"), dpi=120)
    plt.close(fig)
    print("Saved fig3_attrition.png")


def fig4_localization():
    t = np.array([0, 12, 24, 36, 48, 60, 72, 84, 96])
    gps_on = np.array([0.000, 0.13, 0.13, 0.13, 0.14, 0.14, 0.14, 0.14, 0.13])
    gps_off = np.array([0.000, 0.023, 0.065, 0.115, 0.180, 0.253, 0.329, 0.413, 0.502])
    outage = np.array([0.000, 0.135, 0.140, 0.145, 0.155, 0.180, 0.220, 0.275, 0.341])
    tactical = np.array([0.000, 0.005, 0.012, 0.018, 0.025, 0.032, 0.040, 0.048, 0.053])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, gps_off, 'o-', color='crimson', linewidth=2, markersize=7,
            label='GPS off (consumer IMU)')
    ax.plot(t, outage, 's-', color='orange', linewidth=2, markersize=7,
            label='GPS for 30s, then outage')
    ax.plot(t, gps_on, '^-', color='steelblue', linewidth=2, markersize=7,
            label='GPS at 1 Hz, σ=0.1m')
    ax.plot(t, tactical, 'd-', color='forestgreen', linewidth=2, markersize=7,
            label='Tactical-grade IMU, GPS off')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Formation error (m, mean over drones)', fontsize=12)
    ax.set_title('Formation tolerance vs GPS regime', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_localization.png"), dpi=120)
    plt.close(fig)
    print("Saved fig4_localization.png")


def fig5_priority():
    configs = ['Uniform 15', 'Pure key shadow', 'Tiered (10+5)']
    cluster_at_keys = [13.78, 4.63, 7.75]
    cluster_at_filler = [14.21, 17.16, 16.73]

    x = np.arange(len(configs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, cluster_at_keys, w, color='crimson',
                label='Cluster at keys')
    b2 = ax.bar(x + w/2, cluster_at_filler, w, color='steelblue',
                label='Cluster at non-keys')
    ax.set_ylabel('Max-extra-distance (drone units)', fontsize=12)
    ax.set_title('Tiered redundancy: same surplus budget, different allocation',
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', (bar.get_x() + bar.get_width() / 2, h),
                        ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_priority.png"), dpi=120)
    plt.close(fig)
    print("Saved fig5_priority.png")


if __name__ == '__main__':
    fig1_gap_vs_n()
    fig2_recovery()
    fig3_attrition()
    fig4_localization()
    fig5_priority()
    print(f"\nAll figures in {FIGDIR}/")
