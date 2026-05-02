"""Generate the eight figures for the provisional patent application.

Output: figures/fig1_pca_tree.png ... fig8_localization.png

Patent-figure conventions used here:
  - Greyscale / line-drawing aesthetic (white background, dark strokes)
  - Reference numerals labelled at structural elements
  - Figure caption "FIG. N" as a title
  - High DPI for sharp embedding in the compiled PDF
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

DPI = 200

# Plot style: clean, schematic, mostly black-and-white.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#222",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 1.4,
    "patch.linewidth": 1.2,
})


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_sphere(n=80, radius=1.0):
    """Fibonacci-sphere point set."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    return np.column_stack((
        radius * np.cos(theta) * np.sin(phi),
        radius * np.sin(theta) * np.sin(phi),
        radius * np.cos(phi),
    ))


def pca_axis(points):
    pts = points - points.mean(axis=0)
    cov = pts.T @ pts / max(len(pts) - 1, 1)
    w, v = np.linalg.eigh(cov)
    return v[:, -1]


def pca_split(points, idx=None):
    if idx is None:
        idx = np.arange(len(points))
    pts = points[idx]
    axis = pca_axis(pts)
    proj = pts @ axis
    order = np.argsort(proj, kind="stable")
    half = len(idx) // 2
    return idx[order[:half]], idx[order[half:]], axis


def arrow(ax, p, q, **kw):
    a = FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14,
                        color=kw.pop('color', '#222'), lw=kw.pop('lw', 1.4),
                        **kw)
    ax.add_patch(a)


# --------------------------------------------------------------------------
# FIG. 1: PCA tree on a target manifold
# --------------------------------------------------------------------------

def fig1():
    pts = make_sphere(96)

    # Recursively split to depth 3 → 8 leaves
    def recurse(idx, depth, max_depth=3):
        if depth == max_depth or len(idx) <= 1:
            return [idx]
        left, right, _ = pca_split(pts, idx)
        return recurse(left, depth + 1, max_depth) + recurse(right, depth + 1, max_depth)

    leaves = recurse(np.arange(len(pts)), 0)

    fig = plt.figure(figsize=(8, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    greys = ['#222', '#333', '#555', '#777', '#999', '#bbb', '#aaa', '#444']
    for i, leaf in enumerate(leaves):
        ax3d.scatter(pts[leaf, 0], pts[leaf, 1], pts[leaf, 2],
                     c=greys[i % len(greys)], s=22, edgecolor='black', linewidths=0.4)
    # First principal axis at root.
    a0 = pca_axis(pts)
    ax3d.quiver(0, 0, 0, a0[0]*1.4, a0[1]*1.4, a0[2]*1.4, color='black', lw=2)
    ax3d.text(a0[0]*1.6, a0[1]*1.6, a0[2]*1.6, ' 210\n(π_root)', fontsize=9)
    ax3d.set_title('Manifold M (200) with PCA-tree leaves (220)', fontsize=10)
    ax3d.set_xlabel(''); ax3d.set_ylabel(''); ax3d.set_zlabel('')
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])

    # Tree diagram on right.
    axt = fig.add_subplot(1, 2, 2)
    axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.axis('off')

    # Draw a binary tree, depth 3.
    def node(x, y, label, ref):
        axt.add_patch(Circle((x, y), 0.04, color='white', ec='black', zorder=3))
        axt.text(x, y, label, ha='center', va='center', fontsize=8, zorder=4)
        axt.text(x + 0.06, y + 0.04, ref, fontsize=8, color='#444')

    def edge(p, q):
        axt.plot([p[0], q[0]], [p[1], q[1]], color='black', lw=1.2, zorder=2)

    # Coords
    levels = [
        [(0.5, 0.92)],
        [(0.25, 0.7), (0.75, 0.7)],
        [(0.13, 0.46), (0.37, 0.46), (0.63, 0.46), (0.87, 0.46)],
    ]
    leaves_xy = [(0.07, 0.18), (0.19, 0.18), (0.31, 0.18), (0.43, 0.18),
                 (0.57, 0.18), (0.69, 0.18), (0.81, 0.18), (0.93, 0.18)]

    refs_internal = ['230', '240a', '240b', '250a', '250b', '250c', '250d']
    flat = levels[0] + levels[1] + levels[2]
    for (x, y), ref in zip(flat, refs_internal):
        node(x, y, '', ref)
    for i, (x, y) in enumerate(leaves_xy):
        node(x, y, f'L{i+1}', '260' if i == 0 else '')

    # Edges
    edge(levels[0][0], levels[1][0]); edge(levels[0][0], levels[1][1])
    for p, qs in zip(levels[1], [levels[2][:2], levels[2][2:]]):
        for q in qs:
            edge(p, q)
    for p, q in zip(levels[2] + levels[2], leaves_xy):
        edge(p, q)

    axt.set_title('Hierarchical PCA tree T(M) (230)', fontsize=10)
    fig.suptitle('FIG. 1', fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, 'fig1_pca_tree.png'), dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG. 2: Recursive bisection of agents along principal axes
# --------------------------------------------------------------------------

def fig2():
    rng = np.random.default_rng(2)
    n = 32
    starts = rng.uniform(-1, 1, (n, 2))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    # Step 1 — root partition
    axis1 = pca_axis(starts - starts.mean(0))
    proj = (starts - starts.mean(0)) @ axis1
    half = n // 2
    order = np.argsort(proj)
    L = order[:half]; R = order[half:]
    centroid = starts.mean(0)

    for ax, title in zip(axes, ['Root partition', 'Depth-1 partition', 'Depth-2 leaves']):
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.spines[:].set_visible(True)

    # Panel 1
    axes[0].scatter(starts[L, 0], starts[L, 1], c='#222', s=36, label='left', edgecolor='black')
    axes[0].scatter(starts[R, 0], starts[R, 1], c='#bbb', s=36, label='right', edgecolor='black')
    # Draw the splitting line (perpendicular to axis through centroid)
    perp = np.array([-axis1[1], axis1[0]])
    axes[0].plot([centroid[0] - perp[0]*1.5, centroid[0] + perp[0]*1.5],
                 [centroid[1] - perp[1]*1.5, centroid[1] + perp[1]*1.5],
                 'k--', lw=1.0)
    arrow(axes[0], (centroid[0], centroid[1]),
          (centroid[0] + axis1[0]*0.6, centroid[1] + axis1[1]*0.6))
    axes[0].text(centroid[0] + axis1[0]*0.7, centroid[1] + axis1[1]*0.7, 'π (310)', fontsize=9)
    axes[0].text(-1.2, 1.15, 'agents D (300)', fontsize=8)

    # Panel 2 — split each half along its own PCA axis
    def split_idx(idx):
        pts = starts[idx]
        a = pca_axis(pts - pts.mean(0))
        p = (pts - pts.mean(0)) @ a
        h = len(idx) // 2
        o = np.argsort(p)
        return idx[o[:h]], idx[o[h:]], pts.mean(0), a

    LL, LR, cL, aL = split_idx(L)
    RL, RR, cR, aR = split_idx(R)
    quads = [(LL, '#222'), (LR, '#555'), (RL, '#999'), (RR, '#ddd')]
    for q, col in quads:
        axes[1].scatter(starts[q, 0], starts[q, 1], c=col, s=36, edgecolor='black')
    for c, a in [(cL, aL), (cR, aR)]:
        perp = np.array([-a[1], a[0]])
        axes[1].plot([c[0] - perp[0]*0.9, c[0] + perp[0]*0.9],
                     [c[1] - perp[1]*0.9, c[1] + perp[1]*0.9], 'k--', lw=0.9)

    axes[1].text(-1.2, 1.15, 'four subgroups (320)', fontsize=8)

    # Panel 3 — leaves (8 colored)
    def split8(idx):
        out = []
        a, b, _, _ = split_idx(idx)
        for sub in (a, b):
            sa, sb, _, _ = split_idx(sub)
            out += [sa, sb]
        return out

    leaves = split8(L) + split8(R)
    grays = ['#111', '#222', '#333', '#555', '#777', '#999', '#bbb', '#ddd']
    for i, leaf in enumerate(leaves):
        axes[2].scatter(starts[leaf, 0], starts[leaf, 1], c=grays[i],
                        s=36, edgecolor='black')
    axes[2].text(-1.2, 1.15, 'eight leaf groups (330)', fontsize=8)

    fig.suptitle('FIG. 2  —  Recursive bisection of agents along principal axes',
                 fontsize=12, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, 'fig2_recursive_bisection.png'), dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG. 3: Patch protocol on agent failure
# --------------------------------------------------------------------------

def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    rng = np.random.default_rng(7)

    # Setup: 4 leaves in a row, primaries (filled) + 1 surplus.
    leaf_x = np.array([0.2, 0.4, 0.6, 0.8])
    leaf_y = 0.5
    primaries_xy = np.column_stack([leaf_x, np.full_like(leaf_x, leaf_y)])
    surplus_xy = np.array([0.5, 0.18])  # parent centroid

    titles = ['Before failure', 'Primary 3 fails', 'Patch: surplus promotes']
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(t, fontsize=10)
        for x in leaf_x:
            ax.plot([x, x], [leaf_y - 0.02, leaf_y + 0.02], 'k', lw=0.6)
        # leaf labels
        for i, x in enumerate(leaf_x):
            ax.text(x, leaf_y + 0.07, f'L{i+1}', fontsize=8, ha='center')

    # Panel 1
    for i, (x, y) in enumerate(primaries_xy):
        axes[0].add_patch(Circle((x, y), 0.025, color='black'))
        axes[0].text(x, y - 0.07, '410', ha='center', fontsize=8, color='#555')
    axes[0].add_patch(Circle(surplus_xy, 0.025, fill=False, ec='black', lw=1.2))
    axes[0].text(surplus_xy[0] + 0.04, surplus_xy[1], '420 (surplus)', fontsize=8, va='center')
    axes[0].text(0.02, 0.92, 'Primaries (410), surplus (420)', fontsize=8)

    # Panel 2 — primary 3 X'd
    for i, (x, y) in enumerate(primaries_xy):
        if i == 2:
            axes[1].plot([x - 0.025, x + 0.025], [y - 0.025, y + 0.025], 'k', lw=2)
            axes[1].plot([x - 0.025, x + 0.025], [y + 0.025, y - 0.025], 'k', lw=2)
            axes[1].text(x, y - 0.07, '430 (failed)', ha='center', fontsize=8)
        else:
            axes[1].add_patch(Circle((x, y), 0.025, color='black'))
    axes[1].add_patch(Circle(surplus_xy, 0.025, fill=False, ec='black', lw=1.2))
    # broadcast staleness indicator
    axes[1].text(0.02, 0.92, 'Stale-broadcast detection (440)', fontsize=8)

    # Panel 3 — surplus moves to L3
    for i, (x, y) in enumerate(primaries_xy):
        if i == 2:
            continue
        axes[2].add_patch(Circle((x, y), 0.025, color='black'))
    arrow(axes[2], surplus_xy, primaries_xy[2], lw=1.6)
    axes[2].add_patch(Circle(primaries_xy[2], 0.025, color='black'))
    axes[2].text(primaries_xy[2][0], primaries_xy[2][1] - 0.07, '450 (promoted)',
                 ha='center', fontsize=8)
    axes[2].text(0.02, 0.92, 'Closest-surplus rule (460)', fontsize=8)

    fig.suptitle('FIG. 3  —  Failure-recovery patch protocol',
                 fontsize=12, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, 'fig3_patch_protocol.png'), dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG. 4: Shadow fleet structure
# --------------------------------------------------------------------------

def fig4():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.2, 1.2)

    # Primary manifold: ring of 24 leaves
    ang = np.linspace(0, 2*np.pi, 24, endpoint=False)
    primary = np.column_stack([np.cos(ang), np.sin(ang)])
    ax.scatter(primary[:, 0], primary[:, 1], c='black', s=28)
    ax.text(-1.4, 1.05, 'Primary fleet on M (510)', fontsize=9)

    # Mark key positions K (every 4th)
    keys_idx = np.arange(0, 24, 4)
    ax.scatter(primary[keys_idx, 0], primary[keys_idx, 1],
               s=110, facecolor='none', edgecolor='black', lw=1.6)
    for i in keys_idx:
        ax.text(primary[i, 0]*1.18, primary[i, 1]*1.18, '520', fontsize=7, ha='center')

    # Shadow fleet 1: same key positions, offset radially outward
    offset_ring = primary[keys_idx] * 1.25
    ax.scatter(offset_ring[:, 0], offset_ring[:, 1], c='#666', s=32, marker='s')
    ax.text(0.3, 1.0, 'Shadow fleet S₁ on M_K (530)', fontsize=9)

    # Shadow fleet 2: orbit further out
    offset_ring2 = primary[keys_idx] * 1.45
    ax.scatter(offset_ring2[:, 0], offset_ring2[:, 1], c='#aaa', s=32, marker='^')
    ax.text(0.3, 0.85, 'Shadow fleet S₂ on M_K (540)', fontsize=9)

    # Connect each shadow agent to its key with a thin line
    for k, p_off, p_off2 in zip(primary[keys_idx], offset_ring, offset_ring2):
        ax.plot([k[0], p_off[0]], [k[1], p_off[1]], color='#888', lw=0.5)
        ax.plot([k[0], p_off2[0]], [k[1], p_off2[1]], color='#888', lw=0.5)

    fig.suptitle('FIG. 4  —  Shadow fleet allocation at key positions',
                 fontsize=12, fontweight='bold', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, 'fig4_shadow_fleet.png'), dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG. 5: Phase transition through quiescence
# --------------------------------------------------------------------------

def fig5():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    titles = [
        '(a) Converging on M_A',
        '(b) Quiescent state\non M_A',
        '(c) Snapshot latched;\nhold timer',
        '(d) Computing assignment\non M_B; transit',
    ]

    for ax, t in zip(axes, titles):
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.2, 1.2)
        ax.set_title(t, fontsize=9)

    rng = np.random.default_rng(12)

    # Manifold A: triangle pattern
    n = 18
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    M_A = np.column_stack([0.9*np.cos(ang), 0.9*np.sin(ang)])
    M_B = np.column_stack([0.6*np.cos(ang*2), 0.9*np.sin(ang)])  # squashed

    # (a) drones converging — show drones with motion-blur arrows from random to ring
    starts_a = rng.uniform(-1.2, 1.2, (n, 2))
    for i in range(n):
        arrow(axes[0], starts_a[i], M_A[i] * 0.95, color='#888', lw=0.7)
    axes[0].scatter(M_A[:, 0]*0.95, M_A[:, 1]*0.95, c='black', s=18)
    axes[0].text(-1.3, -1.1, 'broadcast B(t) volatile (610)', fontsize=8)

    # (b) quiescent
    axes[1].scatter(M_A[:, 0], M_A[:, 1], c='black', s=24)
    axes[1].text(-1.3, -1.1, 'broadcast B(t) ≡ const (620)', fontsize=8)

    # (c) hold timer / snapshot — same plot but with dashed circle "lock"
    axes[2].scatter(M_A[:, 0], M_A[:, 1], c='black', s=24)
    box = FancyBboxPatch((-1.25, 0.85), 2.5, 0.18,
                         boxstyle="round,pad=0.02", lw=1.2, ec='black', fc='none')
    axes[2].add_patch(box)
    axes[2].text(0, 0.94, 'snapshot S latched (630)', ha='center', fontsize=8)

    # (d) drones transiting to B
    for i in range(n):
        arrow(axes[3], M_A[i], M_B[i], color='#888', lw=0.7)
    axes[3].scatter(M_B[:, 0], M_B[:, 1], c='black', s=18)
    axes[3].text(-1.3, -1.1, 'broadcast B(t) volatile again (640)', fontsize=8)

    fig.suptitle('FIG. 5  —  Phase transition synchronized through quiescence',
                 fontsize=12, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, 'fig5_phase_transition.png'), dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG. 6: Threat-to-manifold mappings
# --------------------------------------------------------------------------

def fig6():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))

    # (a) high-value missile — dense spherical formation
    ax = axes[0]
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.2, 1.2)
    ax.set_title('(a) Missile threat → dense sphere', fontsize=9)
    ang = np.linspace(0, 2*np.pi, 36, endpoint=False)
    pts = np.column_stack([0.7*np.cos(ang), 0.7*np.sin(ang)])
    ax.scatter(pts[:, 0], pts[:, 1], c='black', s=18)
    ax.add_patch(Circle((0, 0), 0.7, fill=False, ec='black', lw=0.6, ls='--'))
    arrow(ax, (-1.3, -0.9), (-0.5, -0.4))
    ax.text(-1.4, -1.05, 'threat 710', fontsize=8)
    ax.text(0.6, 0.85, 'M_sphere (720)', fontsize=8)

    # (b) drone-swarm threat — mesh
    ax = axes[1]
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.2, 1.2)
    ax.set_title('(b) Drone swarm → 2-D mesh', fontsize=9)
    xs, ys = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-0.6, 0.6, 5))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    ax.scatter(pts[:, 0], pts[:, 1], c='black', s=18)
    arrow(ax, (-1.3, 1.0), (-0.7, 0.5))
    arrow(ax, (-1.3, 0.7), (-0.7, 0.3))
    arrow(ax, (-1.3, 0.4), (-0.7, 0.0))
    ax.text(-1.4, 1.05, 'incoming swarm 730', fontsize=8)

    # (c) cluster munition — 3D cloud (here represented as scattered pts)
    ax = axes[2]
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.2, 1.2)
    ax.set_title('(c) Cluster munition → 3-D volume', fontsize=9)
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.8, 0.8, (45, 2))
    ax.scatter(pts[:, 0], pts[:, 1], c='black', s=14)
    arrow(ax, (-1.3, -0.9), (-0.5, -0.4))
    ax.text(-1.4, -1.05, 'submunitions 740', fontsize=8)

    # (d) artillery — ballistic-trajectory screen
    ax = axes[3]
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.2, 1.2)
    ax.set_title('(d) Artillery → ballistic screen', fontsize=9)
    # Ballistic arc
    t = np.linspace(0, 1, 30)
    x = -1.0 + 2*t; y = -0.7 + 2.5*t*(1-t)
    ax.plot(x, y, 'k--', lw=0.8)
    ax.scatter(x, y, c='black', s=14)
    arrow(ax, (-1.4, -0.9), (-0.95, -0.7))
    ax.text(-1.4, -1.05, 'incoming round 750', fontsize=8)

    fig.suptitle('FIG. 6  —  Threat-classification → manifold mappings',
                 fontsize=12, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, 'fig6_threat_manifolds.png'), dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG. 7: Sampling manifolds for scientific use
# --------------------------------------------------------------------------

def fig7():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # (a) atmospheric sampling column (vertical)
    ax = axes[0]
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 2.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('(a) Atmospheric column', fontsize=9)
    # Tower of agents at varying heights
    heights = np.linspace(0.1, 2.4, 14)
    xs = 0.05 * np.cos(np.arange(14))
    ax.scatter(xs, heights, c='black', s=22)
    ax.add_patch(Rectangle((-0.45, 0), 0.9, 0.05, color='#444'))
    ax.text(-0.55, 2.4, 'M_column (810)', fontsize=8)

    # (b) horizontal grid — aerial survey
    ax = axes[1]
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('(b) Survey grid', fontsize=9)
    xs, ys = np.meshgrid(np.linspace(-0.85, 0.85, 7), np.linspace(-0.85, 0.85, 7))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    ax.scatter(pts[:, 0], pts[:, 1], c='black', s=14)
    ax.text(-0.9, 0.9, 'M_grid (820)', fontsize=8)

    # (c) adaptive — tornado tracking
    ax = axes[2]
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('(c) Adaptive: tracking phenomenon', fontsize=9)
    # spiral around moving center, plus a future-position ring (dashed)
    t = np.linspace(0, 6*np.pi, 80)
    r = 0.05 + 0.04*t
    cx, cy = -0.2, -0.1
    ax.plot(cx + r*np.cos(t), cy + r*np.sin(t), color='#aaa', lw=0.6)
    # current ring
    ang = np.linspace(0, 2*np.pi, 16, endpoint=False)
    ring_now = np.column_stack([cx + 0.4*np.cos(ang), cy + 0.4*np.sin(ang)])
    ax.scatter(ring_now[:, 0], ring_now[:, 1], c='black', s=18)
    # forecast ring (dashed)
    cx2, cy2 = 0.3, 0.25
    ring_next = np.column_stack([cx2 + 0.4*np.cos(ang), cy2 + 0.4*np.sin(ang)])
    ax.add_patch(Circle((cx2, cy2), 0.4, fill=False, ec='#888', ls='--', lw=0.8))
    arrow(ax, (cx, cy), (cx2, cy2), color='#444', lw=1.0)
    ax.text(cx2 + 0.05, cy2 + 0.45, 'forecast (840)', fontsize=8)
    ax.text(-0.95, 0.9, 'M_adaptive (830)', fontsize=8)

    fig.suptitle('FIG. 7  —  Sampling manifolds for scientific applications',
                 fontsize=12, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, 'fig7_sampling_manifolds.png'), dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# FIG. 8: Localization via fiducial selection
# --------------------------------------------------------------------------

def fig8():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.2, 1.2)

    # Place a few high-confidence fiducials (filled, with concentric rings) and
    # a degraded agent that trilaterates against them.
    fiducials = np.array([[-0.9, 0.6], [0.9, 0.5], [0.0, -0.8]])
    for i, p in enumerate(fiducials):
        ax.add_patch(Circle(p, 0.06, color='black'))
        # range circle to the degraded agent
        d = np.linalg.norm(p)
        ax.add_patch(Circle(p, d, fill=False, ec='#888', ls='--', lw=0.7))
        ax.text(p[0] + 0.08, p[1] + 0.08, f'910 (fiducial F{i+1})', fontsize=8)

    # The degraded agent at origin
    ax.add_patch(Circle((0, 0), 0.06, fill=False, ec='black', lw=1.6))
    ax.text(0.08, 0.08, '920 (degraded agent)', fontsize=8)

    # Other broadcast participants (unselected)
    rng = np.random.default_rng(5)
    others = rng.uniform(-1.2, 1.2, (10, 2))
    ax.scatter(others[:, 0], others[:, 1], c='#aaa', s=20, marker='x')
    ax.text(-1.4, 1.05, 'other agents — broadcast confidence below threshold (930)',
            fontsize=8)
    ax.text(-1.4, -1.1, 'fiducial selection threshold ψ (940)', fontsize=8)

    fig.suptitle('FIG. 8  —  Localization via fiducial selection and trilateration',
                 fontsize=12, fontweight='bold', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, 'fig8_localization.png'), dpi=DPI)
    plt.close(fig)


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print("Wrote 8 figures to", OUT)


if __name__ == '__main__':
    main()
