# /// script
# dependencies = ["numpy<3", "matplotlib"]
# ###
"""Streaming-manifold benchmark: continuous time-varying targets.

Simulates a mocap-style scenario where the target manifold updates
continuously rather than via discrete phase transitions. The manifold
M(t) morphs between known shapes over time (sphere → torus → cube →
star, with smooth interpolation between them). At each tick the
algorithm sees the current M(t), rebuilds the PCA tree, and re-derives
target positions.

Two regimes:
  - Slow morph: full cycle every 60 seconds (drones easily track)
  - Fast morph: full cycle every 5 seconds (tracking lag becomes visible)

Metrics:
  - Tracking lag: mean ‖drone true_pos − M(t) target‖ over time
  - Re-derivation latency: wall time per ASSIGN call as N grows

The architecture's value here is that the same primitive that handled
discrete phase transitions in `simulator.py` handles continuous updates
without any modification — same PCA tree per frame, same deterministic
selection, same drone steering. The only operational change is that
the target_pos array updates every frame instead of every phase.
"""

import os
import numpy as np
import time

NUM_DRONES = int(os.environ.get("N", "100"))
DRONE_TICK = 0.04
APPROACH_RADIUS = 4.0
REPULSION_RADIUS = 3.5
MAX_TICKS = int(os.environ.get("MAX_TICKS", "1500"))
CYCLE_SECONDS = float(os.environ.get("CYCLE", "30.0"))
MORPH_RATE = 1.0 / CYCLE_SECONDS
WORLD = 40.0
N_SEEDS = int(os.environ.get("N_SEEDS", "2"))


def make_sphere(n, radius=15, center=(0, 0, 20)):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    cx, cy, cz = center
    return np.column_stack((
        radius * np.cos(theta) * np.sin(phi) + cx,
        radius * np.sin(theta) * np.sin(phi) + cy,
        radius * np.cos(phi) + cz,
    ))


_PLASTIC = 1.32471795724474602596


def _r2(i):
    a1, a2 = 1.0 / _PLASTIC, 1.0 / (_PLASTIC ** 2)
    return (0.5 + a1 * i) % 1.0, (0.5 + a2 * i) % 1.0


def make_torus(n, R=12, r=5, center=(0, 0, 20)):
    cx, cy, cz = center
    pts = np.empty((n, 3))
    for i in range(n):
        u, v = _r2(i)
        theta, phi = 2 * np.pi * u, 2 * np.pi * v
        pts[i] = [
            (R + r * np.cos(phi)) * np.cos(theta) + cx,
            (R + r * np.cos(phi)) * np.sin(theta) + cy,
            r * np.sin(phi) + cz,
        ]
    return pts


def make_cube_shell(n, size=16, center=(0, 0, 20)):
    cx, cy, cz = center
    half = size / 2
    side = 4
    cell = size / side

    def m(face, u, v):
        return [(half, u, v), (-half, u, v), (u, half, v),
                (u, -half, v), (u, v, half), (u, v, -half)][face]

    base, extra = n // 6, n - (n // 6) * 6
    positions = []
    for face in range(6):
        count = base + (1 if face < extra else 0)
        grid = []
        for row in range(side):
            cols = range(side) if row % 2 == 0 else range(side - 1, -1, -1)
            for col in cols:
                u = -half + (row + 0.5) * cell
                v = -half + (col + 0.5) * cell
                grid.append((u, v))
        grid.append((0.0, 0.0))
        for u, v in grid[:count]:
            p = m(face, u, v)
            positions.append([p[0] + cx, p[1] + cy, p[2] + cz])
    return np.array(positions[:n])


def make_star_3d(n, outer_r=20, inner_r=9, depth=8, center=(0, 0, 20)):
    cx, cy, cz = center

    def star_point(t):
        seg = t * 10
        seg_idx = int(seg) % 10
        seg_frac = seg - int(seg)
        ao = lambda k: np.pi / 2 + (k % 5) * 2 * np.pi / 5
        ai = lambda k: np.pi / 2 + (k + 0.5) * 2 * np.pi / 5
        if seg_idx % 2 == 0:
            a1, r1 = ao(seg_idx // 2), outer_r
            a2, r2 = ai(seg_idx // 2), inner_r
        else:
            a1, r1 = ai(seg_idx // 2), inner_r
            a2, r2 = ao(seg_idx // 2 + 1), outer_r
        x = r1 * np.cos(a1) * (1 - seg_frac) + r2 * np.cos(a2) * seg_frac
        y = r1 * np.sin(a1) * (1 - seg_frac) + r2 * np.sin(a2) * seg_frac
        return x, y

    n_layers = max(2, int(np.ceil(depth / 2.0)))
    pts_per_layer = int(np.ceil(n / n_layers))
    positions = []
    for layer in range(n_layers):
        z_off = -depth / 2 + depth * layer / max(1, n_layers - 1)
        for i in range(pts_per_layer):
            if len(positions) >= n:
                break
            t = (i + 0.5 * (layer % 2)) / pts_per_layer
            x, y = star_point(t)
            positions.append([x + cx, y + cy, cz + z_off])
    return np.array(positions[:n])


SHAPES = [make_sphere, make_torus, make_cube_shell, make_star_3d]


def streaming_manifold(t, n):
    """Continuously interpolate among the four shapes as t cycles 0→1.
    Returns a manifold of n points at time t."""
    phase = (t * MORPH_RATE) % 1.0
    n_shapes = len(SHAPES)
    pos_in_cycle = phase * n_shapes
    src_idx = int(pos_in_cycle) % n_shapes
    dst_idx = (src_idx + 1) % n_shapes
    alpha = pos_in_cycle - int(pos_in_cycle)
    src = SHAPES[src_idx](n)
    dst = SHAPES[dst_idx](n)
    return src * (1 - alpha) + dst * alpha


class ManifoldNode:
    def __init__(self, positions, depth=0):
        self.positions = np.array(positions)
        self.center = np.mean(positions, axis=0)
        self.depth = depth
        self.split_axis = None
        self.left = self.right = None
        if len(positions) > 1:
            self._split()

    def _split(self):
        c = self.positions - self.center
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        self.split_axis = Vt[0]
        proj = c @ self.split_axis
        order = np.argsort(proj, kind='stable')
        mid = len(order) // 2
        self.left = ManifoldNode(self.positions[order[:mid]], self.depth + 1)
        self.right = ManifoldNode(self.positions[order[mid:]], self.depth + 1)


def compute_target(my_id, drones, root):
    node = root
    cur = list(drones)
    my_pos = next(np.array(d['pos']) for d in drones if d['id'] == my_id)
    while node.left is not None and len(cur) > 1:
        n = len(cur)
        nl = len(node.left.positions)
        nt = len(node.positions)
        dl = max(0, min(n, int(round(n * nl / nt))))
        if dl == 0:
            node = node.right; continue
        if dl == n:
            node = node.left; continue
        positions = np.array([d['pos'] for d in cur])
        proj = positions @ node.split_axis
        order = np.argsort(proj, kind='stable')
        groups = [
            [cur[order[i]] for i in range(dl)],
            [cur[order[i]] for i in range(dl, n)],
        ]
        subs = [node.left, node.right]
        for i, group in enumerate(groups):
            if any(d['id'] == my_id for d in group):
                node = subs[i]; cur = group; break
    while node.left is not None:
        dl_ = np.linalg.norm(my_pos - node.left.center)
        dr_ = np.linalg.norm(my_pos - node.right.center)
        node = node.left if dl_ <= dr_ else node.right
    if len(node.positions) == 1:
        return node.positions[0]
    return node.center


def assign_all(positions, current_targets):
    """Compute target position for every drone given current manifold."""
    n = len(positions)
    tree = ManifoldNode(current_targets)
    drones = [{'id': i, 'pos': positions[i].copy()} for i in range(n)]
    target_pos = np.zeros((n, 3))
    for i in range(n):
        target_pos[i] = compute_target(i, drones, tree)
    return target_pos


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    n = NUM_DRONES
    starts = rng.uniform(-WORLD, WORLD, (n, 3))
    pos = starts.copy()
    vel = np.zeros((n, 3))
    last_dist = np.full(n, np.inf)
    stall = np.zeros(n, dtype=int)

    measurements = []
    REASSIGN_INTERVAL = 5  # ticks between reassignments (saves cost)

    for tick in range(MAX_TICKS):
        t = tick * DRONE_TICK
        if tick % REASSIGN_INTERVAL == 0:
            t0 = time.perf_counter()
            current_targets = streaming_manifold(t, n)
            target_pos = assign_all(pos, current_targets)
            assign_dt = time.perf_counter() - t0

        # Steering
        new_vel = np.zeros((n, 3))
        for i in range(n):
            tgt = target_pos[i]
            diff = tgt - pos[i]
            dist = float(np.linalg.norm(diff))
            is_final = dist < APPROACH_RADIUS
            attr = (diff / dist) * min(0.6, dist * 0.1) if dist > 0 else np.zeros(3)
            er = REPULSION_RADIUS * 0.4 if is_final else REPULSION_RADIUS
            rep = np.zeros(3)
            for j in range(n):
                if j == i:
                    continue
                d = pos[i] - pos[j]
                r = float(np.linalg.norm(d))
                if 0 < r < er:
                    unit = d / r
                    closing = max(0.0, -np.dot(vel[i] - vel[j], unit))
                    force = ((er - r) / r) * (1.0 + closing * 2.0)
                    rep += unit * force * 0.15
            v = attr + rep
            if is_final:
                progress = last_dist[i] - dist
                if abs(progress) < 0.01 and dist > 0.3:
                    stall[i] += 1
                else:
                    stall[i] = max(0, stall[i] - 5)
                last_dist[i] = dist
                if stall[i] > 10:
                    fade = max(0.0, 1.0 - (stall[i] - 10) / 40.0)
                    rep *= fade
                    v = attr + rep
            s_norm = float(np.linalg.norm(v))
            max_speed = 0.8
            if is_final and dist < 3.0:
                max_speed = max(0.05, dist * 0.2)
            if s_norm > max_speed:
                v = (v / s_norm) * max_speed
            new_vel[i] = v

        vel = new_vel
        pos += vel

        if tick % 25 == 0:
            tracking_err = float(np.mean(np.linalg.norm(pos - target_pos, axis=1)))
            measurements.append((t, tracking_err, assign_dt * 1000))

    return measurements


def main():
    print(f"Streaming manifold benchmark: N={NUM_DRONES}, "
          f"cycle={CYCLE_SECONDS}s "
          f"(rate={MORPH_RATE:.4f} cycles/s)\n")

    all_runs = []
    for seed in range(N_SEEDS):
        all_runs.append(simulate(seed))

    n_steps = min(len(r) for r in all_runs)
    times = np.array([all_runs[0][i][0] for i in range(n_steps)])
    track = np.array([[all_runs[s][i][1] for s in range(N_SEEDS)] for i in range(n_steps)])
    assign_t = np.array([[all_runs[s][i][2] for s in range(N_SEEDS)] for i in range(n_steps)])

    print(f"Mean ASSIGN call latency: {assign_t.mean():.2f} ms")
    print(f"Median tracking error (steady-state, t > 5s): "
          f"{np.median(track[times > 5]):.2f} m")
    print(f"Worst-case tracking error: {track.max():.2f} m\n")

    print(f"{'t_s':>6} {'tracking_err_m':>16}")
    step = max(1, len(times) // 12)
    for i in range(0, len(times), step):
        print(f"{times[i]:>6.1f} {track[i].mean():>16.3f}")


if __name__ == '__main__':
    main()
