# /// script
# dependencies = [
#   "numpy<3",
#   "matplotlib",
#   "scipy",
# ]
# ///
"""
Decentralized Drone Swarm — Multi-Manifold Phase Transitions

Extends the hierarchical divide-and-conquer formation algorithm to cycle
through a sequence of target manifolds. Each phase forms, holds for the
same duration that formation took, then transitions deterministically to
the next manifold.

Phase transition trick:
  When all drones are locked, the broadcast is quiescent — every read
  returns identical positions. So each drone can independently snapshot
  the broadcast during quiescence and recompute waypoints against the
  next manifold's tree, and the snapshots will agree across drones. The
  deterministic-consensus property carries over for free, with no extra
  signaling beyond the existing telemetry channel.

Per-drone phase machine:
  - phase_start_time = monotonic time at the start of this phase
  - all_locked_observed_at = first time this drone saw every drone locked
    in the current phase (None until then)
  - hold_duration = all_locked_observed_at - phase_start_time
  - snapshot_positions = positions captured at the all-locked moment
  - When (now - all_locked_observed_at) >= hold_duration, the drone
    advances phase_idx, recomputes its path against PHASE_TREES[next],
    and unlocks.

Run interactive:
    python3 simulator.py

Render video (requires ffmpeg):
    SAVE_VIDEO=1 VIDEO_PATH=swarm.mp4 python3 simulator.py
"""

import os
import sys
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters ---
NUM_DRONES = 100
DRONE_TICK = 0.04
APPROACH_RADIUS = 4.0  # within this distance of leaf, switch to "final approach" mode
REPULSION_RADIUS = 3.5
MIN_SEPARATION = 2.0

SAVE_VIDEO = os.environ.get("SAVE_VIDEO", "0") == "1"
VIDEO_PATH = os.environ.get("VIDEO_PATH", "swarm.mp4")
VIDEO_FPS = 25
RANDOM_SEED = int(os.environ.get("SEED", "42"))


def validate_manifold(targets, label, min_sep=MIN_SEPARATION):
    from scipy.spatial.distance import pdist
    dists = pdist(targets)
    min_dist = float(dists.min())
    n_violations = int(np.sum(dists < min_sep))
    status = "OK" if n_violations == 0 else f"FAIL ({n_violations} pairs < {min_sep})"
    print(f"  [{label:>6}] {len(targets)} targets, min pairwise = {min_dist:.3f}  {status}")
    return n_violations == 0


# --- Manifolds ---
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


# Roberts' R2 plastic-ratio low-discrepancy sequence — gives well-spread
# 2D points without grid artifacts or single-helix streaking.
_PLASTIC = 1.32471795724474602596
_R2_A1 = 1.0 / _PLASTIC
_R2_A2 = 1.0 / (_PLASTIC * _PLASTIC)


def _r2_uv(i):
    return (0.5 + _R2_A1 * i) % 1.0, (0.5 + _R2_A2 * i) % 1.0


def make_torus(n, R=12, r=5, center=(0, 0, 20)):
    cx, cy, cz = center
    positions = np.empty((n, 3))
    for i in range(n):
        u, v = _r2_uv(i)
        theta = 2 * np.pi * u
        phi = 2 * np.pi * v
        positions[i] = [
            (R + r * np.cos(phi)) * np.cos(theta) + cx,
            (R + r * np.cos(phi)) * np.sin(theta) + cy,
            r * np.sin(phi) + cz,
        ]
    return positions


def make_cube_shell(n, size=16, center=(0, 0, 20)):
    """Balanced 4x4 grid per face, plus a face-center for the 17th drone.

    For n=100 (4 remainder): 4 faces get 17 points (4x4 + center), 2 faces
    get 16 points. Grid spacing = size/4 so all in-face neighbors are
    spaced size/4, and corner-adjacent cross-face points are size/4 * sqrt(2).
    """
    cx, cy, cz = center
    half = size / 2
    side = 4
    cell = size / side

    def map_face(face, u, v):
        if face == 0: return (half, u, v)
        if face == 1: return (-half, u, v)
        if face == 2: return (u, half, v)
        if face == 3: return (u, -half, v)
        if face == 4: return (u, v, half)
        return (u, v, -half)

    base = n // 6
    extra = n - base * 6

    positions = []
    for face in range(6):
        count = base + (1 if face < extra else 0)
        # 4x4 grid laid out in zigzag rows so partial counts (count<16)
        # spread the points instead of bunching on the bottom rows.
        grid = []
        for row in range(side):
            cols = range(side) if row % 2 == 0 else range(side - 1, -1, -1)
            for col in cols:
                u = -half + (row + 0.5) * cell
                v = -half + (col + 0.5) * cell
                grid.append((u, v))
        # Add the face center at the end so it only appears for count==17.
        grid.append((0.0, 0.0))
        for u, v in grid[:count]:
            p = map_face(face, u, v)
            positions.append([p[0] + cx, p[1] + cy, p[2] + cz])
    return np.array(positions[:n])


def make_star_3d(n, outer_r=20, inner_r=9, depth=8, center=(0, 0, 20)):
    cx, cy, cz = center

    def star_point(t):
        seg = t * 10
        seg_idx = int(seg) % 10
        seg_frac = seg - int(seg)
        angle_out = lambda k: np.pi / 2 + (k % 5) * 2 * np.pi / 5
        angle_in = lambda k: np.pi / 2 + (k + 0.5) * 2 * np.pi / 5
        if seg_idx % 2 == 0:
            a1, r1 = angle_out(seg_idx // 2), outer_r
            a2, r2 = angle_in(seg_idx // 2), inner_r
        else:
            a1, r1 = angle_in(seg_idx // 2), inner_r
            a2, r2 = angle_out(seg_idx // 2 + 1), outer_r
        x = r1 * np.cos(a1) * (1 - seg_frac) + r2 * np.cos(a2) * seg_frac
        y = r1 * np.sin(a1) * (1 - seg_frac) + r2 * np.sin(a2) * seg_frac
        return x, y

    n_layers = max(2, int(np.ceil(depth / MIN_SEPARATION)))
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


# Sequence of manifolds. Drones cycle through these in order.
PHASES = [
    ("sphere", make_sphere(NUM_DRONES)),
    ("torus",  make_torus(NUM_DRONES)),
    ("cube",   make_cube_shell(NUM_DRONES)),
    ("star",   make_star_3d(NUM_DRONES)),
]
N_PHASES = len(PHASES)

print("=== Manifold Validation ===")
for label, targets in PHASES:
    validate_manifold(targets, label)
print()


class ManifoldNode:
    """Binary tree decomposition of a target manifold via PCA splits."""

    def __init__(self, positions, depth=0):
        self.positions = np.array(positions)
        self.center = np.mean(positions, axis=0)
        self.depth = depth
        self.split_axis = None
        self.left = None
        self.right = None
        if len(positions) > 1:
            self._split()

    def _split(self):
        centered = self.positions - self.center
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        self.split_axis = Vt[0]
        proj = centered @ self.split_axis
        order = np.argsort(proj, kind='stable')
        mid = len(order) // 2
        self.left = ManifoldNode(self.positions[order[:mid]], self.depth + 1)
        self.right = ManifoldNode(self.positions[order[mid:]], self.depth + 1)


# Trees are deterministic — every drone would build identical trees from
# identical inputs. Precomputing once is just an optimization.
PHASE_TREES = [ManifoldNode(targets) for _, targets in PHASES]


def compute_leaf_target(my_id, drones, root):
    """Recursive divide-and-conquer assignment (strict-mode).

    At each tree node, partition `drones` by projection onto the node's
    split axis, sized proportionally to the target sub-manifold counts.
    Drone partitions are paired straight to same-side target subtrees
    (no cross-swap — the swap heuristic was removed because it can break
    the count invariant nl-drones → nl-targets and lets two drones
    descend to the same leaf in lopsided subtrees; see PROOFS.md).
    Returns my single assigned leaf position — no intermediate waypoints.
    The drone steers continuously toward this point; closed-loop control
    handles the trajectory.
    """
    node = root
    current_drones = list(drones)

    my_pos = None
    for d in drones:
        if d['id'] == my_id:
            my_pos = np.array(d['pos'])
            break

    while node.left is not None and len(current_drones) > 1:
        n = len(current_drones)
        n_left = len(node.left.positions)
        n_total = len(node.positions)
        d_left = max(0, min(n, int(round(n * n_left / n_total))))

        if d_left == 0:
            node = node.right
            continue
        if d_left == n:
            node = node.left
            continue

        positions = np.array([d['pos'] for d in current_drones])
        proj = positions @ node.split_axis
        order = np.argsort(proj, kind='stable')

        groups = [
            [current_drones[order[i]] for i in range(d_left)],
            [current_drones[order[i]] for i in range(d_left, n)],
        ]
        # Match the projection-half drone groups straight to the same-side
        # target subtrees. We do NOT swap on cost-crossed: that heuristic
        # saves <0.5% on total path length empirically, and breaks the
        # count invariant nl-drones->nl-targets, which lets two drones
        # eventually descend to the same leaf in lopsided subtrees. Strict
        # straight pairing keeps the assignment bijective.
        subs = [node.left, node.right]

        for i, group in enumerate(groups):
            if any(d['id'] == my_id for d in group):
                node = subs[i]
                current_drones = group
                break

    while node.left is not None:
        dl = np.linalg.norm(my_pos - node.left.center)
        dr = np.linalg.norm(my_pos - node.right.center)
        node = node.left if dl <= dr else node.right

    if len(node.positions) == 1:
        return node.positions[0].copy()
    return node.center.copy()


# ==========================================================================
# Shared broadcast — radio-style telemetry. Each drone owns one slot.
# ==========================================================================

class Broadcast:
    def __init__(self, n):
        self.n = n
        self.positions = np.zeros((n, 3))
        self.velocities = np.zeros((n, 3))
        self.locked = np.zeros(n, dtype=bool)
        self.wp_idx = np.zeros(n, dtype=int)
        self.wp_count = np.zeros(n, dtype=int)
        self.phase_idx = np.zeros(n, dtype=int)
        self.done = np.zeros(n, dtype=bool)
        self._lock = threading.Lock()

    def read(self):
        with self._lock:
            return (
                self.positions.copy(),
                self.velocities.copy(),
                self.locked.copy(),
                self.wp_idx.copy(),
                self.wp_count.copy(),
                self.phase_idx.copy(),
                self.done.copy(),
            )

    def write(self, drone_id, pos, vel, locked, wp_idx, wp_count, phase_idx, done):
        with self._lock:
            self.positions[drone_id] = pos
            self.velocities[drone_id] = vel
            self.locked[drone_id] = locked
            self.wp_idx[drone_id] = wp_idx
            self.wp_count[drone_id] = wp_count
            self.phase_idx[drone_id] = phase_idx
            self.done[drone_id] = done


# ==========================================================================
# DroneAgent — independent thread, reads broadcast, computes, publishes.
# ==========================================================================

class DroneAgent(threading.Thread):
    def __init__(self, drone_id, start_pos, initial_broadcast, shared):
        super().__init__(daemon=True)
        self.drone_id = drone_id
        self.pos = np.array(start_pos, dtype=float)
        self.velocity = np.zeros(3)
        self.locked = False
        self.shared = shared
        self.running = True
        self.stall_count = 0
        self.last_dist = float('inf')
        self.done = False

        self.phase_idx = 0
        self.phase_start_time = time.monotonic()
        self.all_locked_observed_at = None
        self.hold_duration = None
        self._snapshot = None

        self.target = compute_leaf_target(drone_id, initial_broadcast, PHASE_TREES[0])

        self.shared.write(
            drone_id, self.pos, self.velocity, False,
            0, 1, self.phase_idx, False,
        )

    def _begin_next_phase(self):
        self.phase_idx += 1
        next_initial = [{'id': i, 'pos': p.copy()} for i, p in enumerate(self._snapshot)]
        self.target = compute_leaf_target(self.drone_id, next_initial, PHASE_TREES[self.phase_idx])
        self.locked = False
        self.stall_count = 0
        self.last_dist = float('inf')
        self.phase_start_time = time.monotonic()
        self.all_locked_observed_at = None
        self.hold_duration = None
        self._snapshot = None

    def run(self):
        while self.running and not self.done:
            all_pos, all_vel, all_locked, _, _, _, _ = self.shared.read()

            # --- Hold / transition logic (only when locked) ---
            if self.locked:
                # Latch the all-locked moment + a quiescent snapshot.
                if self.all_locked_observed_at is None and bool(np.all(all_locked)):
                    self.all_locked_observed_at = time.monotonic()
                    self._snapshot = all_pos.copy()
                    self.hold_duration = self.all_locked_observed_at - self.phase_start_time
                    if self.drone_id == 0:
                        print(
                            f"[phase {self.phase_idx} {PHASES[self.phase_idx][0]:>6}] "
                            f"formed in {self.hold_duration:.2f}s, holding {self.hold_duration:.2f}s",
                            flush=True,
                        )

                if self.all_locked_observed_at is not None:
                    elapsed = time.monotonic() - self.all_locked_observed_at
                    if elapsed >= self.hold_duration:
                        if self.phase_idx + 1 < N_PHASES:
                            self._begin_next_phase()
                        else:
                            self.done = True
                            self.shared.write(
                                self.drone_id, self.pos, self.velocity, True,
                                0, 1, self.phase_idx, True,
                            )
                            break

                if self.locked:  # still holding
                    self.shared.write(
                        self.drone_id, self.pos, self.velocity, True,
                        0, 1, self.phase_idx, False,
                    )
                    time.sleep(DRONE_TICK)
                    continue

            # --- Navigation: steer continuously toward the assigned leaf. ---
            target = self.target
            diff = target - self.pos
            dist = np.linalg.norm(diff)

            is_final = dist < APPROACH_RADIUS

            attr = (diff / dist) * min(0.6, dist * 0.1) if dist > 0 else np.zeros(3)

            effective_repulsion = REPULSION_RADIUS * 0.4 if is_final else REPULSION_RADIUS
            rep = np.zeros(3)
            for j in range(self.shared.n):
                if j == self.drone_id or all_locked[j]:
                    continue
                d = self.pos - all_pos[j]
                r = np.linalg.norm(d)
                if 0 < r < effective_repulsion:
                    unit = d / r
                    closing = max(0.0, -np.dot(self.velocity - all_vel[j], unit))
                    force = ((effective_repulsion - r) / r) * (1.0 + closing * 2.0)
                    rep += unit * force * 0.15

            v = attr + rep

            if is_final:
                progress = self.last_dist - dist
                if abs(progress) < 0.01 and dist > 0.3:
                    self.stall_count += 1
                else:
                    self.stall_count = max(0, self.stall_count - 5)
                self.last_dist = dist
                if self.stall_count > 10:
                    fade = max(0.0, 1.0 - (self.stall_count - 10) / 40.0)
                    rep *= fade
                    v = attr + rep

            s = np.linalg.norm(v)
            max_speed = 0.8
            if is_final and dist < 3.0:
                max_speed = max(0.05, dist * 0.2)
            if s > max_speed:
                v = (v / s) * max_speed

            self.velocity = v
            self.pos += v

            if is_final and dist < 0.3:
                self.pos = target.copy()
                self.velocity = np.zeros(3)
                self.locked = True

            self.shared.write(
                self.drone_id, self.pos, self.velocity, self.locked,
                0, 1, self.phase_idx, False,
            )

            time.sleep(DRONE_TICK)

    def stop(self):
        self.running = False


# ==========================================================================
# Bootstrap
# ==========================================================================

np.random.seed(RANDOM_SEED)
initial_positions = np.random.uniform(-40, 40, (NUM_DRONES, 3))
initial_broadcast = [{'id': i, 'pos': pos.copy()} for i, pos in enumerate(initial_positions)]

shared = Broadcast(NUM_DRONES)
agents = [DroneAgent(i, initial_positions[i], initial_broadcast, shared) for i in range(NUM_DRONES)]

for agent in agents:
    agent.start()


# ==========================================================================
# Display
# ==========================================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
ax.set_xlim([-40, 40]); ax.set_ylim([-40, 40]); ax.set_zlim([0, 50])

scatter = ax.scatter(
    initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2],
    c='cyan', marker='o', s=40, edgecolors='k')

target_scatter = ax.scatter(
    PHASES[0][1][:, 0], PHASES[0][1][:, 1], PHASES[0][1][:, 2],
    c='red', marker='x', s=20, alpha=0.25)

title = ax.set_title("")
frame_count = [0]
last_phase_shown = [0]


def display_update(_):
    positions, _, locked_flags, _, _, phase_indices, done_flags = shared.read()

    # The displayed "current phase" is the most common phase across drones.
    phase_now = int(np.bincount(phase_indices, minlength=N_PHASES).argmax())
    if phase_now != last_phase_shown[0]:
        targets = PHASES[phase_now][1]
        target_scatter._offsets3d = (targets[:, 0], targets[:, 1], targets[:, 2])
        last_phase_shown[0] = phase_now

    colors = ['lime' if locked_flags[i] else 'cyan' for i in range(NUM_DRONES)]
    scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    scatter.set_facecolor(colors)

    locked_count = int(locked_flags.sum())
    done_count = int(done_flags.sum())
    frame_count[0] += 1
    label = PHASES[phase_now][0]
    title.set_text(
        f"Phase {phase_now+1}/{N_PHASES} ({label}) | "
        f"Locked {locked_count}/{NUM_DRONES} | "
        f"Done {done_count}/{NUM_DRONES} | "
        f"Frame {frame_count[0]}"
    )
    ax.view_init(elev=20, azim=frame_count[0] * 0.5)
    return scatter, target_scatter


if SAVE_VIDEO:
    # Record real-time snapshots at VIDEO_FPS, then render the buffered
    # frames to an mp4 offline. This decouples rendering speed from
    # simulation pacing and gives a clean 1:1 real-time video.
    print(f"Recording at {VIDEO_FPS} fps ...")
    plt.close(fig)

    snapshots = []  # list of (positions, locked_flags, phase_indices, done_flags)
    sim_start = time.monotonic()
    timeout_s = 240.0

    next_capture = sim_start
    capture_dt = 1.0 / VIDEO_FPS
    while True:
        now = time.monotonic()
        if now < next_capture:
            time.sleep(min(0.005, next_capture - now))
            continue
        next_capture += capture_dt

        positions, _, locked_flags, _, _, phase_indices, done_flags = shared.read()
        snapshots.append((
            positions.copy(), locked_flags.copy(),
            phase_indices.copy(), done_flags.copy(),
        ))

        if done_flags.all():
            # A short tail so the final frame is visible.
            for _ in range(2 * VIDEO_FPS):
                positions, _, locked_flags, _, _, phase_indices, done_flags = shared.read()
                snapshots.append((
                    positions.copy(), locked_flags.copy(),
                    phase_indices.copy(), done_flags.copy(),
                ))
                time.sleep(capture_dt)
            break
        if now - sim_start > timeout_s:
            print(f"  timeout after {timeout_s}s, captured {len(snapshots)} frames")
            break

    for agent in agents:
        agent.stop()
    print(f"  captured {len(snapshots)} frames over {time.monotonic() - sim_start:.1f}s")

    # Render buffered frames offline.
    print(f"Rendering to {VIDEO_PATH} ...")
    rfig = plt.figure(figsize=(10, 8))
    rax = rfig.add_subplot(projection='3d')
    rax.set_xlim([-40, 40]); rax.set_ylim([-40, 40]); rax.set_zlim([0, 50])

    init_pos, init_locked, init_phase, _ = snapshots[0]
    rscatter = rax.scatter(
        init_pos[:, 0], init_pos[:, 1], init_pos[:, 2],
        c='cyan', marker='o', s=40, edgecolors='k')
    rtarget = rax.scatter(
        PHASES[0][1][:, 0], PHASES[0][1][:, 1], PHASES[0][1][:, 2],
        c='red', marker='x', s=20, alpha=0.25)
    rtitle = rax.set_title("")

    last_phase = [-1]

    def render_frame(idx):
        positions, locked_flags, phase_indices, done_flags = snapshots[idx]
        phase_now = int(np.bincount(phase_indices, minlength=N_PHASES).argmax())
        if phase_now != last_phase[0]:
            targets = PHASES[phase_now][1]
            rtarget._offsets3d = (targets[:, 0], targets[:, 1], targets[:, 2])
            last_phase[0] = phase_now

        colors = ['lime' if locked_flags[i] else 'cyan' for i in range(NUM_DRONES)]
        rscatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        rscatter.set_facecolor(colors)

        label = PHASES[phase_now][0]
        locked_count = int(locked_flags.sum())
        done_count = int(done_flags.sum())
        rtitle.set_text(
            f"Phase {phase_now+1}/{N_PHASES} ({label})  |  "
            f"Locked {locked_count}/{NUM_DRONES}  |  "
            f"Done {done_count}/{NUM_DRONES}  |  "
            f"t={idx / VIDEO_FPS:.1f}s"
        )
        rax.view_init(elev=20, azim=idx * 0.5)
        return rscatter, rtarget

    writer = animation.FFMpegWriter(fps=VIDEO_FPS, bitrate=4000)
    rani = animation.FuncAnimation(
        rfig, render_frame, frames=len(snapshots),
        interval=1000 // VIDEO_FPS, blit=False,
    )
    rani.save(VIDEO_PATH, writer=writer)
    print(f"Saved {VIDEO_PATH} ({len(snapshots) / VIDEO_FPS:.1f}s)")
    sys.exit(0)


ani = animation.FuncAnimation(fig, display_update, interval=40, blit=False)

try:
    plt.show()
finally:
    for agent in agents:
        agent.stop()

    positions, _, locked_flags, _, _, phase_indices, done_flags = shared.read()
    print(f"\n=== Final State ===")
    print(f"  Drones done:   {int(done_flags.sum())}/{NUM_DRONES}")
    print(f"  Drones locked: {int(locked_flags.sum())}/{NUM_DRONES}")
    print(f"  Phases reached: {np.bincount(phase_indices, minlength=N_PHASES)}")
