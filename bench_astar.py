# /// script
# dependencies = ["numpy<3"]
# ///
"""Swarm-A* exploration bench.

Distributed real-time A* over manifold transitions. The swarm starts
knowing only its initial position and a goal coordinate; the cost
field between is unknown. Each iteration:

  1. Every drone runs the same A* over the same broadcast map
     (unknown cells use an optimistic baseline cost).
  2. Take the first STEP_CELLS of the returned path -> next manifold.
  3. Drones transit current -> next centre with independent random
     walks (Mode B), sensing the cost field as they go.
  4. Sensed cells broadcast and the shared map updates.
  5. Repeat until centroid enters the goal disk OR a termination
     condition fires (max-iterations, timeout, or no-progress).

OPERATIONAL ENGINEERING NOTES
- Per-iteration progress is logged to stderr, flushed.
- Each run has a hard wall-clock budget; runs that exceed it return
  status='timeout' with partial state preserved.
- A no-progress detector aborts runs where the centroid hasn't moved
  closer to the goal in NO_PROGRESS_PATIENCE iterations.
- Results are checkpointed after every (env, seed) pair via an atomic
  rename. Re-running the bench picks up where it left off.
- Every run returns a status taxonomy: reached / max_iter / timeout /
  stuck / error:<msg>. Aggregates print at the end.
"""

import heapq
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np

# ---------------------------------------------------------------- config

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "100"))
N_DRONES = int(os.environ.get("N_DRONES", "20"))
N_SEEDS = int(os.environ.get("N_SEEDS", "20"))
STEP_CELLS = int(os.environ.get("STEP_CELLS", "5"))
WALK_SIGMA = float(os.environ.get("WALK_SIGMA", "1.0"))
SENSE_R = float(os.environ.get("SENSE_R", "2.5"))
SENSE_NOISE = 0.05
BASELINE_COST = 0.05
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "60"))
GOAL_TOLERANCE = float(os.environ.get("GOAL_TOLERANCE", str(STEP_CELLS)))
TRANSIT_SUBSTEPS = 8
RUN_TIMEOUT_S = float(os.environ.get("RUN_TIMEOUT_S", "15.0"))
NO_PROGRESS_PATIENCE = int(os.environ.get("NO_PROGRESS_PATIENCE", "6"))
PROGRESS_EPS = 0.5  # cells; "made progress" if d_to_goal dropped by this much
VERBOSE = os.environ.get("VERBOSE", "1") != "0"

assert WORLD_SIZE >= 30
assert 1 <= N_DRONES <= 200
assert STEP_CELLS >= 1
assert TRANSIT_SUBSTEPS >= 1
assert RUN_TIMEOUT_S > 0


def log(msg):
    if VERBOSE:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


# ----------------------------------------------------------------- world

def make_world(seed: int, n_threats: int = 3,
               threat_min: float = 0.5, threat_max: float = 1.0,
               sigma_min: float = 5.0, sigma_max: float = 12.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cx = rng.uniform(20, WORLD_SIZE - 20, size=n_threats)
    cy = rng.uniform(20, WORLD_SIZE - 20, size=n_threats)
    sigmas = rng.uniform(sigma_min, sigma_max, size=n_threats)
    mags = rng.uniform(threat_min, threat_max, size=n_threats)

    xs = np.arange(WORLD_SIZE)
    ys = np.arange(WORLD_SIZE)
    xx, yy = np.meshgrid(xs, ys)
    cost = np.full((WORLD_SIZE, WORLD_SIZE), BASELINE_COST)
    for x0, y0, s, m in zip(cx, cy, sigmas, mags):
        d2 = (xx - x0) ** 2 + (yy - y0) ** 2
        cost += m * np.exp(-d2 / (2.0 * s * s))
    assert cost.shape == (WORLD_SIZE, WORLD_SIZE)
    return cost


# ----------------------------------------------------------------- A* core

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]
NEIGHBOUR_D = [float(np.hypot(dx, dy)) for dx, dy in NEIGHBOURS]


def grid_astar(cost_map: np.ndarray, start, goal,
               baseline: float = BASELINE_COST,
               max_expansions: int = 50000):
    """A* over the 2D grid (8-connected). Unknown cells (NaN) -> baseline.

    Hard cap on expansions to fail fast on pathological inputs.
    Returns (path_or_None, total_cost, expansions).
    """
    H, W = cost_map.shape
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
        return None, float("inf"), 0

    def h(x, y):
        return float(np.hypot(x - gx, y - gy)) * baseline

    open_pq = [(h(sx, sy), 0.0, sx, sy)]
    came_from = {}
    g_score = {(sx, sy): 0.0}
    closed = set()
    expansions = 0

    while open_pq:
        if expansions >= max_expansions:
            return None, float("inf"), expansions
        f, g, x, y = heapq.heappop(open_pq)
        if (x, y) in closed:
            continue
        closed.add((x, y))
        expansions += 1
        if (x, y) == (gx, gy):
            path = [(x, y)]
            while (x, y) in came_from:
                x, y = came_from[(x, y)]
                path.append((x, y))
            path.reverse()
            return path, g_score[(gx, gy)], expansions
        for (dx, dy), d in zip(NEIGHBOURS, NEIGHBOUR_D):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if (nx, ny) in closed:
                continue                              # never re-push closed cells
            cell_cost = cost_map[ny, nx]
            if np.isnan(cell_cost):
                cell_cost = baseline
            cell_cost = max(0.0, cell_cost)            # defensive: no negative costs
            ng = g + cell_cost * d
            if ng < g_score.get((nx, ny), float("inf")):
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = (x, y)
                heapq.heappush(open_pq, (ng + h(nx, ny), ng, nx, ny))
    return None, float("inf"), expansions


# ----------------------------------------------------------------- sensing

def sense(cost_field: np.ndarray, pos, footprint_r: float, rng) -> dict:
    samples = {}
    x, y = pos
    R = int(np.ceil(footprint_r))
    H, W = cost_field.shape
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            if dx * dx + dy * dy > footprint_r * footprint_r:
                continue
            cx, cy = int(x) + dx, int(y) + dy
            if 0 <= cx < W and 0 <= cy < H:
                # Clip to non-negative: negative costs would let A* find
                # spurious "cheap" detours through noisy cells, blowing up
                # heap size and expansions.
                v = float(cost_field[cy, cx]) + float(rng.normal(0, SENSE_NOISE))
                samples[(cx, cy)] = max(0.0, v)
    return samples


def update_map(broadcast_map, visited_mask, samples):
    for (cx, cy), v in samples.items():
        if visited_mask[cy, cx]:
            broadcast_map[cy, cx] = 0.5 * (broadcast_map[cy, cx] + v)
        else:
            broadcast_map[cy, cx] = v
            visited_mask[cy, cx] = True


def manifold_targets(centre, n_drones, radius=2.5):
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    return np.stack([centre[0] + radius * np.cos(angles),
                     centre[1] + radius * np.sin(angles)], axis=1)


def integrate_true_cost(cost_field, p_from, p_to, n_pts: int = 16) -> float:
    dist = float(np.linalg.norm(p_to - p_from))
    if dist == 0.0:
        return 0.0
    ds = dist / n_pts
    H, W = cost_field.shape
    total = 0.0
    for t in np.linspace(0.0, 1.0, n_pts, endpoint=False) + 0.5 / n_pts:
        p = p_from + t * (p_to - p_from)
        cx = int(np.clip(p[0], 0, W - 1))
        cy = int(np.clip(p[1], 0, H - 1))
        total += cost_field[cy, cx] * ds
    return total


# ------------------------------------------------------------- algorithms

def run_swarm_astar(cost_field, start, goal, seed: int, label: str = "") -> dict:
    """Returns a dict with 'status' and any partial state collected so far."""
    t_start = time.time()
    rng = np.random.default_rng(seed)
    bm = np.full(cost_field.shape, np.nan)
    vm = np.zeros(cost_field.shape, dtype=bool)
    centre = np.array(start, dtype=float)
    goal_a = np.array(goal, dtype=float)
    drones = manifold_targets(centre, N_DRONES)

    for d in drones:
        update_map(bm, vm, sense(cost_field, d, SENSE_R, rng))

    path = [centre.copy()]
    cumulative_cost = 0.0
    consensus_ok = True
    iterations = 0
    status = "max_iter"
    best_d = float(np.linalg.norm(centre - goal_a))
    iters_since_progress = 0

    for it in range(MAX_ITERATIONS):
        iterations = it + 1
        elapsed = time.time() - t_start
        if elapsed > RUN_TIMEOUT_S:
            status = "timeout"
            log(f"      [{label} seed={seed}] TIMEOUT at iter {it}, elapsed={elapsed:.1f}s")
            break

        d_to_goal = float(np.linalg.norm(centre - goal_a))
        if d_to_goal <= GOAL_TOLERANCE:
            status = "reached"
            break

        if d_to_goal < best_d - PROGRESS_EPS:
            best_d = d_to_goal
            iters_since_progress = 0
        else:
            iters_since_progress += 1
            if iters_since_progress >= NO_PROGRESS_PATIENCE:
                status = "stuck"
                log(f"      [{label} seed={seed}] STUCK at iter {it}, d_to_goal={d_to_goal:.1f}, best={best_d:.1f}")
                break

        t_astar = time.time()
        plan_path, _, expansions = grid_astar(bm, centre, goal_a)
        t_astar = time.time() - t_astar
        if plan_path is None or len(plan_path) < 2:
            status = "error:no_path"
            log(f"      [{label} seed={seed}] NO PATH (expansions={expansions}, t_astar={t_astar*1000:.0f}ms)")
            break
        if t_astar > 2.0:
            log(f"      [{label} seed={seed}] iter={it} SLOW A* expansions={expansions} t={t_astar:.1f}s")

        idx = min(STEP_CELLS, len(plan_path) - 1)
        next_c = np.array(plan_path[idx], dtype=float)

        # Distributed-consensus byte-identity check on a sample of drones.
        # Every drone runs A* over the same broadcast map -> same path. We
        # spot-check 5 redundant calls; in practice this would be byte-equal
        # by construction (same input -> same output).
        for _ in range(min(5, N_DRONES) - 1):
            other_path, _, _ = grid_astar(bm, centre, goal_a)
            if other_path is None or idx >= len(other_path) or other_path[idx] != tuple(next_c.astype(int)):
                consensus_ok = False
                break

        starts = drones.copy()
        ends = manifold_targets(next_c, N_DRONES)
        for s in range(1, TRANSIT_SUBSTEPS + 1):
            t = s / TRANSIT_SUBSTEPS
            interp = (1.0 - t) * starts + t * ends
            drones = interp + rng.normal(0, WALK_SIGMA, size=interp.shape)
            for d in drones:
                update_map(bm, vm, sense(cost_field, d, SENSE_R, rng))

        cumulative_cost += integrate_true_cost(cost_field, centre, next_c)
        centre = next_c
        path.append(centre.copy())

        if VERBOSE:
            log(f"      [{label} seed={seed}] iter={it:>2} d_to_goal={d_to_goal:5.1f} "
                f"sensed={vm.mean()*100:4.1f}% cost={cumulative_cost:5.2f} "
                f"t={elapsed:.1f}s expansions={expansions}")

    return {
        "algorithm": "swarm_astar",
        "status": status,
        "cumulative_cost": float(cumulative_cost),
        "iterations": iterations,
        "reached_goal": status == "reached",
        "fraction_sensed": float(vm.mean()),
        "consensus_ok": bool(consensus_ok),
        "wall_time_s": float(time.time() - t_start),
        "final_d_to_goal": float(np.linalg.norm(centre - goal_a)),
        "path_len": len(path),
    }


def run_oracle_astar(cost_field, start, goal) -> dict:
    t_start = time.time()
    centre = np.array(start, dtype=float)
    goal_a = np.array(goal, dtype=float)
    path = [centre.copy()]
    cumulative_cost = 0.0
    iterations = 0
    status = "max_iter"

    for it in range(MAX_ITERATIONS):
        iterations = it + 1
        if time.time() - t_start > RUN_TIMEOUT_S:
            status = "timeout"
            break
        if np.linalg.norm(centre - goal_a) <= GOAL_TOLERANCE:
            status = "reached"
            break
        plan_path, _, _ = grid_astar(cost_field, centre, goal_a)
        if plan_path is None or len(plan_path) < 2:
            status = "error:no_path"
            break
        idx = min(STEP_CELLS, len(plan_path) - 1)
        next_c = np.array(plan_path[idx], dtype=float)
        cumulative_cost += integrate_true_cost(cost_field, centre, next_c)
        centre = next_c
        path.append(centre.copy())

    return {
        "algorithm": "oracle_astar",
        "status": status,
        "cumulative_cost": float(cumulative_cost),
        "iterations": iterations,
        "reached_goal": status == "reached",
        "wall_time_s": float(time.time() - t_start),
        "final_d_to_goal": float(np.linalg.norm(centre - goal_a)),
        "path_len": len(path),
    }


def run_straight_line(cost_field, start, goal) -> dict:
    t_start = time.time()
    centre = np.array(start, dtype=float)
    goal_a = np.array(goal, dtype=float)
    direction = goal_a - centre
    distance = float(np.linalg.norm(direction))
    path = [centre.copy()]
    cumulative_cost = 0.0
    if distance == 0.0:
        return {
            "algorithm": "straight", "status": "reached",
            "cumulative_cost": 0.0, "iterations": 0, "reached_goal": True,
            "wall_time_s": 0.0, "final_d_to_goal": 0.0, "path_len": 1,
        }
    direction /= distance
    n_steps = int(np.ceil(distance / STEP_CELLS))
    for _ in range(n_steps):
        nxt = centre + STEP_CELLS * direction
        if np.linalg.norm(nxt - goal_a) > distance:
            nxt = goal_a.copy()
        cumulative_cost += integrate_true_cost(cost_field, centre, nxt)
        centre = nxt
        path.append(centre.copy())
        if np.linalg.norm(centre - goal_a) <= GOAL_TOLERANCE:
            break
    return {
        "algorithm": "straight",
        "status": "reached" if np.linalg.norm(centre - goal_a) <= GOAL_TOLERANCE else "max_iter",
        "cumulative_cost": float(cumulative_cost),
        "iterations": len(path) - 1,
        "reached_goal": bool(np.linalg.norm(centre - goal_a) <= GOAL_TOLERANCE),
        "wall_time_s": float(time.time() - t_start),
        "final_d_to_goal": float(np.linalg.norm(centre - goal_a)),
        "path_len": len(path),
    }


# ------------------------------------------------------------- experiment

ENVIRONMENTS = [
    # (name, start, goal, n_threats, threat_max)
    ("single_threat",  (10, 50), (90, 50), 1, 1.0),
    ("two_threats",    (10, 50), (90, 50), 2, 1.0),
    ("dense_field",    (10, 10), (90, 90), 5, 0.8),
]


def run_one(env_name, start, goal, n_threats, threat_max, seed):
    """Run all three algorithms on one (env, seed). Errors -> status='error:...'"""
    out = {"env": env_name, "seed": seed, "start": list(start), "goal": list(goal)}
    cf = make_world(seed, n_threats=n_threats, threat_min=0.4, threat_max=threat_max)
    for name, fn in [
        ("swarm_astar",  lambda: run_swarm_astar(cf, start, goal, seed, label=env_name)),
        ("oracle_astar", lambda: run_oracle_astar(cf, start, goal)),
        ("straight",     lambda: run_straight_line(cf, start, goal)),
    ]:
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {
                "algorithm": name, "status": f"error:{type(e).__name__}:{e}",
                "cumulative_cost": float("nan"), "iterations": 0,
                "reached_goal": False, "wall_time_s": 0.0,
                "final_d_to_goal": float("nan"), "path_len": 0,
                "traceback": traceback.format_exc(),
            }
            log(f"      [{env_name} seed={seed}] {name} ERROR: {e}")
    return out


# ----------------------------------------------------------- checkpointing

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "bench_astar_checkpoint.json")
RESULTS_PATH = os.environ.get("OUT_PATH", "bench_astar_results.json")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return {}
    try:
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return {(r["env"], r["seed"]): r for r in data.get("runs", [])}
    except Exception as e:
        log(f"WARNING: failed to load checkpoint: {e}; starting fresh")
        return {}


def save_checkpoint(runs_by_key, config):
    payload = {"config": config, "runs": list(runs_by_key.values())}
    with tempfile.NamedTemporaryFile(mode="w", dir=".", delete=False, suffix=".tmp") as f:
        json.dump(payload, f, indent=2)
        tmp = f.name
    os.replace(tmp, CHECKPOINT_PATH)


def bootstrap_ci(arr, n=1000, seed=0):
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(a), size=(n, len(a)))
    means = a[idxs].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    config = {
        "world_size": WORLD_SIZE, "n_drones": N_DRONES, "n_seeds": N_SEEDS,
        "step_cells": STEP_CELLS, "walk_sigma": WALK_SIGMA,
        "sense_r": SENSE_R, "max_iterations": MAX_ITERATIONS,
        "baseline_cost": BASELINE_COST, "goal_tolerance": GOAL_TOLERANCE,
        "run_timeout_s": RUN_TIMEOUT_S,
        "no_progress_patience": NO_PROGRESS_PATIENCE,
        "transit_substeps": TRANSIT_SUBSTEPS,
    }
    log(f"# bench_astar  config={config}")

    runs_by_key = load_checkpoint()
    log(f"# loaded {len(runs_by_key)} existing runs from checkpoint")

    todo = []
    for env_name, start, goal, n_threats, threat_max in ENVIRONMENTS:
        for seed in range(N_SEEDS):
            if (env_name, seed) not in runs_by_key:
                todo.append((env_name, start, goal, n_threats, threat_max, seed))

    log(f"# {len(todo)} (env, seed) pairs to run")
    t_bench = time.time()

    for i, (env_name, start, goal, n_threats, threat_max, seed) in enumerate(todo):
        log(f"## [{i+1}/{len(todo)}] env={env_name} seed={seed}")
        t0 = time.time()
        result = run_one(env_name, start, goal, n_threats, threat_max, seed)
        runs_by_key[(env_name, seed)] = result
        save_checkpoint(runs_by_key, config)
        log(f"   done in {time.time()-t0:.1f}s "
            f"swarm={result['swarm_astar']['status']} "
            f"oracle={result['oracle_astar']['status']} "
            f"straight={result['straight']['status']}")

    total_t = time.time() - t_bench
    log(f"\n# All runs complete. Wall time: {total_t:.1f}s")

    # ----- Aggregate
    summary = []
    for env_name, start, goal, n_threats, threat_max in ENVIRONMENTS:
        env_runs = [r for (e, s), r in runs_by_key.items() if e == env_name]
        env_runs.sort(key=lambda r: r["seed"])
        n = len(env_runs)
        s_costs = np.array([r["swarm_astar"]["cumulative_cost"] for r in env_runs])
        o_costs = np.array([r["oracle_astar"]["cumulative_cost"] for r in env_runs])
        st_costs = np.array([r["straight"]["cumulative_cost"] for r in env_runs])
        s_status = [r["swarm_astar"]["status"] for r in env_runs]
        o_status = [r["oracle_astar"]["status"] for r in env_runs]

        s_reached = np.array([s == "reached" for s in s_status])
        o_reached = np.array([s == "reached" for s in o_status])

        # Only compute gap on seeds where BOTH algorithms reached the goal.
        both = s_reached & o_reached
        gap_pct = (s_costs[both] / np.maximum(o_costs[both], 1e-6) - 1.0) * 100.0
        savings_vs_straight = (1.0 - s_costs[both] / np.maximum(st_costs[both], 1e-6)) * 100.0

        s_lo, s_hi = bootstrap_ci(s_costs[s_reached], seed=42)
        o_lo, o_hi = bootstrap_ci(o_costs[o_reached], seed=43)
        st_lo, st_hi = bootstrap_ci(st_costs, seed=44)
        gap_lo, gap_hi = bootstrap_ci(gap_pct, seed=45)
        save_lo, save_hi = bootstrap_ci(savings_vs_straight, seed=46)

        sensed = np.array([r["swarm_astar"]["fraction_sensed"] for r in env_runs])
        consensus = np.array([r["swarm_astar"]["consensus_ok"] for r in env_runs])
        status_counts = {st: s_status.count(st) for st in set(s_status)}

        print(f"\n# {env_name}  N={N_DRONES}  seeds={n}  "
              f"reached={s_reached.sum()}/{n}  status={status_counts}")
        if s_reached.any():
            print(f"  swarm_astar  cost: {s_costs[s_reached].mean():7.3f} "
                  f"[{s_lo:.2f},{s_hi:.2f}]  sensed={sensed.mean()*100:4.1f}%")
        if o_reached.any():
            print(f"  oracle_astar cost: {o_costs[o_reached].mean():7.3f} "
                  f"[{o_lo:.2f},{o_hi:.2f}]")
        print(f"  straight     cost: {st_costs.mean():7.3f} [{st_lo:.2f},{st_hi:.2f}]")
        if both.any():
            print(f"  gap vs oracle (reached-only):       "
                  f"{gap_pct.mean():6.2f}% [{gap_lo:.2f},{gap_hi:.2f}]  n={both.sum()}")
            print(f"  savings vs straight (reached-only): "
                  f"{savings_vs_straight.mean():6.2f}% [{save_lo:.2f},{save_hi:.2f}]")
        print(f"  consensus byte-identity: {consensus.mean()*100:.0f}%")

        summary.append({
            "env": env_name, "n_seeds": n,
            "swarm_status_counts": status_counts,
            "swarm_reached": int(s_reached.sum()),
            "oracle_reached": int(o_reached.sum()),
            "swarm_cost_mean": float(s_costs[s_reached].mean()) if s_reached.any() else None,
            "swarm_cost_ci": [s_lo, s_hi] if s_reached.any() else None,
            "oracle_cost_mean": float(o_costs[o_reached].mean()) if o_reached.any() else None,
            "oracle_cost_ci": [o_lo, o_hi] if o_reached.any() else None,
            "straight_cost_mean": float(st_costs.mean()),
            "straight_cost_ci": [st_lo, st_hi],
            "gap_pct_mean": float(gap_pct.mean()) if both.any() else None,
            "gap_pct_ci": [gap_lo, gap_hi] if both.any() else None,
            "savings_vs_straight_pct_mean": float(savings_vs_straight.mean()) if both.any() else None,
            "savings_vs_straight_pct_ci": [save_lo, save_hi] if both.any() else None,
            "fraction_sensed_mean": float(sensed.mean()),
            "consensus_byte_identical": float(consensus.mean()),
        })

    out = {"config": config, "summary": summary,
           "runs": list(runs_by_key.values())}
    with tempfile.NamedTemporaryFile(mode="w", dir=".", delete=False, suffix=".tmp") as f:
        json.dump(out, f, indent=2)
        tmp = f.name
    os.replace(tmp, RESULTS_PATH)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
