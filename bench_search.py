# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Bayesian search bench (SAR-style).

The collective goal: find a hidden target whose location is unknown but
constrained by an operator-provided prior (lost-at-sea, multi-modal, or
banana-along-a-track). The architecture's claim:

  Every drone runs the same Bayesian update on the same broadcast
  detections, then the same deterministic decision rule on the same
  shared posterior, reaching a byte-identical next-manifold without a
  leader. The four existing layers compose without modification.

The algorithmic claim is separate and weaker: Bayesian-swarm should beat
lawnmower (the Coast Guard gold standard) on non-uniform priors. We
benchmark both.

ALGORITHMS
  bayesian   the architecture's algorithm — pick the candidate manifold
             with maximum expected posterior coverage, deterministic
  lawnmower  parallel sweep over the bounding box of the high-prior
             region; the standard SAR baseline
  random     uniformly random candidate at each step (chaos baseline)
  oracle     knows the target location; lower bound on time-to-detect

SCENARIOS
  lost_at_sea   single Gaussian prior, σ=15 cells (≈15 km drifter LKP)
  multimodal    three Gaussian modes — "could be at A, B, or C"
  banana        thick curve representing a flight track or shore drift

ENGINEERING
  - log-space posterior throughout (avoids underflow over many updates)
  - per-iteration telemetry to stderr (flushed)
  - hard wall-clock cap per run (RUN_TIMEOUT_S, default 30 s)
  - atomic checkpoint per (scenario, algo, seed) via tempfile + rename
  - resumable: re-running picks up where it left off
  - status taxonomy: detected / max_manifolds / timeout / error:<msg>
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
MAX_MANIFOLDS = int(os.environ.get("MAX_MANIFOLDS", "400"))
R_SENSOR = float(os.environ.get("R_SENSOR", "5.0"))
SIGMA_SENSOR = float(os.environ.get("SIGMA_SENSOR", "2.0"))
P_MAX = float(os.environ.get("P_MAX", "0.7"))
R_MANIFOLD = float(os.environ.get("R_MANIFOLD", "5.0"))
DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", "0.5"))
# Relaxed detection: declare detected if MAP cell is within DETECTION_KCELLS
# Chebyshev cells of the true target. Default 0 = strict equality. Setting
# this >0 is a fairness adjustment for particle-filter posteriors which
# discretise differently from grid posteriors and can place MAP one cell
# off without that being a search failure.
DETECTION_KCELLS = int(os.environ.get("DETECTION_KCELLS", "0"))


def detected(map_xy, target):
    """Return True if MAP cell is within DETECTION_KCELLS of target."""
    if DETECTION_KCELLS <= 0:
        return map_xy == target
    return (abs(map_xy[0] - target[0]) <= DETECTION_KCELLS
            and abs(map_xy[1] - target[1]) <= DETECTION_KCELLS)
STEP_MAX = float(os.environ.get("STEP_MAX", "10.0"))   # max manifold-to-manifold cells
N_DIRECTIONS = int(os.environ.get("N_DIRECTIONS", "16"))
RUN_TIMEOUT_S = float(os.environ.get("RUN_TIMEOUT_S", "30.0"))
VERBOSE = os.environ.get("VERBOSE", "1") != "0"
TELEMETRY_EVERY = int(os.environ.get("TELEMETRY_EVERY", "5"))

# Floor on log-posterior cell mass to prevent total annihilation.
LOG_FLOOR = -50.0

assert WORLD_SIZE >= 30
assert 1 <= N_DRONES <= 200
assert R_SENSOR > 0 and SIGMA_SENSOR > 0
assert 0 < P_MAX <= 1
assert STEP_MAX > 0


def log(msg):
    if VERBOSE:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


# ----------------------------------------------------------------- world

XS, YS = np.meshgrid(np.arange(WORLD_SIZE), np.arange(WORLD_SIZE))


def gaussian_prior(centres, sigmas, weights=None) -> np.ndarray:
    if weights is None:
        weights = [1.0 / len(centres)] * len(centres)
    p = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for (cx, cy), s, w in zip(centres, sigmas, weights):
        d2 = (XS - cx) ** 2 + (YS - cy) ** 2
        p += w * np.exp(-d2 / (2 * s * s))
    p /= p.sum()
    return p


def banana_prior(start, end, control, sigma) -> np.ndarray:
    """Quadratic Bezier curve from start to end through control,
    thickened by Gaussian sigma."""
    ts = np.linspace(0, 1, 200)
    pts = []
    for t in ts:
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1]
        pts.append((x, y))
    p = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for cx, cy in pts:
        d2 = (XS - cx) ** 2 + (YS - cy) ** 2
        p += np.exp(-d2 / (2 * sigma * sigma))
    p /= p.sum()
    return p


def sample_target_from_prior(prior: np.ndarray, rng) -> tuple:
    flat = prior.flatten()
    idx = rng.choice(len(flat), p=flat)
    cy, cx = divmod(idx, WORLD_SIZE)
    return (int(cx), int(cy))


# ---------------------------------------------------- Allen Leeway tables

LEEWAY_TABLE_PATH = os.environ.get("LEEWAY_TABLE", "/tmp/OBJECTPROP.DAT")


def load_leeway_table(path: str = LEEWAY_TABLE_PATH) -> dict:
    """Load OpenDrift OBJECTPROP.DAT (transcribed from SAROPS by Allen via
    Breivik). Returns {class_id: {name, desc, dwl, r_cwl, l_cwl}} where each
    component is (slope, offset, syx). Coefficient interpretation follows
    Allen & Plourde (1999), Allen (2005), Breivik & Allen (2008):

      drift_DWL  (m/s) = slope * |wind_10m| + offset    (downwind component)
      drift_CWL  (m/s) = slope * |wind_10m| + offset    (cross-wind, signed)
      Syx                                              (residual std, m/s)

    R_CWL acts when crosswind sign is +1 (drift to right of wind);
    L_CWL acts when sign is −1 (drift to left). Sign flips ("jibing")
    are exponentially distributed in real time per Kratzke 2010 §2.5;
    for the bench, we draw the sign at scenario init and either hold
    it (simple) or evolve via Markov state per manifold (advanced).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Leeway table not found at {path}. Source: OpenDrift OBJECTPROP.DAT")
    with open(path) as f:
        lines = f.readlines()
    table = {}
    i = 0
    while i + 2 < len(lines):
        try:
            coeffs = [float(p) for p in lines[i + 2].split()]
            if len(coeffs) == 9:
                name_parts = lines[i].split()
                if len(name_parts) >= 2:
                    name = name_parts[0]
                    obj_id = int(name_parts[-1])
                    desc = lines[i + 1].strip().lstrip(">").strip()
                    table[obj_id] = {
                        "name": name,
                        "desc": desc,
                        "dwl": tuple(coeffs[0:3]),
                        "r_cwl": tuple(coeffs[3:6]),
                        "l_cwl": tuple(coeffs[6:9]),
                    }
                    i += 3
                    continue
        except (ValueError, IndexError):
            pass
        i += 1
    return table


def leeway_drift_per_manifold(obj_id: int, wind_uv: tuple,
                              sign: int, manifold_seconds: float = 240.0,
                              cell_meters: float = 1000.0,
                              table: dict = None) -> tuple:
    """Compute the leeway-induced (Δx, Δy) per manifold in cells.

    Inputs:
      obj_id           — Allen Leeway class id (1..85)
      wind_uv          — (u, v) of 10-m wind in m/s
      sign             — +1 (right of wind) or −1 (left of wind)
      manifold_seconds — bench wall-time per manifold transition
                         (default 240 s ≈ 4 min, calibrated so a
                         60-manifold mission spans 4 hours of search)
      cell_meters      — world cell size in meters (default 1 km)
      table            — leeway table dict; loaded if None

    Returns drift_vec_cells = (Δx, Δy) per manifold in world cells.
    Slopes are unitless (m/s drift per m/s wind); offsets are m/s
    (interpreted as 0.01 m/s units in some Allen reports — the
    OpenDrift transcription stores them in cm/s; the bench treats
    offsets as small additive m/s perturbations).
    """
    if table is None:
        table = _LEEWAY_CACHE
    if obj_id not in table:
        raise KeyError(f"Leeway class {obj_id} not in table")
    p = table[obj_id]
    wu, wv = wind_uv
    wind_speed = float(np.hypot(wu, wv))
    if wind_speed < 1e-6:
        return (0.0, 0.0)
    # Unit vectors: along wind and 90°-rotated (right-hand)
    e_w = (wu / wind_speed, wv / wind_speed)
    e_perp = (-wv / wind_speed, wu / wind_speed)        # +sign = right of wind
    dwl_slope, dwl_offset, _ = p["dwl"]
    cwl = p["r_cwl"] if sign >= 0 else p["l_cwl"]
    cwl_slope, cwl_offset, _ = cwl
    # Treat offsets as cm/s in OpenDrift's transcription
    dwl_speed = dwl_slope * wind_speed + dwl_offset * 0.01
    cwl_speed = cwl_slope * wind_speed + cwl_offset * 0.01
    drift_u = dwl_speed * e_w[0] + sign * cwl_speed * e_perp[0]
    drift_v = dwl_speed * e_w[1] + sign * cwl_speed * e_perp[1]
    # Convert m/s to cells/manifold
    factor = manifold_seconds / cell_meters
    return (drift_u * factor, drift_v * factor)


# Module-level cache: load once on import; tolerates missing file at module
# load time (errors only on use).
try:
    _LEEWAY_CACHE = load_leeway_table()
except FileNotFoundError:
    _LEEWAY_CACHE = {}


def leeway_scenario_factory(obj_id: int, prior_lkp: tuple, prior_sigma: float,
                            wind_uv: tuple, sign: int = 1,
                            manifold_seconds: float = 240.0,
                            diffuse_sigma: float = 0.4):
    """Construct a Leeway-based scenario dict the bench can consume.

    Compute a per-manifold drift_vec from object class + wind, then plug
    into the existing advect-target / advect-posterior dynamics. Sign is
    held fixed (no jibing in this iteration); the fixed-sign assumption
    is documented as a simplification of Kratzke 2010's stochastic-jibing
    model and flagged for a future C-2.1 enhancement.
    """
    drift_vec = leeway_drift_per_manifold(
        obj_id, wind_uv, sign,
        manifold_seconds=manifold_seconds, table=_LEEWAY_CACHE)
    obj = _LEEWAY_CACHE.get(obj_id, {})
    name = obj.get("name", f"obj-{obj_id}")
    desc = obj.get("desc", "")
    return {
        "make_prior": lambda: gaussian_prior([prior_lkp], [prior_sigma]),
        "advect": True,
        "drift_vec": drift_vec,
        "drift_noise_sigma": 0.3,
        "diffuse_sigma": diffuse_sigma,
        "leeway_class_id": obj_id,
        "leeway_class_name": name,
        "wind_uv": wind_uv,
        "manifold_seconds": manifold_seconds,
        "description": (
            f"Leeway-drift target ({name}, {desc[:50]}) "
            f"at wind={wind_uv}, drift={drift_vec[0]:.2f},{drift_vec[1]:.2f} cells/manifold"),
    }


SCENARIOS = {
    "lost_at_sea": {
        "make_prior": lambda: gaussian_prior([(50, 50)], [15.0]),
        "advect": False,
        "description": "single Gaussian prior at LKP (50,50), σ=15, static target",
    },
    "lost_at_sea_drift": {
        "make_prior": lambda: gaussian_prior([(50, 50)], [10.0]),
        "advect": True,
        # Leeway-style drift parameters (Ryan 1973, Kratzke 2010 SAROPS):
        # ~0.5–2 km/hr surface drift in moderate wind; we treat one
        # manifold = ~30 min of search, so ~0.5 cells/manifold drift +
        # ~0.3 cell diffusion per step from drift-parameter uncertainty.
        "drift_vec": (0.5, 0.2),
        "drift_noise_sigma": 0.3,
        "diffuse_sigma": 0.4,
        "description": "drifting target with Leeway-style advection (Ryan 1973)",
    },
    # ----- Allen Leeway-based scenarios (Phase C-2). Wind = 5 m/s along +x
    # (eastward); 4 min/manifold so 60-manifold mission ≈ 4 hours.
    "leeway_piw": leeway_scenario_factory(
        obj_id=1, prior_lkp=(35, 50), prior_sigma=10.0,
        wind_uv=(5.0, 0.0), sign=+1,
    ) if _LEEWAY_CACHE else {
        "make_prior": lambda: gaussian_prior([(35, 50)], [10.0]),
        "advect": False,
        "description": "Leeway PIW (Leeway table not loaded — fallback static)",
    },
    "leeway_liferaft": leeway_scenario_factory(
        obj_id=7, prior_lkp=(25, 50), prior_sigma=10.0,
        wind_uv=(5.0, 0.0), sign=+1,
    ) if _LEEWAY_CACHE else {
        "make_prior": lambda: gaussian_prior([(25, 50)], [10.0]),
        "advect": False,
        "description": "Leeway liferaft (Leeway table not loaded — fallback static)",
    },
    "leeway_skiff": leeway_scenario_factory(
        obj_id=44, prior_lkp=(20, 50), prior_sigma=10.0,
        wind_uv=(5.0, 0.0), sign=+1,
    ) if _LEEWAY_CACHE else {
        "make_prior": lambda: gaussian_prior([(20, 50)], [10.0]),
        "advect": False,
        "description": "Leeway skiff (Leeway table not loaded — fallback static)",
    },
    "multimodal": {
        "make_prior": lambda: gaussian_prior(
            [(25, 30), (60, 70), (75, 25)],
            [8.0, 8.0, 8.0],
            weights=[0.4, 0.35, 0.25],
        ),
        "advect": False,
        "description": "3-mode mixture (could be A, B, or C)",
    },
    "banana": {
        "make_prior": lambda: banana_prior(
            start=(15, 30), control=(50, 80), end=(85, 35), sigma=4.0,
        ),
        "advect": False,
        "description": "thick curve along a projected flight track",
    },
}


def advect_posterior(log_post: np.ndarray,
                     drift_vec: tuple, diffuse_sigma: float) -> np.ndarray:
    """Advect log-posterior by drift_vec (cells) and Gaussian-diffuse.

    Models drift between manifolds: target moves with currents, and our
    uncertainty about drift parameters causes posterior diffusion.
    """
    from scipy.ndimage import shift, gaussian_filter
    post = np.exp(log_post)
    if drift_vec[0] != 0 or drift_vec[1] != 0:
        post = shift(post, [drift_vec[1], drift_vec[0]], cval=0.0, order=1)
    if diffuse_sigma > 0:
        post = gaussian_filter(post, diffuse_sigma)
    post = np.maximum(post, np.exp(LOG_FLOOR))
    s = post.sum()
    if s > 0:
        post /= s
    return np.log(post)


def advect_target(target: tuple, drift_vec: tuple,
                  drift_noise_sigma: float, rng) -> tuple:
    nx = target[0] + drift_vec[0] + rng.normal(0, drift_noise_sigma)
    ny = target[1] + drift_vec[1] + rng.normal(0, drift_noise_sigma)
    return (int(np.clip(nx, 0, WORLD_SIZE - 1)),
            int(np.clip(ny, 0, WORLD_SIZE - 1)))


# --------------------------------------------------------------- sensor

def detection_field(drone_pos: np.ndarray) -> np.ndarray:
    """For a drone at drone_pos, return P(detect | target at each cell)."""
    d2 = (XS - drone_pos[0]) ** 2 + (YS - drone_pos[1]) ** 2
    p = np.where(d2 <= R_SENSOR ** 2,
                 P_MAX * np.exp(-d2 / (2 * SIGMA_SENSOR ** 2)),
                 0.0)
    return p


def disk_drone_positions(centre, n: int = N_DRONES, radius: float = R_MANIFOLD) -> np.ndarray:
    """Sunflower-style distribution in a disk for even coverage."""
    golden = np.pi * (3.0 - np.sqrt(5.0))
    rs = radius * np.sqrt(np.arange(n) / max(n - 1, 1))
    angles = golden * np.arange(n)
    pos = np.stack([centre[0] + rs * np.cos(angles),
                    centre[1] + rs * np.sin(angles)], axis=1)
    return pos


# ------------------------------------------------------- bayesian update

def bayesian_update(log_post: np.ndarray, drone_positions: np.ndarray,
                    target: tuple, rng) -> tuple:
    """Update log-posterior in place given the manifold's drones.

    For each drone:
      - sample a detection event (Bernoulli with P at the true target)
      - update log_post += log P(observation | target at each cell)

    Returns (log_post, any_detected, n_detections).
    """
    detections = 0
    any_det = False
    H, W = log_post.shape
    tx, ty = target
    for d_pos in drone_positions:
        p_field = detection_field(d_pos)
        p_at_target = float(p_field[ty, tx])
        detected = bool(rng.random() < p_at_target)
        if detected:
            detections += 1
            any_det = True
            with np.errstate(divide="ignore"):
                log_post += np.log(np.maximum(p_field, np.exp(LOG_FLOOR)))
        else:
            log_post += np.log(np.clip(1.0 - p_field, np.exp(LOG_FLOOR), 1.0))
    log_post = np.maximum(log_post, LOG_FLOOR)
    # Normalise via logsumexp.
    m = log_post.max()
    log_post -= m + np.log(np.exp(log_post - m).sum())
    return log_post, any_det, detections


def map_cell(log_post: np.ndarray) -> tuple:
    idx = int(np.argmax(log_post))
    return (idx % WORLD_SIZE, idx // WORLD_SIZE)


def map_confidence(log_post: np.ndarray) -> float:
    return float(np.exp(log_post.max()))


def posterior_entropy(log_post: np.ndarray) -> float:
    p = np.exp(log_post)
    p_safe = np.where(p > 0, p, 1.0)
    return float(-(p * np.log(p_safe)).sum())


# --------------------------------------------------------- algorithms

CARDINAL_DIRS = np.stack([
    np.cos(np.linspace(0, 2 * np.pi, N_DIRECTIONS, endpoint=False)),
    np.sin(np.linspace(0, 2 * np.pi, N_DIRECTIONS, endpoint=False)),
], axis=1)


def candidate_centres(current: np.ndarray) -> list:
    """Candidate next manifold centres: stay-put + N_DIRECTIONS * STEP_MAX."""
    out = [tuple(current)]
    for d in CARDINAL_DIRS:
        c = current + STEP_MAX * d
        if 0 <= c[0] < WORLD_SIZE and 0 <= c[1] < WORLD_SIZE:
            out.append(tuple(c))
    return out


# Pre-compute disk masks for fast mass-in-region scoring.
_DISK_MASKS = {}
def disk_mask(radius: float) -> np.ndarray:
    if radius in _DISK_MASKS:
        return _DISK_MASKS[radius]
    R = int(np.ceil(radius))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    m = (xx ** 2 + yy ** 2 <= radius ** 2).astype(float)
    _DISK_MASKS[radius] = m
    return m


def coverage_mass(post: np.ndarray, centre: tuple, radius: float = R_MANIFOLD) -> float:
    """Mass of the posterior within a disk of radius around centre."""
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    R = int(np.ceil(radius))
    x0, x1 = max(0, cx - R), min(WORLD_SIZE, cx + R + 1)
    y0, y1 = max(0, cy - R), min(WORLD_SIZE, cy + R + 1)
    sub = post[y0:y1, x0:x1]
    mask = disk_mask(radius)
    mx0 = R - (cx - x0); mx1 = mx0 + (x1 - x0)
    my0 = R - (cy - y0); my1 = my0 + (y1 - y0)
    return float((sub * mask[my0:my1, mx0:mx1]).sum())


def algo_bayesian(log_post: np.ndarray, current: np.ndarray, state: dict) -> np.ndarray:
    """Pick the candidate centre that maximises coverage_mass on the posterior.

    Tie-break: prefer non-stay if equal, then smaller index (deterministic).
    """
    post = np.exp(log_post)
    cands = candidate_centres(current)
    best = None
    best_score = -1.0
    for c in cands:
        s = coverage_mass(post, c)
        if s > best_score + 1e-12:
            best_score = s
            best = c
    return np.array(best, dtype=float)


def algo_bayesian_eig(log_post: np.ndarray, current: np.ndarray, state: dict) -> np.ndarray:
    """Pick the candidate centre that maximises expected information gain.

    EIG_drone(c) = MI(O_d ; T) = H[P(detect)] − E_T[H[p(d, T)]]
    Sums per-drone MIs (independence approximation; lower bound on true
    joint MI for overlapping drones).
    """
    post = np.exp(log_post)
    cands = candidate_centres(current)
    best = None
    best_eig = -np.inf
    for c in cands:
        drones = disk_drone_positions(np.array(c, dtype=float))
        eig_total = 0.0
        for d_pos in drones:
            p_field = detection_field(d_pos)
            p_avg = float((post * p_field).sum())
            p_avg = max(min(p_avg, 1.0 - 1e-12), 1e-12)
            h_outcome = -p_avg * np.log(p_avg) - (1.0 - p_avg) * np.log(1.0 - p_avg)
            p_safe = np.clip(p_field, 1e-12, 1.0 - 1e-12)
            h_cell = -p_safe * np.log(p_safe) - (1.0 - p_safe) * np.log(1.0 - p_safe)
            h_cell = np.where(p_field > 1e-9, h_cell, 0.0)
            h_cond = float((post * h_cell).sum())
            eig_total += h_outcome - h_cond
        if eig_total > best_eig + 1e-12:
            best_eig = eig_total
            best = c
    return np.array(best, dtype=float)


def algo_random(log_post, current, state):
    rng = state["rng"]
    cands = candidate_centres(current)
    return np.array(cands[rng.integers(0, len(cands))], dtype=float)


def algo_oracle(log_post, current, state):
    target = state["target"]
    return np.array(target, dtype=float)


def algo_lawnmower(log_post, current, state):
    """Boustrophedon over the bounding box of the high-prior region.

    The bounding box is set once at the start of the run from the *prior*
    (the operator's pre-mission belief). The lawnmower visits a regular
    grid of centres spaced 2*R_MANIFOLD apart, snake order.
    """
    if "lm_path" not in state:
        prior = state["prior"]
        thresh = prior.max() * 0.01
        ys, xs = np.where(prior > thresh)
        if len(xs) == 0:
            ys = np.array([WORLD_SIZE // 2]); xs = np.array([WORLD_SIZE // 2])
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        # Track spacing = 2·SIGMA_SENSOR (standard SAR overlap sweep). With
        # σ_sensor=2, spacing=4: midline single-pass detection 0.7·exp(-0.5)
        # ≈ 0.42, two adjacent passes cumulative ≈ 0.66.
        spacing = max(1, int(round(2 * SIGMA_SENSOR)))
        cols = list(range(x0, x1 + 1, spacing))
        rows = list(range(y0, y1 + 1, spacing))
        path = []
        for ri, ry in enumerate(rows):
            cx_iter = cols if ri % 2 == 0 else list(reversed(cols))
            for cx in cx_iter:
                path.append((float(cx), float(ry)))
        state["lm_path"] = path
        state["lm_idx"] = 0
    i = state["lm_idx"]
    if i >= len(state["lm_path"]):
        # Lawnmower exhausted; stay put.
        return np.array(current, dtype=float)
    state["lm_idx"] = i + 1
    return np.array(state["lm_path"][i], dtype=float)


# ----------------------------------------------------- SAROPS-class PF
# Phase C-3: particle-filter representation faithful to Kratzke 2010 §2 and
# the OPERATIONAL-UNKNOWN choices pinned in audit/07_sarops_class_config.yaml.
# Documented departures from K10 (all flagged in 07 config):
# - Resampling: OMITTED by default, matching K10's apparent design (no
#   resampling step is documented). Sensitivity-sweep alternative: systematic
#   resampling at ESS < N/2.
# - Per-particle leeway slope: drawn ν ∼ N(m, syx·0.01) with syx in cm/s
#   per OpenDrift OBJECTPROP.DAT convention. K10 supports Rayleigh for small m;
#   we use Gaussian for all classes (DOCUMENTED-CHOICE).
# - Crosswind sign: per-particle, with stochastic flips per manifold at
#   exponentially-distributed intervals (jibing). K10 says exponential but
#   does not state the mean rate; we use jibe_rate = 0.05 per manifold.
# - AR(1) wind/current perturbations: omitted in this iteration; constant
#   wind. SECONDARY OPERATIONAL-UNKNOWN, sensitivity-flagged.

PF_N_PARTICLES = int(os.environ.get("PF_N_PARTICLES", "5000"))
PF_JIBE_RATE = float(os.environ.get("PF_JIBE_RATE", "0.05"))
PF_RESAMPLE = os.environ.get("PF_RESAMPLE", "off")           # off | systematic
PF_RESAMPLE_ESS_FRAC = float(os.environ.get("PF_RESAMPLE_ESS_FRAC", "0.5"))


class ParticleFilter:
    """Lightweight particle filter for SAROPS-class belief propagation.

    State per particle: position (x, y) in float-cell coordinates, weight w,
    drawn-once leeway slopes (downwind, right-crosswind, left-crosswind),
    crosswind sign ∈ {−1, +1}.

    Update form: K10 eq.1 product-of-leg-fails (= standard Bayesian likelihood
    update with binary detection observations). Drift propagation uses Allen
    Leeway with per-particle slopes drawn at init.
    """

    __slots__ = ("n", "x", "y", "w", "dwl_slope", "rcwl_slope", "lcwl_slope",
                 "sign", "leeway_id", "wind_uv", "manifold_seconds",
                 "_jibe_rate")

    def __init__(self, n_particles: int, prior: np.ndarray, rng,
                 leeway_class_id: int = None, wind_uv: tuple = (0.0, 0.0),
                 manifold_seconds: float = 240.0,
                 jibe_rate: float = PF_JIBE_RATE):
        self.n = n_particles
        flat = prior.flatten().astype(float)
        flat = flat / flat.sum()
        idxs = rng.choice(len(flat), size=n_particles, replace=True, p=flat)
        cy = idxs // WORLD_SIZE
        cx = idxs % WORLD_SIZE
        self.x = cx.astype(float) + rng.uniform(0.0, 1.0, n_particles)
        self.y = cy.astype(float) + rng.uniform(0.0, 1.0, n_particles)
        self.w = np.full(n_particles, 1.0 / n_particles)
        self.leeway_id = leeway_class_id
        self.wind_uv = tuple(wind_uv)
        self.manifold_seconds = float(manifold_seconds)
        self._jibe_rate = float(jibe_rate)
        if leeway_class_id is not None and leeway_class_id in _LEEWAY_CACHE:
            params = _LEEWAY_CACHE[leeway_class_id]
            dwl_slope, _, dwl_syx = params["dwl"]
            r_slope, _, r_syx = params["r_cwl"]
            l_slope, _, l_syx = params["l_cwl"]
            # Per-particle slope draws (Gaussian; syx in cm/s per OpenDrift)
            self.dwl_slope = rng.normal(dwl_slope, max(dwl_syx * 0.01, 1e-6),
                                        n_particles)
            self.rcwl_slope = rng.normal(r_slope, max(r_syx * 0.01, 1e-6),
                                         n_particles)
            self.lcwl_slope = rng.normal(l_slope, max(l_syx * 0.01, 1e-6),
                                         n_particles)
            self.sign = rng.choice(np.array([-1, 1]), size=n_particles)
        else:
            self.dwl_slope = self.rcwl_slope = self.lcwl_slope = None
            self.sign = None

    def propagate(self, rng, fallback_drift_vec=(0.0, 0.0),
                  fallback_drift_noise=0.0):
        """Drift particles over one manifold transition. If a Leeway class
        is configured, use per-particle Leeway dynamics; otherwise apply
        a fallback drift_vec (used by simple-drift scenarios).
        """
        if self.dwl_slope is None:
            # Fallback: apply scenario drift_vec uniformly + per-particle noise
            dvx, dvy = fallback_drift_vec
            if dvx != 0.0 or dvy != 0.0 or fallback_drift_noise > 0.0:
                noise_x = rng.normal(0.0, fallback_drift_noise, self.n)
                noise_y = rng.normal(0.0, fallback_drift_noise, self.n)
                self.x = np.clip(self.x + dvx + noise_x, 0.0, WORLD_SIZE - 1.0)
                self.y = np.clip(self.y + dvy + noise_y, 0.0, WORLD_SIZE - 1.0)
            return
        wu, wv = self.wind_uv
        wind_speed = float(np.hypot(wu, wv))
        if wind_speed < 1e-6:
            return
        e_w = np.array([wu / wind_speed, wv / wind_speed])
        e_perp = np.array([-wv / wind_speed, wu / wind_speed])
        flip = rng.random(self.n) < self._jibe_rate
        self.sign = np.where(flip, -self.sign, self.sign)
        cwl_slope = np.where(self.sign > 0, self.rcwl_slope, self.lcwl_slope)
        dwl_speed = self.dwl_slope * wind_speed
        cwl_speed = cwl_slope * wind_speed
        factor = self.manifold_seconds / 1000.0    # m → cells
        dx = (dwl_speed * e_w[0]
              + self.sign.astype(float) * cwl_speed * e_perp[0]) * factor
        dy = (dwl_speed * e_w[1]
              + self.sign.astype(float) * cwl_speed * e_perp[1]) * factor
        self.x = np.clip(self.x + dx, 0.0, WORLD_SIZE - 1.0)
        self.y = np.clip(self.y + dy, 0.0, WORLD_SIZE - 1.0)

    def update(self, observations, rng=None):
        """Apply Bayesian likelihood update across all drone observations.

        observations: list of (detected: bool, drone_pos: (dx, dy)) tuples.
        Implements both detection-positive (multiply by p_det) and
        detection-negative (multiply by 1−p_det = K10 eq.1 negative-info)
        forms.

        rng: required only if PF_RESAMPLE != "off"; threading it keeps the
        resampling code path deterministic from the run's seed.
        """
        for detected, d_pos in observations:
            d2 = (self.x - d_pos[0]) ** 2 + (self.y - d_pos[1]) ** 2
            in_range = d2 <= R_SENSOR ** 2
            p_det = np.where(in_range,
                             P_MAX * np.exp(-d2 / (2.0 * SIGMA_SENSOR ** 2)),
                             0.0)
            if detected:
                self.w = self.w * p_det
            else:
                self.w = self.w * (1.0 - p_det)
        # Renormalize, with floor to prevent total annihilation
        self.w = np.maximum(self.w, np.exp(LOG_FLOOR))
        self.w = self.w / self.w.sum()
        # Optional resampling (DOCUMENTED-CHOICE per 07 config; default OFF)
        if PF_RESAMPLE == "systematic":
            ess = 1.0 / (self.w ** 2).sum()
            if ess < PF_RESAMPLE_ESS_FRAC * self.n:
                if rng is None:
                    raise ValueError(
                        "PF_RESAMPLE='systematic' requires update(rng=…) for reproducibility")
                self._systematic_resample(rng)

    def _systematic_resample(self, rng):
        """Systematic resampling (Stone group convention). Threads the
        run's RNG through to keep the resampling code path deterministic
        when invoked.
        """
        positions = (np.arange(self.n) + rng.random()) / self.n
        cumulative = np.cumsum(self.w)
        idxs = np.searchsorted(cumulative, positions)
        idxs = np.clip(idxs, 0, self.n - 1)
        self.x = self.x[idxs].copy()
        self.y = self.y[idxs].copy()
        if self.dwl_slope is not None:
            self.dwl_slope = self.dwl_slope[idxs].copy()
            self.rcwl_slope = self.rcwl_slope[idxs].copy()
            self.lcwl_slope = self.lcwl_slope[idxs].copy()
            self.sign = self.sign[idxs].copy()
        self.w = np.full(self.n, 1.0 / self.n)

    def to_grid(self) -> np.ndarray:
        """Convert weighted particles to a posterior grid (W×H)."""
        grid = np.zeros((WORLD_SIZE, WORLD_SIZE))
        ix = np.clip(self.x.astype(int), 0, WORLD_SIZE - 1)
        iy = np.clip(self.y.astype(int), 0, WORLD_SIZE - 1)
        np.add.at(grid, (iy, ix), self.w)
        grid += 1e-12                          # avoid log(0) downstream
        grid /= grid.sum()
        return grid

    def to_log_grid(self) -> np.ndarray:
        return np.log(self.to_grid())


def run_sarops_class(scenario_name: str, seed: int) -> dict:
    """Run SAROPS-class particle filter over the bench's manifold loop.

    Single shared posterior (centralized) — the architectural counterpart
    is the multi-posterior comms-loss case run via run_comms_channel with
    a SAROPS-class algorithm. This function corresponds to the
    centralized, reliable-comms case where every drone sees the same
    broadcast and runs the same PF update.
    """
    t_start = time.time()
    rng = np.random.default_rng(seed)
    scen = SCENARIOS[scenario_name]
    prior = scen["make_prior"]()
    target = sample_target_from_prior(prior, rng)
    leeway_id = scen.get("leeway_class_id")
    wind_uv = scen.get("wind_uv", (0.0, 0.0))
    manifold_seconds = scen.get("manifold_seconds", 240.0)
    pf = ParticleFilter(PF_N_PARTICLES, prior, rng,
                        leeway_class_id=leeway_id, wind_uv=wind_uv,
                        manifold_seconds=manifold_seconds)
    advect = scen.get("advect", False)
    drift_vec = scen.get("drift_vec", (0.0, 0.0))
    drift_noise = scen.get("drift_noise_sigma", 0.0)
    centre = np.array([WORLD_SIZE / 2.0, WORLD_SIZE / 2.0])
    distance = 0.0
    found = False
    detection_iter = -1
    n_detections_total = 0
    iterations = 0
    status = "max_manifolds"
    budget = ALGO_BUDGETS.get("sarops_class", 100)
    confidence_history = []
    entropy_history = []

    for k in range(budget):
        iterations = k + 1
        if time.time() - t_start > RUN_TIMEOUT_S:
            status = "timeout"
            break
        # Sense
        drones = disk_drone_positions(centre)
        observations = []
        for d_pos in drones:
            p_field = detection_field(d_pos)
            p_at_target = float(p_field[target[1], target[0]])
            obs = bool(rng.random() < p_at_target)
            observations.append((obs, d_pos))
            if obs:
                n_detections_total += 1
        # PF update
        pf.update(observations, rng=rng)
        # Detection check via grid representation
        log_post = pf.to_log_grid()
        confidence_history.append(map_confidence(log_post))
        entropy_history.append(posterior_entropy(log_post))
        any_det = any(o[0] for o in observations)
        if any_det and confidence_history[-1] >= DETECTION_THRESHOLD:
            if detected(map_cell(log_post), target):
                found = True
                detection_iter = k
                status = "detected"
                break
        # Decision: coverage-mass-argmax over the PF posterior grid
        next_c = algo_bayesian(log_post, centre, {"prior": prior, "target": target})
        delta = next_c - centre
        d_norm = float(np.linalg.norm(delta))
        if d_norm > STEP_MAX:
            delta *= STEP_MAX / d_norm
            next_c = centre + delta
        next_c[0] = np.clip(next_c[0], 0, WORLD_SIZE - 1)
        next_c[1] = np.clip(next_c[1], 0, WORLD_SIZE - 1)
        distance += float(np.linalg.norm(next_c - centre))
        centre = next_c
        # Advect target (if drifting scenario)
        if advect:
            target = advect_target(target, drift_vec, drift_noise, rng)
        # Propagate particles (Leeway if configured, else fallback drift_vec)
        pf.propagate(rng, fallback_drift_vec=drift_vec,
                     fallback_drift_noise=drift_noise)

    return {
        "scenario": scenario_name,
        "algorithm": "sarops_class",
        "seed": seed,
        "status": status,
        "found": found,
        "detection_iter": int(detection_iter),
        "iterations": iterations,
        "distance": float(distance),
        "wall_time_s": float(time.time() - t_start),
        "n_detections_total": int(n_detections_total),
        "final_confidence": float(confidence_history[-1] if confidence_history else 0),
        "final_entropy": float(entropy_history[-1] if entropy_history else 0),
        "target": list(target),
        "pf_n_particles": PF_N_PARTICLES,
        "pf_resample": PF_RESAMPLE,
    }


# --------------------------------------------- Phase C-4: B-C consensus
# Hare, Bandyopadhyay, Chung (2018) "Distributed Bayesian Filtering using
# Logarithmic Opinion Pool for Dynamic Sensor Networks." Each drone runs
# a local Bayesian update on its own observation only, then consensus
# rounds mix the log-posteriors across drones. Convergence is asymptotic
# (mean-square or to an error ball) — not byte-identical per tick.

BC_CONSENSUS_ROUNDS = int(os.environ.get("BC_CONSENSUS_ROUNDS", "5"))


def run_bandyopadhyay_chung(scenario_name: str, seed: int,
                            channel: Channel = None,
                            n_consensus_rounds: int = None) -> dict:
    """Distributed Bayesian filter with log-opinion-pool consensus loop.

    Each drone updates its own local posterior from its own detection
    event, then T rounds of consensus average the log-posteriors across
    drones. Under reliable comms (lossless channel), consensus converges
    to a common posterior — but it takes T>1 rounds to do so. Under
    loss, consensus is approximate.

    Compare to ours (broadcast-substrate): we exchange the raw detection
    events on the broadcast, every drone runs the SAME update on the
    SAME data, no consensus rounds needed. B-C exchanges processed
    posteriors and runs consensus.
    """
    from collections import Counter
    if n_consensus_rounds is None:
        n_consensus_rounds = BC_CONSENSUS_ROUNDS
    if channel is None:
        channel = IIDChannel(0.0)
    t_start = time.time()
    rng = np.random.default_rng(seed)
    scen = SCENARIOS[scenario_name]
    prior = scen["make_prior"]()
    target = sample_target_from_prior(prior, rng)
    initial_log_post = np.log(np.clip(prior, np.exp(LOG_FLOOR), None))
    m0 = initial_log_post.max()
    initial_log_post -= m0 + np.log(np.exp(initial_log_post - m0).sum())
    log_posts = [initial_log_post.copy() for _ in range(N_DRONES)]

    advect = scen.get("advect", False)
    drift_vec = scen.get("drift_vec", (0.0, 0.0))
    drift_noise = scen.get("drift_noise_sigma", 0.0)
    diffuse_sigma = scen.get("diffuse_sigma", 0.0)
    budget = ALGO_BUDGETS.get("bayesian_bc", 100)

    centre = np.array([WORLD_SIZE / 2.0, WORLD_SIZE / 2.0])
    distance = 0.0
    found = False
    detection_iter = -1
    n_detections_total = 0
    unanimous_count = 0
    n_decisions = 0
    status = "max_manifolds"
    iterations = 0

    for k in range(budget):
        iterations = k + 1
        if time.time() - t_start > RUN_TIMEOUT_S:
            status = "timeout"
            break

        drone_positions = disk_drone_positions(centre)
        observations = []
        for d_pos in drone_positions:
            p_field = detection_field(d_pos)
            p_at_target = float(p_field[target[1], target[0]])
            obs = bool(rng.random() < p_at_target)
            observations.append((obs, d_pos, p_field))
            if obs:
                n_detections_total += 1

        # Local update: each drone updates from ITS OWN observation only.
        for r_idx in range(N_DRONES):
            obs, d_pos, p_field = observations[r_idx]
            if obs:
                log_posts[r_idx] += np.log(np.maximum(p_field, np.exp(LOG_FLOOR)))
            else:
                log_posts[r_idx] += np.log(np.clip(1.0 - p_field, np.exp(LOG_FLOOR), 1.0))
            log_posts[r_idx] = np.maximum(log_posts[r_idx], LOG_FLOOR)

        # Log-opinion-pool consensus: T rounds of averaging on log_posts.
        # Per Hare-Bandyopadhyay-Chung 2018: log_p_i ← sum_j w_ij log_p_j
        # with uniform weights w_ij = 1/N (fully-connected) under reliable
        # comms; under loss, channel.drops decides whether each pairwise
        # message is received this round.
        for _ in range(n_consensus_rounds):
            new_log_posts = []
            for r_idx in range(N_DRONES):
                received = [log_posts[r_idx]]   # always have own state
                for s_idx in range(N_DRONES):
                    if s_idx == r_idx:
                        continue
                    if not channel.drops(s_idx, r_idx, k, rng):
                        received.append(log_posts[s_idx])
                avg = np.mean(received, axis=0)
                # Renormalise
                mx = avg.max()
                avg -= mx + np.log(np.exp(avg - mx).sum())
                new_log_posts.append(avg)
            log_posts = new_log_posts

        # Detection check via receiver 0
        any_det = any(o[0] for o in observations)
        if any_det:
            mc = map_confidence(log_posts[0])
            if mc >= DETECTION_THRESHOLD and detected(map_cell(log_posts[0]), target):
                found = True
                detection_iter = k
                status = "detected"
                break

        # Decision rule: each drone runs coverage-mass-argmax on its own
        # log_post. Majority vote breaks ties.
        decisions = []
        for r_idx in range(N_DRONES):
            d = algo_bayesian(log_posts[r_idx], centre,
                              {"prior": prior, "target": target})
            decisions.append((int(d[0]), int(d[1])))
        if all(d == decisions[0] for d in decisions):
            unanimous_count += 1
        n_decisions += 1
        cnt = Counter(decisions)
        next_c = np.array(cnt.most_common(1)[0][0], dtype=float)
        delta = next_c - centre
        d_norm = float(np.linalg.norm(delta))
        if d_norm > STEP_MAX:
            delta *= STEP_MAX / d_norm
            next_c = centre + delta
        next_c[0] = np.clip(next_c[0], 0, WORLD_SIZE - 1)
        next_c[1] = np.clip(next_c[1], 0, WORLD_SIZE - 1)
        distance += float(np.linalg.norm(next_c - centre))
        centre = next_c

        if advect:
            target = advect_target(target, drift_vec, drift_noise, rng)
            for r_idx in range(N_DRONES):
                log_posts[r_idx] = advect_posterior(log_posts[r_idx], drift_vec, diffuse_sigma)

    final_conf = float(map_confidence(log_posts[0]))
    return {
        "scenario": scenario_name,
        "algorithm": "bayesian_bc",
        "seed": seed,
        "channel_label": channel.label(),
        "status": status,
        "found": found,
        "detection_iter": int(detection_iter),
        "iterations": iterations,
        "distance": float(distance),
        "wall_time_s": float(time.time() - t_start),
        "n_detections_total": int(n_detections_total),
        "final_confidence": final_conf,
        "final_entropy": float(posterior_entropy(log_posts[0])),
        "unanimity_rate": float(unanimous_count / max(n_decisions, 1)),
        "n_decisions": int(n_decisions),
        "n_consensus_rounds": int(n_consensus_rounds),
        "target": list(target),
    }


# ----------------------------------------- Phase C-5: drift-aware lawnmower
# Sweeps the predicted drift cone (the region the target could be in given
# elapsed time × Leeway slope × wind direction × drift uncertainty) rather
# than the static prior bounding box. Eliminates the strawman issue where
# a static-box lawnmower is unfair against drifting targets.

def algo_lawnmower_drift_aware(log_post, current, state):
    """Sweep the predicted drift cone. Cone is computed at scenario init
    from leeway parameters + elapsed manifolds × drift_vec.

    For Leeway scenarios: cone width = 2σ in crosswind direction, length =
    drift_speed × time_elapsed in downwind direction, anchored at the
    posterior centroid. For non-Leeway scenarios with a drift_vec, the
    cone is a rectangle aligned with drift_vec extended by 2σ across the
    perpendicular axis.
    """
    if "lm_drift_path" not in state:
        prior = state["prior"]
        scen = state.get("scenario", {})
        drift_vec = scen.get("drift_vec", (0.0, 0.0))
        budget = state.get("budget", 100)
        # Cone end-position after full mission
        end_dx = drift_vec[0] * budget
        end_dy = drift_vec[1] * budget
        # Centroid of prior is the start of the cone
        ys, xs = np.where(prior > prior.max() * 0.01)
        if len(xs) == 0:
            cx_start, cy_start = WORLD_SIZE // 2, WORLD_SIZE // 2
        else:
            cx_start = float(xs.mean())
            cy_start = float(ys.mean())
        # Bounding box of cone: [start, start+drift] expanded by 2σ_lat
        sigma_lat = max(5.0, abs(end_dx) * 0.3 + abs(end_dy) * 0.3)
        x0 = max(0, int(min(cx_start, cx_start + end_dx) - sigma_lat))
        x1 = min(WORLD_SIZE, int(max(cx_start, cx_start + end_dx) + sigma_lat))
        y0 = max(0, int(min(cy_start, cy_start + end_dy) - sigma_lat))
        y1 = min(WORLD_SIZE, int(max(cy_start, cy_start + end_dy) + sigma_lat))
        spacing = max(1, int(round(2 * SIGMA_SENSOR)))
        cols = list(range(x0, x1 + 1, spacing))
        rows = list(range(y0, y1 + 1, spacing))
        path = []
        for ri, ry in enumerate(rows):
            cx_iter = cols if ri % 2 == 0 else list(reversed(cols))
            for cx in cx_iter:
                path.append((float(cx), float(ry)))
        state["lm_drift_path"] = path
        state["lm_drift_idx"] = 0
    i = state["lm_drift_idx"]
    if i >= len(state["lm_drift_path"]):
        return np.array(current, dtype=float)
    state["lm_drift_idx"] = i + 1
    return np.array(state["lm_drift_path"][i], dtype=float)


ALGORITHMS = {
    "bayesian": algo_bayesian,
    "bayesian_eig": algo_bayesian_eig,
    "sarops_class": None,    # special-cased in main(); uses run_sarops_class
    "bayesian_bc": None,     # special-cased; T=5
    "bayesian_bc_t1": None,  # special-cased; T=1 (degenerate broadcast-equivalent)
    "bayesian_bc_t20": None, # special-cased; T=20 (high-T B-C)
    "bayesian_bc_t50": None, # special-cased; T=50 (very-high-T B-C)
    "lawnmower": algo_lawnmower,
    "lawnmower_drift": algo_lawnmower_drift_aware,
    "random": algo_random,
    "oracle": algo_oracle,
}


BC_T_BY_ALGO = {
    "bayesian_bc": 5,
    "bayesian_bc_t1": 1,
    "bayesian_bc_t20": 20,
    "bayesian_bc_t50": 50,
}


# ----------------------------------------------------------------- run

ALGO_BUDGETS = {
    # Lawnmower needs the full 2*sigma_sensor sweep budget. Bayesian /
    # EIG / oracle find within tens of manifolds; capping random at
    # 200 keeps the chaos-baseline numbers honest without burning
    # extra wall time per seed.
    "lawnmower": MAX_MANIFOLDS,
    "lawnmower_drift": MAX_MANIFOLDS,
    "bayesian": 100,
    "bayesian_eig": 100,
    "sarops_class": 100,
    "bayesian_bc": 100,
    "bayesian_bc_t1": 100,
    "bayesian_bc_t20": 100,
    "bayesian_bc_t50": 100,
    "random": 200,
    "oracle": 30,
}


def run_one(scenario_name: str, algo_name: str, seed: int) -> dict:
    """One (scenario, algorithm, seed) run."""
    t_start = time.time()
    rng = np.random.default_rng(seed)
    scen = SCENARIOS[scenario_name]
    prior = scen["make_prior"]()
    target = sample_target_from_prior(prior, rng)
    log_post = np.log(np.clip(prior, np.exp(LOG_FLOOR), None))
    log_post -= log_post.max() + np.log(np.exp(log_post - log_post.max()).sum())
    budget = ALGO_BUDGETS.get(algo_name, MAX_MANIFOLDS)

    advect = scen.get("advect", False)
    drift_vec = scen.get("drift_vec", (0.0, 0.0))
    drift_noise = scen.get("drift_noise_sigma", 0.0)
    diffuse_sigma = scen.get("diffuse_sigma", 0.0)

    centre = np.array([WORLD_SIZE / 2.0, WORLD_SIZE / 2.0])
    distance = 0.0
    found = False
    detection_iter = -1
    n_detections_total = 0

    state = {"rng": rng, "prior": prior, "target": target,
             "scenario": scen, "budget": budget}
    algo = ALGORITHMS[algo_name]

    entropy_history = [posterior_entropy(log_post)]
    confidence_history = [map_confidence(log_post)]

    status = "max_manifolds"
    iterations = 0

    for k in range(budget):
        iterations = k + 1
        elapsed = time.time() - t_start
        if elapsed > RUN_TIMEOUT_S:
            status = "timeout"
            break

        drones = disk_drone_positions(centre)
        log_post, any_det, n_det = bayesian_update(log_post, drones, target, rng)
        n_detections_total += n_det

        confidence_history.append(map_confidence(log_post))
        entropy_history.append(posterior_entropy(log_post))

        if any_det and confidence_history[-1] >= DETECTION_THRESHOLD:
            map_xy = map_cell(log_post)
            if detected(map_xy, target):
                found = True
                detection_iter = k
                status = "detected"
                break

        # Advect target and posterior between manifolds (drifting scenarios).
        if advect:
            target = advect_target(target, drift_vec, drift_noise, rng)
            state["target"] = target
            log_post = advect_posterior(log_post, drift_vec, diffuse_sigma)

        if VERBOSE and k % TELEMETRY_EVERY == 0:
            log(f"      [{scenario_name}/{algo_name} seed={seed}] "
                f"k={k:>3} centre={tuple(centre.astype(int))} "
                f"target={target} dets={n_detections_total} "
                f"conf={confidence_history[-1]:.3f} "
                f"H={entropy_history[-1]:.2f} t={elapsed:.1f}s")

        next_c = algo(log_post, centre, state)
        # Clamp step size and world bounds
        delta = next_c - centre
        d = np.linalg.norm(delta)
        if d > STEP_MAX:
            delta *= STEP_MAX / d
            next_c = centre + delta
        next_c[0] = np.clip(next_c[0], 0, WORLD_SIZE - 1)
        next_c[1] = np.clip(next_c[1], 0, WORLD_SIZE - 1)
        distance += float(np.linalg.norm(next_c - centre))
        centre = next_c

    return {
        "scenario": scenario_name,
        "algorithm": algo_name,
        "seed": seed,
        "status": status,
        "found": found,
        "detection_iter": int(detection_iter),
        "iterations": iterations,
        "distance": float(distance),
        "wall_time_s": float(time.time() - t_start),
        "n_detections_total": int(n_detections_total),
        "final_confidence": float(confidence_history[-1]),
        "final_entropy": float(entropy_history[-1]),
        "target": list(target),
    }


# --------------------------------------------------- channel models

class Channel:
    """Base class for broadcast channel loss models.

    The bench passes a Channel object to run_comms_channel(). For each
    (sender, receiver, tick) tuple, the channel decides whether the
    broadcast is dropped. Channels MUST be deterministic given (rng,
    sender, receiver, tick) so a single run is reproducible from its
    seed plus the channel construction parameters.
    """

    def drops(self, sender_idx: int, receiver_idx: int,
              tick: int, rng) -> bool:
        raise NotImplementedError

    def label(self) -> str:
        raise NotImplementedError


class IIDChannel(Channel):
    """Independent per-(sender, receiver, tick) Bernoulli drop. The
    original drop_rate model from earlier bench versions; kept for
    backward compatibility."""

    def __init__(self, drop_rate: float):
        self.drop_rate = drop_rate

    def drops(self, sender_idx, receiver_idx, tick, rng):
        return rng.random() < self.drop_rate

    def label(self):
        return f"iid_{self.drop_rate:.2f}"


class GilbertElliottChannel(Channel):
    """Two-state Markov per-link burst loss. Each (sender, receiver) link
    has independent state evolution.

    GOOD state: drop probability = p_good (typically 0)
    BAD state:  drop probability = p_bad  (typically 1)
    transitions: P(G→B) = alpha; P(B→G) = beta

    Mean BAD-state duration ≈ 1/beta ticks. For ~1s blackout at 25 Hz
    (40 ms ticks), beta ≈ 0.04. For ~30s blackout, beta ≈ 0.00133.
    Stationary P(BAD) = alpha / (alpha + beta).

    Reference: Gilbert (1960) BSTJ; Elliott (1963) BSTJ. Standard model
    for bursty packet loss in real radio channels (e.g., Aguayo et al.
    2004 Roofnet shows non-bursty 802.11 mesh; satellite links exhibit
    GE-like bursts during occlusion).
    """

    def __init__(self, p_good=0.0, p_bad=1.0,
                 alpha=0.001, beta=0.05):
        self.p_good = p_good
        self.p_bad = p_bad
        self.alpha = alpha
        self.beta = beta
        self.states = {}                        # (s, r) -> "G" | "B"

    def drops(self, sender_idx, receiver_idx, tick, rng):
        key = (sender_idx, receiver_idx)
        s = self.states.get(key, "G")
        if s == "G" and rng.random() < self.alpha:
            s = "B"
        elif s == "B" and rng.random() < self.beta:
            s = "G"
        self.states[key] = s
        loss_prob = self.p_bad if s == "B" else self.p_good
        return rng.random() < loss_prob

    def label(self):
        return (f"ge_a{self.alpha:.4f}_b{self.beta:.4f}"
                f"_pg{self.p_good:.2f}_pb{self.p_bad:.2f}")


class AsymmetricChannel(Channel):
    """A deaf_fraction subset of receivers is persistently 'deaf' (drops
    every broadcast); remaining receivers have IID drop at base_drop.

    Models LOS-shadowed drones, antenna-pointing failures, geometric
    occlusion where some swarm members can never receive while others
    receive normally. The deaf set is picked deterministically from
    deaf_seed so the channel is reproducible.

    Stronger than uniform IID drop because the architecture cannot
    recover information that never reaches a deaf drone — each deaf
    drone diverges immediately and never re-converges.
    """

    def __init__(self, base_drop: float, deaf_fraction: float,
                 n_drones: int, deaf_seed: int = 0):
        rng = np.random.default_rng(deaf_seed)
        n_deaf = int(round(deaf_fraction * n_drones))
        if n_deaf > 0:
            deaf_idxs = rng.choice(n_drones, size=n_deaf, replace=False)
            self.deaf = set(int(i) for i in deaf_idxs)
        else:
            self.deaf = set()
        self.base_drop = base_drop
        self.deaf_fraction = deaf_fraction

    def drops(self, sender_idx, receiver_idx, tick, rng):
        if receiver_idx in self.deaf:
            return True
        return rng.random() < self.base_drop

    def label(self):
        return f"asym_base{self.base_drop:.2f}_deaf{self.deaf_fraction:.2f}"


# --------------------------------------------------- comms-drop stress

def run_comms_channel(scenario_name: str, algo_name: str,
                      seed: int, channel: Channel) -> dict:
    """Generalized comms-drop run. Takes a Channel object that decides,
    per (sender, receiver, tick), whether each broadcast is dropped.

    All previous IID-drop_rate behaviour is reproduced by passing
    IIDChannel(drop_rate). Gilbert-Elliott burst loss and asymmetric
    deaf-fraction loss are now supported as well.
    """
    return _run_comms_inner(scenario_name, algo_name, seed, channel)


def run_comms_drop(scenario_name: str, algo_name: str,
                   seed: int, drop_rate: float) -> dict:
    """Backward-compatible IID-drop_rate wrapper around run_comms_channel.
    """
    return _run_comms_inner(scenario_name, algo_name, seed,
                            IIDChannel(drop_rate))


def _run_comms_inner(scenario_name: str, algo_name: str,
                     seed: int, channel: Channel) -> dict:
    """Multi-posterior comms run. N drones each maintain their own
    posterior; each broadcast is dropped per the supplied Channel
    object. Unanimity rate and find rate are recorded.

    The swarm's motion uses majority-vote tie-breaking — the
    deterministic-aggregation extension of the byte-identical
    decision protocol under degraded comms.
    """
    from collections import Counter
    t_start = time.time()
    rng = np.random.default_rng(seed)
    scen = SCENARIOS[scenario_name]
    prior = scen["make_prior"]()
    target = sample_target_from_prior(prior, rng)
    initial_log_post = np.log(np.clip(prior, np.exp(LOG_FLOOR), None))
    m0 = initial_log_post.max()
    initial_log_post -= m0 + np.log(np.exp(initial_log_post - m0).sum())
    log_posts = [initial_log_post.copy() for _ in range(N_DRONES)]

    advect = scen.get("advect", False)
    drift_vec = scen.get("drift_vec", (0.0, 0.0))
    drift_noise = scen.get("drift_noise_sigma", 0.0)
    diffuse_sigma = scen.get("diffuse_sigma", 0.0)
    budget = ALGO_BUDGETS.get(algo_name, MAX_MANIFOLDS)

    centre = np.array([WORLD_SIZE / 2.0, WORLD_SIZE / 2.0])
    distance = 0.0
    found = False
    detection_iter = -1
    n_detections_total = 0
    unanimous_count = 0
    n_decisions = 0
    status = "max_manifolds"
    iterations = 0
    state = {"rng": rng, "prior": prior, "target": target,
             "scenario": scen, "budget": budget}
    algo = ALGORITHMS[algo_name]

    for k in range(budget):
        iterations = k + 1
        if time.time() - t_start > RUN_TIMEOUT_S:
            status = "timeout"
            break

        drone_positions = disk_drone_positions(centre)
        observations = []
        for d_pos in drone_positions:
            p_field = detection_field(d_pos)
            p_at_target = float(p_field[target[1], target[0]])
            obs = bool(rng.random() < p_at_target)
            observations.append((obs, d_pos, p_field))
            if obs:
                n_detections_total += 1

        for r_idx in range(N_DRONES):
            # Each drone ALWAYS applies its own observation (a drone's own
            # sensor reading does not pass through the broadcast channel).
            # The channel.drops() check applies only to broadcasts FROM
            # OTHER drones. At iid_100% this becomes the "voting baseline":
            # each drone uses only its own data and the swarm coordinates
            # via majority vote on decisions, not via shared observations.
            own_obs, own_pos, own_p_field = observations[r_idx]
            if own_obs:
                log_posts[r_idx] += np.log(np.maximum(own_p_field, np.exp(LOG_FLOOR)))
            else:
                log_posts[r_idx] += np.log(np.clip(1.0 - own_p_field, np.exp(LOG_FLOOR), 1.0))
            for sender_idx, (obs, d_pos, p_field) in enumerate(observations):
                if sender_idx == r_idx:
                    continue
                if channel.drops(sender_idx, r_idx, k, rng):
                    continue
                if obs:
                    log_posts[r_idx] += np.log(np.maximum(p_field, np.exp(LOG_FLOOR)))
                else:
                    log_posts[r_idx] += np.log(np.clip(1.0 - p_field, np.exp(LOG_FLOOR), 1.0))
            log_posts[r_idx] = np.maximum(log_posts[r_idx], LOG_FLOOR)
            mx = log_posts[r_idx].max()
            log_posts[r_idx] -= mx + np.log(np.exp(log_posts[r_idx] - mx).sum())

        # Detection check via receiver 0; if any drone observed AND that
        # receiver's posterior collapsed onto target, declare found.
        any_det = any(o[0] for o in observations)
        if any_det:
            mc = map_confidence(log_posts[0])
            if mc >= DETECTION_THRESHOLD and detected(map_cell(log_posts[0]), target):
                found = True
                detection_iter = k
                status = "detected"
                break

        decisions = []
        for r_idx in range(N_DRONES):
            d = algo(log_posts[r_idx], centre, state)
            decisions.append((int(d[0]), int(d[1])))

        if all(d == decisions[0] for d in decisions):
            unanimous_count += 1
        n_decisions += 1

        cnt = Counter(decisions)
        next_c = np.array(cnt.most_common(1)[0][0], dtype=float)
        delta = next_c - centre
        d_norm = float(np.linalg.norm(delta))
        if d_norm > STEP_MAX:
            delta *= STEP_MAX / d_norm
            next_c = centre + delta
        next_c[0] = np.clip(next_c[0], 0, WORLD_SIZE - 1)
        next_c[1] = np.clip(next_c[1], 0, WORLD_SIZE - 1)
        distance += float(np.linalg.norm(next_c - centre))
        centre = next_c

        if advect:
            target = advect_target(target, drift_vec, drift_noise, rng)
            state["target"] = target
            for r_idx in range(N_DRONES):
                log_posts[r_idx] = advect_posterior(log_posts[r_idx], drift_vec, diffuse_sigma)

    # Reconstruct an IID drop_rate field for backward compatibility with
    # the existing aggregator; channel models that aren't IID get -1.
    drop_rate_compat = (channel.drop_rate
                        if isinstance(channel, IIDChannel) else -1.0)
    return {
        "scenario": scenario_name,
        "algorithm": algo_name,
        "seed": seed,
        "drop_rate": drop_rate_compat,
        "channel_label": channel.label(),
        "status": status,
        "found": found,
        "detection_iter": int(detection_iter),
        "iterations": iterations,
        "distance": float(distance),
        "wall_time_s": float(time.time() - t_start),
        "n_detections_total": int(n_detections_total),
        "unanimity_rate": float(unanimous_count / max(n_decisions, 1)),
        "n_decisions": int(n_decisions),
        "target": list(target),
    }


# ----------------------------------------------------------- checkpoint

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "bench_search_checkpoint.json")
RESULTS_PATH = os.environ.get("OUT_PATH", "bench_search_results.json")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return {}, {}, {}
    try:
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        primary = {(r["scenario"], r["algorithm"], r["seed"]): r
                   for r in data.get("runs", [])}
        comms = {(r["scenario"], r["algorithm"], r["seed"], r["drop_rate"]): r
                 for r in data.get("comms_runs", [])}
        channel = {(r["scenario"], r["algorithm"], r["seed"],
                    r.get("channel_label", "")): r
                   for r in data.get("channel_runs", [])}
        return primary, comms, channel
    except Exception as e:
        log(f"WARNING: failed to load checkpoint: {e}; starting fresh")
        return {}, {}, {}


def save_checkpoint(runs_by_key, comms_by_key, channel_by_key, config):
    payload = {"config": config,
               "runs": list(runs_by_key.values()),
               "comms_runs": list(comms_by_key.values()),
               "channel_runs": list(channel_by_key.values())}
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


DROP_RATES = [0.0, 0.1, 0.3, 0.5]
COMMS_SCENARIOS = ["lost_at_sea"]
COMMS_ALGORITHMS = ["bayesian", "bayesian_eig"]


def channel_factory(spec: dict, n_drones: int) -> Channel:
    """Construct a Channel from a serializable spec. Spec format:
        {"type": "iid", "drop": 0.3}
        {"type": "ge",  "alpha": 0.043, "beta": 0.1, "p_good": 0.0, "p_bad": 1.0}
        {"type": "asym","base": 0.0, "deaf_fraction": 0.25, "deaf_seed": 0}
    Returning a fresh Channel keeps state isolated per (channel_spec, run).
    """
    t = spec["type"]
    if t == "iid":
        return IIDChannel(drop_rate=float(spec.get("drop", 0.0)))
    if t == "ge":
        return GilbertElliottChannel(
            p_good=float(spec.get("p_good", 0.0)),
            p_bad=float(spec.get("p_bad", 1.0)),
            alpha=float(spec["alpha"]),
            beta=float(spec["beta"]))
    if t == "asym":
        return AsymmetricChannel(
            base_drop=float(spec.get("base", 0.0)),
            deaf_fraction=float(spec["deaf_fraction"]),
            n_drones=n_drones,
            deaf_seed=int(spec.get("deaf_seed", 0)))
    raise ValueError(f"Unknown channel type: {t}")


# Phase C-1 channel sweep set. Stationary P(BAD)=0.3 in GE; α = 0.43·β.
# Burst length is measured in *manifolds* not real seconds — each
# manifold-transition corresponds to a few real seconds depending on
# step distance, so the labelled durations are approximations.
CHANNEL_SWEEPS = [
    {"type": "iid",  "drop": 0.0,  "label": "iid_0%"},
    {"type": "iid",  "drop": 0.3,  "label": "iid_30%"},
    {"type": "iid",  "drop": 0.5,  "label": "iid_50%"},
    {"type": "iid",  "drop": 0.7,  "label": "iid_70%"},
    {"type": "iid",  "drop": 0.9,  "label": "iid_90%"},
    # iid_100% is the "trivial voting baseline" — no broadcasts received.
    # If this performs comparably to iid_0%, voting is the contribution
    # and broadcast-as-shared-state is not. If it craters, the broadcast
    # is doing real work (sharing observations across drones).
    {"type": "iid",  "drop": 1.0,  "label": "iid_100%_voting_baseline"},
    {"type": "ge",   "alpha": 0.21,    "beta": 0.5,   "label": "ge_short_2manifolds"},
    {"type": "ge",   "alpha": 0.043,   "beta": 0.1,   "label": "ge_medium_10manifolds"},
    {"type": "ge",   "alpha": 0.0086,  "beta": 0.02,  "label": "ge_long_50manifolds"},
    {"type": "asym", "base": 0.0,  "deaf_fraction": 0.10, "label": "asym_deaf10%"},
    {"type": "asym", "base": 0.0,  "deaf_fraction": 0.25, "label": "asym_deaf25%"},
    {"type": "asym", "base": 0.0,  "deaf_fraction": 0.50, "label": "asym_deaf50%"},
    {"type": "asym", "base": 0.0,  "deaf_fraction": 1.00, "label": "asym_deaf100%"},
]


def main():
    config = {
        "world_size": WORLD_SIZE, "n_drones": N_DRONES, "n_seeds": N_SEEDS,
        "max_manifolds": MAX_MANIFOLDS, "r_sensor": R_SENSOR,
        "sigma_sensor": SIGMA_SENSOR, "p_max": P_MAX,
        "r_manifold": R_MANIFOLD, "step_max": STEP_MAX,
        "n_directions": N_DIRECTIONS, "detection_threshold": DETECTION_THRESHOLD,
        "detection_kcells": DETECTION_KCELLS,
        "run_timeout_s": RUN_TIMEOUT_S,
        "drop_rates": DROP_RATES,
    }
    log(f"# bench_search config={config}")

    runs_by_key, comms_by_key, channel_by_key = load_checkpoint()
    log(f"# loaded {len(runs_by_key)} primary, {len(comms_by_key)} comms, "
        f"{len(channel_by_key)} channel-sweep from checkpoint")

    todo = []
    for scenario_name in SCENARIOS:
        for algo_name in ALGORITHMS:
            for seed in range(N_SEEDS):
                k = (scenario_name, algo_name, seed)
                if k not in runs_by_key:
                    todo.append(k)

    todo_comms = []
    for scen in COMMS_SCENARIOS:
        for algo in COMMS_ALGORITHMS:
            for dr in DROP_RATES:
                for seed in range(N_SEEDS):
                    k = (scen, algo, seed, dr)
                    if k not in comms_by_key:
                        todo_comms.append(k)

    log(f"# {len(todo)} primary runs, {len(todo_comms)} comms-drop runs")
    t_bench = time.time()

    for i, (scen, algo, seed) in enumerate(todo):
        log(f"## [{i+1}/{len(todo)}] {scen} / {algo} seed={seed}")
        try:
            if algo == "sarops_class":
                result = run_sarops_class(scen, seed)
            elif algo in BC_T_BY_ALGO:
                T = BC_T_BY_ALGO[algo]
                result = run_bandyopadhyay_chung(scen, seed,
                                                 n_consensus_rounds=T)
                result["algorithm"] = algo
                result["n_consensus_rounds"] = T
            else:
                result = run_one(scen, algo, seed)
        except Exception as e:
            result = {
                "scenario": scen, "algorithm": algo, "seed": seed,
                "status": f"error:{type(e).__name__}:{e}",
                "found": False, "detection_iter": -1, "iterations": 0,
                "distance": float("nan"), "wall_time_s": 0.0,
                "n_detections_total": 0, "final_confidence": float("nan"),
                "final_entropy": float("nan"), "target": [-1, -1],
                "traceback": traceback.format_exc(),
            }
            log(f"   ERROR: {e}")
        runs_by_key[(scen, algo, seed)] = result
        save_checkpoint(runs_by_key, comms_by_key, channel_by_key, config)
        log(f"   {result['status']} iter={result['iterations']} "
            f"conf={result['final_confidence']:.3f} dist={result['distance']:.1f}")

    for i, (scen, algo, seed, dr) in enumerate(todo_comms):
        log(f"## comms [{i+1}/{len(todo_comms)}] {scen}/{algo} seed={seed} drop={dr}")
        try:
            result = run_comms_drop(scen, algo, seed, dr)
        except Exception as e:
            result = {
                "scenario": scen, "algorithm": algo, "seed": seed, "drop_rate": dr,
                "status": f"error:{type(e).__name__}:{e}",
                "found": False, "detection_iter": -1, "iterations": 0,
                "distance": float("nan"), "wall_time_s": 0.0,
                "n_detections_total": 0, "unanimity_rate": float("nan"),
                "n_decisions": 0, "target": [-1, -1],
                "traceback": traceback.format_exc(),
            }
            log(f"   ERROR: {e}")
        comms_by_key[(scen, algo, seed, dr)] = result
        save_checkpoint(runs_by_key, comms_by_key, channel_by_key, config)
        log(f"   {result['status']} iter={result['iterations']} "
            f"unanimity={result['unanimity_rate']:.2f}")

    # ----- Phase C-1: Gilbert-Elliott + asymmetric channel sweep
    todo_channel = []
    for scen in COMMS_SCENARIOS:
        for algo in COMMS_ALGORITHMS:
            for ch_spec in CHANNEL_SWEEPS:
                for seed in range(N_SEEDS):
                    k = (scen, algo, seed, ch_spec["label"])
                    if k not in channel_by_key:
                        todo_channel.append((scen, algo, seed, ch_spec))

    log(f"# {len(todo_channel)} channel-sweep runs queued")
    for i, (scen, algo, seed, ch_spec) in enumerate(todo_channel):
        log(f"## channel [{i+1}/{len(todo_channel)}] "
            f"{scen}/{algo} seed={seed} ch={ch_spec['label']}")
        try:
            ch = channel_factory(ch_spec, N_DRONES)
            result = run_comms_channel(scen, algo, seed, ch)
            result["channel_label"] = ch_spec["label"]
            result["channel_spec"] = {k_: v for k_, v in ch_spec.items()
                                      if k_ != "label"}
        except Exception as e:
            result = {
                "scenario": scen, "algorithm": algo, "seed": seed,
                "channel_label": ch_spec["label"],
                "channel_spec": ch_spec,
                "status": f"error:{type(e).__name__}:{e}",
                "found": False, "detection_iter": -1, "iterations": 0,
                "distance": float("nan"), "wall_time_s": 0.0,
                "n_detections_total": 0, "unanimity_rate": float("nan"),
                "n_decisions": 0, "target": [-1, -1],
                "traceback": traceback.format_exc(),
            }
            log(f"   ERROR: {e}")
        channel_by_key[(scen, algo, seed, ch_spec["label"])] = result
        save_checkpoint(runs_by_key, comms_by_key, channel_by_key, config)
        log(f"   {result['status']} iter={result['iterations']} "
            f"unanimity={result.get('unanimity_rate', float('nan')):.2f}")

    log(f"\n# All runs complete. Wall time: {time.time() - t_bench:.1f}s")

    # ----- Aggregate
    summary = []
    for scenario in SCENARIOS:
        print(f"\n# {scenario}  ({SCENARIOS[scenario]['description']})")
        print(f"# N_drones={N_DRONES}  seeds={N_SEEDS}  max_manifolds={MAX_MANIFOLDS}")
        for algo in ALGORITHMS:
            runs = [runs_by_key[(scenario, algo, s)]
                    for s in range(N_SEEDS) if (scenario, algo, s) in runs_by_key]
            if not runs:
                continue
            found = np.array([r["found"] for r in runs])
            iters_when_found = np.array(
                [r["iterations"] for r in runs if r["found"]], dtype=float)
            dist = np.array([r["distance"] for r in runs], dtype=float)
            conf = np.array([r["final_confidence"] for r in runs], dtype=float)
            ent = np.array([r["final_entropy"] for r in runs], dtype=float)

            it_lo, it_hi = bootstrap_ci(iters_when_found) if iters_when_found.size else (0, 0)
            dist_lo, dist_hi = bootstrap_ci(dist)

            print(f"  {algo:>10}  found={found.sum():>3}/{N_SEEDS}  "
                  f"iters_when_found={iters_when_found.mean() if iters_when_found.size else float('nan'):>5.1f} "
                  f"[{it_lo:.1f},{it_hi:.1f}]  "
                  f"dist={dist.mean():>5.1f} [{dist_lo:.1f},{dist_hi:.1f}]  "
                  f"final_conf={conf.mean():.3f}  final_H={ent.mean():.2f}")

            summary.append({
                "scenario": scenario, "algorithm": algo,
                "n_seeds": N_SEEDS,
                "found": int(found.sum()),
                "found_rate": float(found.mean()),
                "iters_when_found_mean": (float(iters_when_found.mean())
                                          if iters_when_found.size else None),
                "iters_when_found_ci": [it_lo, it_hi] if iters_when_found.size else None,
                "distance_mean": float(dist.mean()),
                "distance_ci": [dist_lo, dist_hi],
                "final_confidence_mean": float(conf.mean()),
                "final_entropy_mean": float(ent.mean()),
            })

    # ----- Comms-drop aggregate
    comms_summary = []
    for scen in COMMS_SCENARIOS:
        for algo in COMMS_ALGORITHMS:
            print(f"\n# comms-drop: {scen} / {algo}")
            print(f"# {'drop_rate':>10} {'found':>10} {'iters':>20} {'unanimity':>15}")
            for dr in DROP_RATES:
                rs = [comms_by_key[(scen, algo, s, dr)]
                      for s in range(N_SEEDS) if (scen, algo, s, dr) in comms_by_key]
                if not rs:
                    continue
                found = np.array([r["found"] for r in rs])
                iters_when_found = np.array(
                    [r["iterations"] for r in rs if r["found"]], dtype=float)
                unan = np.array([r["unanimity_rate"] for r in rs], dtype=float)
                it_lo, it_hi = bootstrap_ci(iters_when_found) if iters_when_found.size else (0, 0)
                un_lo, un_hi = bootstrap_ci(unan)
                it_str = (f"{iters_when_found.mean():>5.1f} [{it_lo:.1f},{it_hi:.1f}]"
                          if iters_when_found.size else "—")
                print(f"  {dr:>10.2f} {found.sum():>4}/{N_SEEDS:<5}   {it_str:>20}   "
                      f"{unan.mean():.3f} [{un_lo:.3f},{un_hi:.3f}]")
                comms_summary.append({
                    "scenario": scen, "algorithm": algo, "drop_rate": dr,
                    "n_seeds": N_SEEDS,
                    "found": int(found.sum()),
                    "found_rate": float(found.mean()),
                    "iters_when_found_mean": (float(iters_when_found.mean())
                                              if iters_when_found.size else None),
                    "iters_when_found_ci": [it_lo, it_hi] if iters_when_found.size else None,
                    "unanimity_rate_mean": float(unan.mean()),
                    "unanimity_rate_ci": [un_lo, un_hi],
                })

    # ----- Channel-sweep aggregate (Phase C-1: GE + asymmetric)
    channel_summary = []
    for scen in COMMS_SCENARIOS:
        for algo in COMMS_ALGORITHMS:
            print(f"\n# channel-sweep: {scen} / {algo}")
            print(f"# {'channel':>30} {'found':>10} {'iters':>20} {'unanimity':>15}")
            for ch_spec in CHANNEL_SWEEPS:
                lbl = ch_spec["label"]
                rs = [channel_by_key[(scen, algo, s, lbl)]
                      for s in range(N_SEEDS)
                      if (scen, algo, s, lbl) in channel_by_key]
                if not rs:
                    continue
                found = np.array([r["found"] for r in rs])
                iters_when_found = np.array(
                    [r["iterations"] for r in rs if r["found"]], dtype=float)
                unan = np.array([r["unanimity_rate"] for r in rs], dtype=float)
                it_lo, it_hi = bootstrap_ci(iters_when_found) if iters_when_found.size else (0, 0)
                un_lo, un_hi = bootstrap_ci(unan)
                it_str = (f"{iters_when_found.mean():>5.1f} [{it_lo:.1f},{it_hi:.1f}]"
                          if iters_when_found.size else "—")
                print(f"  {lbl:>30} {found.sum():>4}/{N_SEEDS:<5}   {it_str:>20}   "
                      f"{unan.mean():.3f} [{un_lo:.3f},{un_hi:.3f}]")
                channel_summary.append({
                    "scenario": scen, "algorithm": algo, "channel_label": lbl,
                    "channel_spec": {k_: v for k_, v in ch_spec.items() if k_ != "label"},
                    "n_seeds": N_SEEDS,
                    "found": int(found.sum()),
                    "found_rate": float(found.mean()),
                    "iters_when_found_mean": (float(iters_when_found.mean())
                                              if iters_when_found.size else None),
                    "iters_when_found_ci": [it_lo, it_hi] if iters_when_found.size else None,
                    "unanimity_rate_mean": float(unan.mean()),
                    "unanimity_rate_ci": [un_lo, un_hi],
                })

    out = {"config": config, "summary": summary,
           "comms_summary": comms_summary,
           "channel_summary": channel_summary,
           "runs": list(runs_by_key.values()),
           "comms_runs": list(comms_by_key.values()),
           "channel_runs": list(channel_by_key.values())}
    with tempfile.NamedTemporaryFile(mode="w", dir=".", delete=False, suffix=".tmp") as f:
        json.dump(out, f, indent=2)
        tmp = f.name
    os.replace(tmp, RESULTS_PATH)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
