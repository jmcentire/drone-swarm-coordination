# /// script
# dependencies = ["numpy<3"]
# ///
"""3D HCP (hexagonal close-packed) lattice generator and neighbor graph.

Each interior node has 12 nearest neighbors at the lattice spacing. Edge nodes
have fewer; the design relies on every drone having at least 4 non-coplanar
neighbors for 3D trilateration.

Layer structure (AB stacking):
  A layers: triangular lattice in xy
  B layers: triangular lattice shifted by (1/2, sqrt(3)/6) into hollows of A
  z spacing: spacing * sqrt(2/3)

The neighbor graph is built by brute-force distance threshold rather than
crystallographic lookup, so it generalizes to perturbed (post-PBD) positions.
"""

from __future__ import annotations

import numpy as np

SQRT_3_OVER_2 = np.sqrt(3.0) / 2.0
SQRT_3_OVER_6 = np.sqrt(3.0) / 6.0
SQRT_2_OVER_3 = np.sqrt(2.0 / 3.0)


def hcp_positions(hex_radius: int, n_layers: int, spacing: float = 1.0) -> np.ndarray:
    """Generate HCP lattice positions in a hexagonal-prism patch.

    hex_radius: number of rings around center in each layer.
        0 -> 1 atom, 1 -> 7, 2 -> 19, 3 -> 37, 4 -> 61, 5 -> 91, 6 -> 127.
    n_layers: number of vertical layers (alternates A/B).
    spacing: nearest-neighbor distance.

    Returns: (N, 3) float array.
    """
    z_step = spacing * SQRT_2_OVER_3
    layer_shifts = [
        (0.0, 0.0),
        (0.5 * spacing, spacing * SQRT_3_OVER_6),
    ]

    positions: list[tuple[float, float, float]] = []
    for layer in range(n_layers):
        shift_x, shift_y = layer_shifts[layer % 2]
        z = layer * z_step
        for i in range(-hex_radius, hex_radius + 1):
            for j in range(-hex_radius, hex_radius + 1):
                if abs(i + j) > hex_radius:
                    continue
                x = (i + 0.5 * j) * spacing + shift_x
                y = j * spacing * SQRT_3_OVER_2 + shift_y
                positions.append((x, y, z))
    return np.asarray(positions, dtype=np.float64)


def build_neighbor_graph(
    positions: np.ndarray, comms_range: float
) -> list[list[int]]:
    """Build adjacency list: drone i's neighbors are within comms_range.

    O(N^2). Fine for N < ~1000.
    """
    n = positions.shape[0]
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    neighbors = [
        np.where(dists[i] <= comms_range)[0].tolist() for i in range(n)
    ]
    return neighbors


def degree_distribution(neighbors: list[list[int]]) -> np.ndarray:
    return np.array([len(ns) for ns in neighbors])


def non_coplanar_neighbors(
    positions: np.ndarray, neighbors: list[list[int]], min_volume: float = 1e-6
) -> np.ndarray:
    """For each drone, count how many of its neighbors span a non-degenerate
    3D frame (i.e. enough independent directions for 3D trilateration).

    Returns: (N,) int array. A value >= 4 means the drone can in principle
    resolve its 3D position from range measurements alone.

    Uses SVD rank check: if the (N_i, 3) matrix of neighbor offsets has 3
    singular values above sqrt(min_volume) * spacing, the neighbors span R^3.
    """
    n = positions.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ns = neighbors[i]
        if len(ns) < 4:
            counts[i] = 0
            continue
        offsets = positions[ns] - positions[i]
        # SVD on offsets: rank-3 iff three non-trivial singular values
        s = np.linalg.svd(offsets, compute_uv=False)
        if len(s) >= 3 and s[2] > np.sqrt(min_volume):
            counts[i] = len(ns)
        else:
            counts[i] = 0
    return counts


if __name__ == "__main__":
    # Smoke test: 3-ring × 3-layer lattice should have 37*3 = 111 drones.
    pos = hcp_positions(hex_radius=3, n_layers=3, spacing=1.0)
    print(f"positions: {pos.shape}")
    ngh = build_neighbor_graph(pos, comms_range=1.1)
    deg = degree_distribution(ngh)
    print(f"degree: min={deg.min()} max={deg.max()} mean={deg.mean():.2f}")
    print(f"degree histogram: {np.bincount(deg)}")
    ncp = non_coplanar_neighbors(pos, ngh)
    print(
        f"drones with >=4 non-coplanar neighbors: "
        f"{(ncp >= 4).sum()}/{len(pos)}"
    )
