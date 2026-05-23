# /// script
# dependencies = ["numpy<3", "scipy"]
# ///
"""Test compute_manifold_targets in isolation."""

import numpy as np

from lattice import hcp_positions
from mission import compute_manifold_targets


def main() -> None:
    spacing = 10.0
    rng = np.random.default_rng(2000)
    pos0 = hcp_positions(3, 3, spacing=spacing)
    n = pos0.shape[0]
    print(f"original positions: shape={pos0.shape}, centroid={pos0.mean(axis=0)}")
    print(f"position[0]={pos0[0]}, position[55]={pos0[55]}, position[-1]={pos0[-1]}")

    # Simulate post-MOVE positions: original + 20m in x + perturbation
    current = pos0 + np.array([20.0, 0.0, 0.0]) + rng.normal(scale=0.02 * spacing, size=pos0.shape)
    print(f"current centroid: {current.mean(axis=0)}")

    alive = np.ones(n, dtype=bool)
    centroid = current.mean(axis=0)
    heading = np.array([1.0, 0.0, 0.0])

    targets = compute_manifold_targets(alive, centroid, heading, spacing, current_positions=current)
    print(f"targets centroid: {targets.mean(axis=0)}")

    offsets = targets - current
    errs = np.linalg.norm(offsets, axis=1)
    print(f"per-drone offset: min={errs.min():.4f}m mean={errs.mean():.4f}m max={errs.max():.4f}m")

    # Identify the worst-matched drones
    worst = np.argsort(errs)[-5:]
    for w in worst:
        print(f"  drone {w}: current={current[w]} target={targets[w]} err={errs[w]:.3f}m")

    # Sanity check: what if we just use identity assignment (each drone to its own original HCP slot translated)?
    ideal_targets = pos0 + np.array([20.0, 0.0, 0.0])
    ideal_offsets = ideal_targets - current
    ideal_errs = np.linalg.norm(ideal_offsets, axis=1)
    print(f"\nideal identity-assignment offset: max={ideal_errs.max():.4f}m mean={ideal_errs.mean():.4f}m")


if __name__ == "__main__":
    main()
