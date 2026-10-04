"""ext42: Mode collapse visualization / mitigation tracking during GAN training."""
import numpy as np
import torch


def mode_collapse_metric(generated, n_modes=2, eps=0.5):
    """Fraction of distinct modes covered by generated samples (low => collapse)."""
    centers = [np.array([2, 2]), np.array([-2, -2])]
    covered = set()
    for g in generated:
        for i, c in enumerate(centers):
            if np.linalg.norm(g - c) < eps:
                covered.add(i)
    return len(covered) / len(centers)


if __name__ == "__main__":
    collapsed = np.tile(np.array([2.0, 2.0]), (100, 1))   # all same mode
    diverse = np.vstack([np.random.randn(50, 2) + 2,
                         np.random.randn(50, 2) - 2])
    print("coverage collapsed:", mode_collapse_metric(collapsed))
    print("coverage diverse  :", mode_collapse_metric(diverse))
