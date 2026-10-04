"""
Practical 8: Binary Cross-Entropy (BCE) vs MSE for Binary Classification
Objective: Build a single Sigmoid neuron to classify a binary target y in {0,1}.
Compare loss surface convexities by plotting loss landscapes of MSE vs BCE over a
range of possible weight and bias values.
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bce_loss(y, p):
    # clip to avoid log(0)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def mse_loss(y, p):
    return np.mean((y - p) ** 2)


def main():
    np.random.seed(0)
    n = 50
    X = np.random.randn(n, 2)
    # simple linear target
    true_w = np.array([1.5, -1.0])
    y = (np.dot(X, true_w) + 0.2 > 0).astype(float)

    w1s = np.linspace(-3, 3, 60)
    w2s = np.linspace(-3, 3, 60)
    W1, W2 = np.meshgrid(w1s, w2s)

    def loss_for(bias, loss_fn):
        Z = W1 * X[:, 0].mean() + W2 * X[:, 1].mean() + bias  # approximate surface
        # better: compute average loss across samples for each (w1,w2)
        total = np.zeros_like(W1)
        for i in range(n):
            z = W1 * X[i, 0] + W2 * X[i, 1] + bias
            total += loss_fn(y[i], sigmoid(z))
        return total / n

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (bias, fn, title) in zip(
        axes,
        [(0.2, bce_loss, "BCE loss surface"),
         (0.2, mse_loss, "MSE loss surface")],
    ):
        surf = loss_for(bias, fn)
        cp = ax.contourf(W1, W2, surf, levels=40, cmap="viridis")
        ax.contour(W1, W2, surf, levels=15, colors="white", alpha=0.4)
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")
        ax.set_title(title)
        fig.colorbar(cp, ax=ax)
    plt.suptitle("Loss landscapes: BCE (convex/smooth) vs MSE (flatter, slower)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
