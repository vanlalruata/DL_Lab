"""Practical 21 - 2-Layer DNN (1 hidden layer) forward pass from scratch (NumPy).

Build a 2-layer fully-connected network (input -> hidden -> output) using only
NumPy. Demonstrates weight initialization, forward propagation through a hidden
layer with a non-linear activation, and visualises the decision boundary
*before* any training - showing the random initial output.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def relu(x):
    return np.maximum(0, x)


def main():
    np.random.seed(0)
    X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
    X = X.T  # shape (2, N)

    n_in, n_h, n_out = 2, 8, 1
    W1 = np.random.randn(n_h, n_in) * 0.5   # (hidden, in)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_out, n_h) * 0.5
    b2 = np.zeros((n_out, 1))

    # forward pass
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)              # output (1, N)

    # decision boundary at random init
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    A1g = relu(W1 @ grid + b1)
    out = (sigmoid(W2 @ A1g + b2) > 0.5).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, out, alpha=0.3, cmap="RdBu")
    plt.scatter(X[0], X[1], c=y, cmap="RdBu", s=12)
    plt.title("practical_21: random-init decision boundary (no training)")
    plt.savefig("part_a/figures/practical_21_random_boundary.png")
    plt.show()

    print("Layer shapes:")
    print(f"  X   : {X.shape}")
    print(f"  W1  : {W1.shape} -> Z1: {Z1.shape} -> A1: {A1.shape}")
    print(f"  W2  : {W2.shape} -> Z2: {Z2.shape} -> A2: {A2.shape}")
    print(f"  hidden params: {W1.size + b1.size}; output params: {W2.size + b2.size}")


if __name__ == "__main__":
    main()
