"""
Practical 13: Gradient Descent Variants Comparison
Objective: Implement Batch GD, SGD, and Mini-Batch GD on a linear regression
problem. Plot loss reduction vs CPU time and weight trajectories on a 2D contour.
"""

import numpy as np
import time
import matplotlib.pyplot as plt


def make_data(n=100, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    true_w = np.array([3.0, -2.0])
    y = X.dot(true_w) + rng.randn(n) * 0.1
    return X, y


def loss(X, y, w):
    return np.mean((X.dot(w) - y) ** 2)


def grad(X, y, w):
    return 2 * X.T.dot(X.dot(w) - y) / len(X)


def batch_gd(X, y, lr=0.05, epochs=200):
    w = np.zeros(X.shape[1])
    traj, losses, times = [w.copy()], [], []
    t0 = time.perf_counter()
    for _ in range(epochs):
        w -= lr * grad(X, y, w)
        traj.append(w.copy())
        losses.append(loss(X, y, w))
        times.append(time.perf_counter() - t0)
    return np.array(traj), losses, times


def sgd(X, y, lr=0.05, epochs=200):
    w = np.zeros(X.shape[1])
    traj, losses, times = [w.copy()], [], []
    t0 = time.perf_counter()
    n = len(X)
    for _ in range(epochs):
        for i in np.random.permutation(n):
            w -= lr * grad(X[i:i + 1], y[i:i + 1], w)
        traj.append(w.copy())
        losses.append(loss(X, y, w))
        times.append(time.perf_counter() - t0)
    return np.array(traj), losses, times


def minibatch_gd(X, y, lr=0.05, epochs=200, bs=16):
    w = np.zeros(X.shape[1])
    traj, losses, times = [w.copy()], [], []
    t0 = time.perf_counter()
    n = len(X)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            w -= lr * grad(X[b], y[b], w)
        traj.append(w.copy())
        losses.append(loss(X, y, w))
        times.append(time.perf_counter() - t0)
    return np.array(traj), losses, times


def main():
    X, y = make_data()
    tr_b, l_b, t_b = batch_gd(X, y)
    tr_s, l_s, t_s = sgd(X, y)
    tr_m, l_m, t_m = minibatch_gd(X, y)

    # contour map of loss
    w1 = np.linspace(-1, 5, 100)
    w2 = np.linspace(-5, 1, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Z = np.array([[loss(X, y, np.array([a, b])) for a in w1] for b in w2])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (tr, title) in zip(axes, [(tr_b, "Batch GD"), (tr_s, "SGD"), (tr_m, "Mini-Batch GD")]):
        ax.contour(W1, W2, Z, levels=40, cmap="viridis")
        ax.plot(tr[:, 0], tr[:, 1], "r.-", lw=1, markersize=3)
        ax.set_title(title)
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")

    plt.figure(figsize=(7, 4))
    plt.plot(t_b, l_b, label="Batch GD")
    plt.plot(t_s, l_s, label="SGD")
    plt.plot(t_m, l_m, label="Mini-Batch GD")
    plt.xlabel("CPU time (s)")
    plt.ylabel("loss")
    plt.title("Loss vs training time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
