"""
Practical 15: Adaptive Learning Rates (AdaGrad vs RMSProp)
Objective: Implement AdaGrad and RMSProp from scratch. Run both on an objective
with sparse/frequent gradients to show AdaGrad stalls (accumulated squares) while
RMSProp keeps learning.
"""

import numpy as np
import matplotlib.pyplot as plt


def adagrad(w0, grad_fn, lr=0.1, eps=1e-8, steps=2000):
    w = np.array(w0, float)
    cache = np.zeros_like(w)
    traj = [w.copy()]
    for _ in range(steps):
        g = grad_fn(w)
        cache += g ** 2
        w -= lr / (np.sqrt(cache) + eps) * g
        traj.append(w.copy())
    return np.array(traj)


def rmsprop(w0, grad_fn, lr=0.1, beta=0.9, eps=1e-8, steps=2000):
    w = np.array(w0, float)
    cache = np.zeros_like(w)
    traj = [w.copy()]
    for _ in range(steps):
        g = grad_fn(w)
        cache = beta * cache + (1 - beta) * g ** 2
        w -= lr / (np.sqrt(cache) + eps) * g
        traj.append(w.copy())
    return np.array(traj)


def main():
    # sparse/frequent gradient objective: a simple convex quadratic but with a
    # feature that is active rarely. We model gradient sparsity directly.
    def make_grad(sparse_idx):
        def grad(w):
            g = 2 * (w - np.array([3.0, 1.0]))
            # zero out one coordinate most of the time (sparse signal)
            if np.random.rand() < 0.9:
                g[sparse_idx] = 0.0
            return g
        return grad

    traj_a = adagrad([0.0, 0.0], make_grad(1), lr=0.5, steps=3000)
    traj_r = rmsprop([0.0, 0.0], make_grad(1), lr=0.5, steps=3000)

    # distance to optimum over time (coordinate 1 is sparse)
    dist_a = np.linalg.norm(traj_a - np.array([3.0, 1.0]), axis=1)
    dist_r = np.linalg.norm(traj_r - np.array([3.0, 1.0]), axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(dist_a, label="AdaGrad")
    plt.plot(dist_r, label="RMSProp")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("distance to optimum (log)")
    plt.title("AdaGrad stalls on sparse gradients; RMSProp keeps learning")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
