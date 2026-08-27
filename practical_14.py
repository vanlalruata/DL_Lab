"""
Practical 14: Momentum & Nesterov Accelerated Gradient (NAG)
Objective: Implement Polyak Momentum and NAG in NumPy. Optimize a 2D pathological
ravine (Rosenbrock) and visualize how Momentum/NAG dampen transverse oscillations.
"""

import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(w):
    x, y = w[0], w[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(w):
    x, y = w[0], w[1]
    gx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    gy = 200 * (y - x ** 2)
    return np.array([gx, gy])


def momentum(w0, lr=1e-3, mu=0.9, steps=5000):
    w = np.array(w0, float)
    v = np.zeros_like(w)
    traj = [w.copy()]
    for _ in range(steps):
        g = rosenbrock_grad(w)
        v = mu * v - lr * g
        w += v
        traj.append(w.copy())
    return np.array(traj)


def nesterov(w0, lr=1e-3, mu=0.9, steps=5000):
    w = np.array(w0, float)
    v = np.zeros_like(w)
    traj = [w.copy()]
    for _ in range(steps):
        g = rosenbrock_grad(w + mu * v)
        v = mu * v - lr * g
        w += v
        traj.append(w.copy())
    return np.array(traj)


def main():
    traj_m = momentum([-1.5, 2.0], lr=1e-3, mu=0.9, steps=8000)
    traj_n = nesterov([-1.5, 2.0], lr=1e-3, mu=0.9, steps=8000)

    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[rosenbrock([a, b]) for a in x] for b in y])
    Z = np.log1p(Z)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, traj, title in [(axes[0], traj_m, "Polyak Momentum"),
                            (axes[1], traj_n, "Nesterov Accelerated Gradient")]:
        ax.contour(X, Y, Z, levels=50, cmap="viridis")
        ax.plot(traj[:, 0], traj[:, 1], "r.-", lw=1, markersize=2)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    plt.suptitle("Optimizing Rosenbrock with Momentum / NAG")
    plt.show()


if __name__ == "__main__":
    main()
