"""
Practical 2: Single-Layer Perceptron Learning Algorithm (PLA)
Objective: Code the original Rosenblatt Perceptron with the step activation
function. Generate a linearly separable 2D dataset, train the perceptron, and
plot the evolving decision boundary at each epoch until convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class Perceptron:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.w = None
        self.b = 0.0

    def step(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y, epochs=100, record_every=1):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        boundaries = []  # store (w, b) snapshots per epoch
        for epoch in range(epochs):
            errors = 0
            for i in range(n_samples):
                xi = X[i]
                target = y[i]
                pred = self.step(np.dot(self.w, xi) + self.b)
                update = self.lr * (target - pred)
                if update != 0:
                    errors += 1
                    self.w += update * xi
                    self.b += update
            if epoch % record_every == 0:
                boundaries.append((self.w.copy(), self.b))
            if errors == 0:
                boundaries.append((self.w.copy(), self.b))
                print(f"Converged at epoch {epoch}")
                break
        return boundaries

    def predict(self, X):
        return self.step(np.dot(X, self.w) + self.b)


def main():
    X, y = make_blobs(n_samples=200, n_features=2, centers=2,
                      cluster_std=1.2, random_state=42)
    y = np.where(y == 0, 0, 1)  # ensure labels {0,1}

    model = Perceptron(lr=0.01)
    boundaries = model.fit(X, y, epochs=100)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="red", label="class 0", s=20)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="blue", label="class 1", s=20)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx = np.linspace(x_min, x_max, 100)
    cmap = plt.cm.viridis
    for idx, (w, b) in enumerate(boundaries):
        if abs(w[1]) < 1e-9:
            continue
        yy = -(w[0] * xx + b) / w[1]
        ax.plot(xx, yy, color=cmap(idx / max(1, len(boundaries) - 1)),
                alpha=0.6, lw=1.2)
    # final boundary (thick)
    w, b = boundaries[-1]
    yy = -(w[0] * xx + b) / w[1]
    ax.plot(xx, yy, color="black", lw=2.5, label="final boundary")
    ax.set_title("Perceptron evolving decision boundary")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
