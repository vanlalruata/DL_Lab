"""
Practical 3: Demonstrating the Linear Separability Constraint
Objective: Test the Perceptron on a non-linearly separable dataset (XOR / 
concentric circles). Visualize how the Perceptron fails to converge, plotting
the perpetual oscillations in classification loss and boundary updates.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


class Perceptron:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.w = None
        self.b = 0.0

    def step(self, x):
        return np.where(x >= 0, 1, 0)

    def fit_record(self, X, y, epochs=200):
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        loss_history = []
        w_history = [self.w.copy()]
        for _ in range(epochs):
            errors = 0
            for i in range(n_samples):
                pred = self.step(np.dot(self.w, X[i]) + self.b)
                update = self.lr * (y[i] - pred)
                if update != 0:
                    errors += 1
                    self.w += update * X[i]
                    self.b += update
            # misclassification count as "loss"
            preds = self.step(np.dot(X, self.w) + self.b)
            loss_history.append(np.mean(preds != y))
            w_history.append(self.w.copy())
        return loss_history, w_history


def make_xor(n=200, noise=0.05, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2) * 2 - 1
    y = np.where(X[:, 0] * X[:, 1] >= 0, 0, 1).astype(int)
    X += rng.randn(n, 2) * noise
    return X, y


def main():
    # XOR dataset
    X, y = make_xor(200, seed=1)
    model = Perceptron(lr=0.01)
    loss_history, w_history = model.fit_record(X, y, epochs=200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: data scatter
    axes[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="red", s=20, label="0")
    axes[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="blue", s=20, label="1")
    axes[0].set_title("Non-linearly separable XOR data")
    axes[0].legend()

    # Right: oscillating loss (never reaches 0 -> no convergence)
    axes[1].plot(loss_history, color="purple")
    axes[1].set_title("Perceptron misclassification rate (never converges on XOR)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("error rate")

    plt.tight_layout()
    plt.show()

    print(f"Final error rate after {len(loss_history)} epochs: "
          f"{loss_history[-1]:.3f} (stays > 0 -> no convergence)")


if __name__ == "__main__":
    main()
