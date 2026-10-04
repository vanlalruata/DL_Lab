"""Practical 22 - 2-Layer DNN with manual backpropagation from scratch (NumPy).

A complete 2-2-1 fully-connected network trained with hand-coded forward and
backward passes (no autograd, no torch). Uses sigmoid output with binary
cross-entropy so the gradient of the loss w.r.t. the logits is simply (a2 - y).
Solves the XOR-like structure of the make_moons dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def sigmoid_grad(a):
    return a * (1 - a)


def main():
    np.random.seed(0)
    X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
    X = X.T.astype(np.float64)
    Y = y.reshape(1, -1).astype(np.float64)

    n_in, n_h, n_out = 2, 8, 1
    W1 = np.random.randn(n_h, n_in) * 0.5
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_out, n_h) * 0.5
    b2 = np.zeros((n_out, 1))

    lr = 0.1
    losses = []
    for ep in range(2000):
        # forward
        Z1 = W1 @ X + b1
        A1 = sigmoid(Z1)
        Z2 = W2 @ A1 + b2
        A2 = sigmoid(Z2)

        # loss (BCE)
        eps = 1e-9
        loss = -np.mean(Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps))
        losses.append(loss)

        # backward
        dZ2 = A2 - Y                       # (1, N)  (sigmoid + BCE combined)
        dW2 = dZ2 @ A1.T / X.shape[1]      # (1, h)
        db2 = np.mean(dZ2, axis=1, keepdims=True)
        dA1 = W2.T @ dZ2
        dZ1 = dA1 * sigmoid_grad(A1)
        dW1 = dZ1 @ X.T / X.shape[1]
        db1 = np.mean(dZ1, axis=1, keepdims=True)

        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1

        if ep % 500 == 0:
            acc = float(np.mean((A2 > 0.5) == Y))
            print(f"epoch {ep:4d}  loss={loss:.4f}  acc={acc:.3f}")

    print(f"final loss={losses[-1]:.4f}")

    # decision boundary
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    A1g = sigmoid(W1 @ grid + b1)
    out = (sigmoid(W2 @ A1g + b2) > 0.5).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, out, alpha=0.3, cmap="RdBu")
    plt.scatter(X[0], X[1], c=Y.ravel(), cmap="RdBu", s=12)
    plt.title("practical_22: trained 2-layer DNN (manual backprop)")
    plt.savefig("part_a/figures/practical_22_boundary.png")
    plt.show()

    plt.figure()
    plt.plot(losses)
    plt.xlabel("epoch"); plt.ylabel("BCE loss")
    plt.title("practical_22: training loss (manual backprop)")
    plt.savefig("part_a/figures/practical_22_loss.png")
    plt.show()


if __name__ == "__main__":
    main()
