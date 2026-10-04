"""Practical 23 - 3-Layer DNN from scratch (NumPy) and vanishing gradients.

Two hidden layers implemented fully in NumPy. We compare the gradient magnitudes
through the layers for Sigmoid vs ReLU activations to demonstrate why ReLU
mitigates the vanishing-gradient problem in deeper networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def relu(x):
    return np.maximum(0, x)


def run(act_name):
    np.random.seed(0)
    X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
    X = X.T.astype(np.float64)
    Y = y.reshape(1, -1).astype(np.float64)

    act = sigmoid if act_name == "sigmoid" else relu
    dact = (lambda a: a * (1 - a)) if act_name == "sigmoid" else (lambda a: (a > 0).astype(float))

    # 2 -> 8 -> 4 -> 1
    W1 = np.random.randn(8, 2) * 0.5
    b1 = np.zeros((8, 1))
    W2 = np.random.randn(4, 8) * 0.5
    b2 = np.zeros((4, 1))
    W3 = np.random.randn(1, 4) * 0.5
    b3 = np.zeros((1, 1))

    # forward
    Z1 = W1 @ X + b1; A1 = act(Z1)
    Z2 = W2 @ A1 + b2; A2 = act(Z2)
    Z3 = W3 @ A2 + b3; A3 = sigmoid(Z3)

    # backward with dL/dZ3 = A3 - Y
    dZ3 = A3 - Y
    dW3 = dZ3 @ A2.T / X.shape[1]
    dA2 = W3.T @ dZ3
    dZ2 = dA2 * dact(A2); dW2 = dZ2 @ A1.T / X.shape[1]
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * dact(A1); dW1 = dZ1 @ X.T / X.shape[1]

    return [np.linalg.norm(d) for d in (dW1, dW2, dW3)]


def main():
    layers = ["layer1 (2->8)", "layer2 (8->4)", "layer3 (4->1)"]
    g_sig = run("sigmoid")
    g_relu = run("relu")
    for n, s, r in zip(layers, g_sig, g_relu):
        print(f"{n:18s} | sigmoid grad={s:.4f}  relu grad={r:.4f}")

    plt.figure()
    x = np.arange(len(layers))
    plt.plot(x, g_sig, "o-", label="Sigmoid")
    plt.plot(x, g_relu, "s-", label="ReLU")
    plt.xticks(x, layers)
    plt.yscale("log")
    plt.ylabel("gradient norm (log)")
    plt.title("practical_23: gradient flow in 3-layer DNN")
    plt.legend()
    plt.savefig("part_a/figures/practical_23_gradients.png")
    plt.show()


if __name__ == "__main__":
    main()
