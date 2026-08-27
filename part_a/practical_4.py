"""
Practical 4: Activation Function Zoo and Gradient Visualizer
Objective: Implement forward and derivative functions for Sigmoid, Tanh, ReLU,
Leaky ReLU, ELU, and SELU. Plot a 2x3 grid comparing each activation function
side-by-side with its first derivative over x in [-5, 5].
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_grad(x, alpha=1.0):
    return np.where(x > 0, 1.0, alpha * np.exp(x))


def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


def selu_grad(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, 1.0, alpha * np.exp(x))


def main():
    x = np.linspace(-5, 5, 500)
    funcs = [
        ("Sigmoid", sigmoid, sigmoid_grad),
        ("Tanh", tanh, tanh_grad),
        ("ReLU", relu, relu_grad),
        ("Leaky ReLU", lambda v: leaky_relu(v), lambda v: leaky_relu_grad(v)),
        ("ELU", lambda v: elu(v), lambda v: elu_grad(v)),
        ("SELU", lambda v: selu(v), lambda v: selu_grad(v)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (name, f, g) in zip(axes.flat, funcs):
        ax.plot(x, f(x), label="f(x)", lw=2)
        ax.plot(x, g(x), label="f'(x)", lw=2, linestyle="--")
        ax.set_title(name)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Activation Functions and Their Derivatives")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
