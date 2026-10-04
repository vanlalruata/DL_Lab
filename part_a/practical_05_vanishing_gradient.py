"""
Practical 5: The Vanishing Gradient Simulation in Deep Feedforward Networks
Objective: Build an N-layer forward pass in pure NumPy. Propagate an initial
gradient backward through 10 hidden layers using Sigmoid vs ReLU. Plot the
gradient magnitude at each layer to show Sigmoid decay toward zero.
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def backprop_grads(weights, activations, act_grad_fn, init_grad=1.0):
    """weights: list of W_l (each shape [out, in]); activations: list of pre-activation inputs z_l.
    Returns list of gradient magnitudes at each layer (product of incoming grad
    and local derivative)."""
    grad = init_grad
    magnitudes = []
    for w, z in zip(reversed(weights), reversed(activations)):
        local = act_grad_fn(z)            # derivative at this layer
        grad = grad * np.mean(local) * np.mean(np.abs(w))
        magnitudes.append(abs(grad))
    return magnitudes


def main():
    np.random.seed(0)
    n_layers = 10
    dim = 64
    # random fixed weights and random pre-activation inputs per layer
    weights = [np.random.randn(dim, dim) * 0.5 for _ in range(n_layers)]
    z_sigmoid = [np.random.randn(dim) for _ in range(n_layers)]
    z_relu = [np.random.randn(dim) + 0.5 for _ in range(n_layers)]  # positive-ish for ReLU

    sig_mag = backprop_grads(weights, z_sigmoid, sigmoid_grad)
    relu_mag = backprop_grads(weights, z_relu, relu_grad)

    layers = list(range(1, n_layers + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(layers, sig_mag, marker="o", label="Sigmoid")
    plt.plot(layers, relu_mag, marker="s", label="ReLU")
    plt.yscale("log")
    plt.xlabel("Layer (counted from output backward)")
    plt.ylabel("Gradient magnitude (log)")
    plt.title("Vanishing Gradient: Sigmoid vs ReLU over 10 layers")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    print(f"Sigmoid grad after 10 layers: {sig_mag[-1]:.3e}")
    print(f"ReLU grad after 10 layers:    {relu_mag[-1]:.3e}")


if __name__ == "__main__":
    main()
