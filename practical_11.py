"""
Practical 11: 2-Layer MLP Backpropagation from Scratch (Solving XOR)
Objective: Build a 2-2-1 MLP (2 inputs -> 2 hidden sigmoid -> 1 output sigmoid).
Derive and code manual backpropagation (dL/dW1, dL/db1, dL/dW2, dL/db2) to learn XOR.
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


class MLP2Layer:
    def __init__(self):
        # 2 hidden neurons, 1 output neuron
        self.W1 = np.random.randn(2, 2) * 0.5   # shape [hidden, input]
        self.b1 = np.zeros(2)
        self.W2 = np.random.randn(1, 2) * 0.5   # shape [output, hidden]
        self.b2 = np.zeros(1)

    def forward(self, x):
        self.z1 = np.dot(self.W1, x) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, x, y, lr=0.5):
        # output error
        d_z2 = (self.a2 - y) * sigmoid_grad(self.z2)   # shape [1]
        dW2 = np.outer(d_z2, self.a1)                  # [1,2]
        db2 = d_z2
        # hidden error
        d_a1 = np.dot(self.W2.T, d_z2)                 # [2]
        d_z1 = d_a1 * sigmoid_grad(self.z1)            # [2]
        dW1 = np.outer(d_z1, x)                        # [2,2]
        db1 = d_z1
        # gradient descent update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def loss(self, x, y):
        return 0.5 * (self.a2 - y) ** 2


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)

    model = MLP2Layer()
    epochs = 20000
    losses = []
    for ep in range(epochs):
        total = 0
        for i in range(4):
            model.forward(X[i])
            total += model.loss(X[i], Y[i])
            model.backward(X[i], Y[i], lr=0.5)
        losses.append(total / 4)

    print("Final predictions:")
    for i in range(4):
        p = model.forward(X[i])
        print(f"  XOR({X[i].astype(int)}) = {p[0]:.3f} (target {Y[i][0]})")

    plt.plot(losses)
    plt.title("XOR training loss (manual backprop)")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
