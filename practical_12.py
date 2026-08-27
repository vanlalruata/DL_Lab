"""
Practical 12: General N-Layer MLP Engine in NumPy
Objective: Construct a flexible NumPy class that accepts arbitrary layer
topologies (e.g., [4, 16, 8, 3]). Implement automated forward, backward, and
mini-batch gradient updates to classify the Iris dataset.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def sigmoid_grad(a):  # a is the activation (sigmoid output)
    return a * (1 - a)


class MLP:
    def __init__(self, layers, lr=0.05):
        self.lr = lr
        self.layers = layers
        # weights[l]: from layer l to l+1  shape [layers[l+1], layers[l]]
        self.W = [np.random.randn(layers[i + 1], layers[i]) * 0.5
                  for i in range(len(layers) - 1)]
        self.b = [np.zeros(layers[i + 1]) for i in range(len(layers) - 1)]

    def forward(self, x):
        self.acts = [x]
        a = x
        for i in range(len(self.W) - 1):
            z = np.dot(a, self.W[i].T) + self.b[i]
            a = sigmoid(z)
            self.acts.append(a)
        # output layer: softmax
        z = np.dot(a, self.W[-1].T) + self.b[-1]
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        a = exp / np.sum(exp, axis=1, keepdims=True)
        self.acts.append(a)
        return a

    def backward(self, y):
        m = y.shape[0]
        grads_w, grads_b = [], []
        d = self.acts[-1] - y            # softmax + CCE gradient
        for i in reversed(range(len(self.W))):
            a_prev = self.acts[i]
            gW = np.dot(d.T, a_prev) / m   # [out, in]
            gb = np.mean(d, axis=0)
            grads_w.insert(0, gW)
            grads_b.insert(0, gb)
            if i > 0:
                d = np.dot(d, self.W[i]) * sigmoid_grad(self.acts[i])
        for i in range(len(self.W)):
            self.W[i] -= self.lr * grads_w[i]
            self.b[i] -= self.lr * grads_b[i]

    def train(self, X, Y, epochs=200, batch_size=16):
        losses = []
        m = X.shape[0]
        for ep in range(epochs):
            idx = np.random.permutation(m)
            for s in range(0, m, batch_size):
                bidx = idx[s:s + batch_size]
                p = self.forward(X[bidx])
                loss = -np.mean(np.sum(Y[bidx] * np.log(p + 1e-12), axis=1))
                self.backward(Y[bidx])
            p = self.forward(X)
            losses.append(-np.mean(np.sum(Y * np.log(p + 1e-12), axis=1)))
        return losses


def main():
    data = load_iris()
    X = data.data
    Y = OneHotEncoder(sparse_output=False).fit_transform(data.target.reshape(-1, 1))
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.3, random_state=1)

    model = MLP(layers=[4, 16, 8, 3], lr=0.05)
    losses = model.train(X_tr, Y_tr, epochs=200, batch_size=16)
    pred = np.argmax(model.forward(X_te), axis=1)
    true = np.argmax(Y_te, axis=1)
    acc = np.mean(pred == true)
    print(f"Iris test accuracy: {acc:.3f}")

    plt.plot(losses)
    plt.title("Iris MLP training loss")
    plt.xlabel("epoch")
    plt.ylabel("CCE loss")
    plt.show()


if __name__ == "__main__":
    main()
