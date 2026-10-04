"""
Practical 17: Modern Optimizer Variants (AdamW & AdaDelta)
Objective: Implement AdamW (decoupled weight decay) and AdaDelta (learning-rate
free). Compare weight norm decay between standard Adam with L2 penalty vs AdamW
over 100 epochs.
"""

import numpy as np
import matplotlib.pyplot as plt


class AdamL2:
    """Standard Adam with L2 weight decay folded into the gradient."""
    def __init__(self, lr=0.001, wd=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.wd = lr, wd
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, w, g):
        if self.m is None:
            self.m = np.zeros_like(w); self.v = np.zeros_like(w)
        self.t += 1
        g = g + self.wd * w                      # L2 penalty inside gradient
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g ** 2)
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return w - self.lr * mh / (np.sqrt(vh) + self.eps)


class AdamW:
    """Adam with decoupled weight decay."""
    def __init__(self, lr=0.001, wd=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.wd = lr, wd
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, w, g):
        if self.m is None:
            self.m = np.zeros_like(w); self.v = np.zeros_like(w)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g ** 2)
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        w = w - self.lr * mh / (np.sqrt(vh) + self.eps)
        w = w - self.lr * self.wd * w           # decoupled decay
        return w


class AdaDelta:
    """Learning-rate-free AdaDelta."""
    def __init__(self, rho=0.95, eps=1e-8):
        self.rho, self.eps = rho, eps
        self.Eg2 = self.Ex2 = None

    def step(self, w, g):
        if self.Eg2 is None:
            self.Eg2 = np.zeros_like(w); self.Ex2 = np.zeros_like(w)
        self.Eg2 = self.rho * self.Eg2 + (1 - self.rho) * g ** 2
        dw = (np.sqrt(self.Ex2 + self.eps) / np.sqrt(self.Eg2 + self.eps)) * g
        self.Ex2 = self.rho * self.Ex2 + (1 - self.rho) * dw ** 2
        return w - dw


def make_grad():
    return lambda w: 2 * (w - np.array([3.0] * 5))


def main():
    g = make_grad()
    opt_l2 = AdamL2(lr=0.001, wd=0.01)
    opt_w = AdamW(lr=0.001, wd=0.01)
    w_l2 = np.random.randn(5) + 1.0
    w_w = w_l2.copy()
    norms_l2, norms_w = [], []
    for _ in range(100):
        w_l2 = opt_l2.step(w_l2, g(w_l2))
        w_w = opt_w.step(w_w, g(w_w))
        norms_l2.append(np.linalg.norm(w_l2))
        norms_w.append(np.linalg.norm(w_w))

    plt.figure(figsize=(7, 4))
    plt.plot(norms_l2, label="Adam + L2 penalty")
    plt.plot(norms_w, label="AdamW (decoupled)")
    plt.xlabel("epoch")
    plt.ylabel("weight norm ||w||")
    plt.title("Weight norm decay: Adam+L2 vs AdamW (100 epochs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
