"""
Practical 16: Complete Implementation of the Adam Optimizer
Objective: Code Adam: first moment m_t, second moment v_t, and bias corrections
m_hat, v_hat. Plot weight trajectories and show the smoothing impact of bias
correction during the first 10 update steps.
"""

import numpy as np
import matplotlib.pyplot as plt


class Adam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, w, g):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g ** 2)
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def make_quad_grad():
    return lambda w: 2 * (w - np.array([3.0, 2.0]))


def main():
    opt = Adam(lr=0.1)
    w = np.array([0.0, 0.0], float)
    g = make_quad_grad()
    traj = [w.copy()]
    m_hats = []
    v_hats = []
    for _ in range(50):
        w = opt.step(w, g(w))
        traj.append(w.copy())

    # bias correction inspection for first 10 steps
    m_seq, mhat_seq = [], []
    opt2 = Adam(lr=1.0)
    w2 = np.array([0.0], float)
    g2 = lambda x: np.array([1.0])
    for i in range(10):
        w2 = opt2.step(w2, g2(w2))
        m_seq.append(opt2.m.copy()[0])
        mhat_seq.append((opt2.m / (1 - opt2.b1 ** opt2.t))[0])

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 11), m_seq, "o-", label="m_t (biased)")
    plt.plot(range(1, 11), mhat_seq, "s-", label="m_hat_t (bias corrected)")
    plt.title("Effect of bias correction in first 10 Adam steps")
    plt.xlabel("step")
    plt.legend()

    plt.figure(figsize=(6, 5))
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], "r.-", markersize=3)
    plt.scatter([3], [2], c="green", marker="*", s=200, label="optimum")
    plt.title("Adam weight trajectory")
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
