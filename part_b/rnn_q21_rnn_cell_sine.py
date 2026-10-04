"""ext21: Vanilla RNN cell from scratch; train on sine-wave prediction."""
import numpy as np


class RNNCell:
    def __init__(self, in_s, hidden, lr=1e-2):
        self.Wxh = np.random.randn(hidden, in_s) * 0.1
        self.Whh = np.random.randn(hidden, hidden) * 0.1
        self.bh = np.zeros(hidden)
        self.Why = np.random.randn(in_s, hidden) * 0.1
        self.by = np.zeros(in_s)
        self.lr = lr

    def step(self, x, h):
        h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
        y = self.Why @ h + self.by
        return y, h


def make_sine(seq_len=50):
    t = np.linspace(0, 10, seq_len)
    return np.sin(t).reshape(seq_len, 1)


if __name__ == "__main__":
    cell = RNNCell(1, 16)
    data = make_sine()
    h = np.zeros(16)
    # one unrolled forward pass
    for t in range(len(data) - 1):
        y, h = cell.step(data[t], h)
    print("predicted next value shape:", y.shape)
