"""ext25: GRU cell from scratch; compare gate structure to LSTM."""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GRUCell:
    def __init__(self, in_s, hidden):
        s = np.sqrt(1.0 / hidden)
        self.Wz = np.random.randn(hidden, in_s + hidden) * s  # update gate
        self.Wr = np.random.randn(hidden, in_s + hidden) * s  # reset gate
        self.Wh = np.random.randn(hidden, in_s + hidden) * s

    def step(self, x, h):
        xh = np.concatenate([x, h])
        z = sigmoid(self.Wz @ xh)              # how much of new vs old
        r = sigmoid(self.Wr @ xh)              # reset memory
        h_hat = np.tanh(self.Wh @ np.concatenate([x, r * h]))
        h = (1 - z) * h + z * h_hat            # no separate cell state (vs LSTM)
        return h


if __name__ == "__main__":
    cell = GRUCell(1, 16)
    h = np.zeros(16)
    for _ in range(5):
        h = cell.step(np.array([0.5]), h)
    print("GRU hidden shape:", h.shape)
