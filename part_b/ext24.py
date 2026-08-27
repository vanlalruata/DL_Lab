"""ext24: LSTM cell from scratch; explain forget/input/output gates."""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTMCell:
    def __init__(self, in_s, hidden):
        s = np.sqrt(1.0 / hidden)
        self.Wf = np.random.randn(hidden, in_s + hidden) * s
        self.Wi = np.random.randn(hidden, in_s + hidden) * s
        self.Wo = np.random.randn(hidden, in_s + hidden) * s
        self.Wc = np.random.randn(hidden, in_s + hidden) * s

    def step(self, x, h, c):
        xh = np.concatenate([x, h])
        f = sigmoid(self.Wf @ xh)   # forget gate: what to drop from cell
        i = sigmoid(self.Wi @ xh)   # input gate: what new to store
        o = sigmoid(self.Wo @ xh)   # output gate: what to expose
        g = np.tanh(self.Wc @ xh)   # candidate cell state
        c = f * c + i * g           # updated cell state
        h = o * np.tanh(c)         # new hidden state
        return h, c


if __name__ == "__main__":
    cell = LSTMCell(1, 16)
    h, c = np.zeros(16), np.zeros(16)
    for _ in range(5):
        h, c = cell.step(np.array([0.5]), h, c)
    print("LSTM hidden shape after steps:", h.shape)
