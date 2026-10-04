"""ext28: Sequence-to-sequence character-level text generation (PyTorch)."""
import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, vocab, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.rnn = nn.RNN(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, h=None):
        x = self.emb(x)
        out, h = self.rnn(x, h)
        return self.fc(out), h


def make_batch(text, seq_len=10, step=3):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    idx = [c2i[c] for c in text]
    xs, ys = [], []
    for i in range(0, len(idx) - seq_len, step):
        xs.append(idx[i:i + seq_len])
        ys.append(idx[i + 1:i + seq_len + 1])
    return torch.tensor(xs), torch.tensor(ys), len(chars)


if __name__ == "__main__":
    text = "hello world hello deep learning hello neural network"
    xs, ys, V = make_batch(text)
    m = CharRNN(V)
    out, _ = m(xs[:2])
    print("next-char logits shape:", out.shape)
