"""ext30: Bahdanau-style attention over LSTM hidden states."""
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.Wa = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, enc_h):  # enc_h: (N, T, hidden)
        scores = self.v(torch.tanh(self.Wa(enc_h)))  # (N, T, 1)
        weights = torch.softmax(scores, dim=1)
        ctx = (weights * enc_h).sum(dim=1)           # (N, hidden)
        return ctx, weights


if __name__ == "__main__":
    h = torch.randn(8, 12, 32)
    attn = Attention(32)
    ctx, w = attn(h)
    print("context:", ctx.shape, "weights sum per sample:", w.sum(1).mean().item())
