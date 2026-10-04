"""part_e / pe04 - Decoder-only GPT-style model from scratch (char-level).

Implements a minimal GPT (token+positional embeddings, masked self-attention,
causal LM head) trained on a built-in corpus, then generates text. Pure PyTorch.
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

FIG = os.path.join(os.path.dirname(__file__), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS = (
    "the quick brown fox jumps over the lazy dog the dog barks and the fox runs "
    "a cat sits on the mat a dog walks in the park the sun rises in the east the "
    "moon shines at night the river flows to the sea the bird sings a song the "
    "child plays with the fox and the dog the fox is quick the dog is lazy "
)


class CausalAttention(nn.Module):
    def __init__(self, d, h=2):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)

    def forward(self, x):
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk)
        q = qkv[:, :, 0].transpose(1, 2)  # (B, h, T, dk)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / self.dk ** 0.5  # (B, h, T, T)
        mask = torch.triu(torch.ones(T, T), 1).bool().to(scores.device)
        scores.masked_fill_(mask, float("-inf"))
        attn = scores.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)  # (B, T, h*dk)
        return out


class MiniGPT(nn.Module):
    def __init__(self, vocab, d=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d); self.pe = nn.Parameter(torch.zeros(1, 64, d))
        self.attn = CausalAttention(d, 2)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        x = self.emb(x) + self.pe[:, :x.size(1)]
        x = x + self.attn(x); x = x + self.ff(x)
        return self.head(x)


def main():
    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in CORPUS])
    seq_len = 24
    idx = torch.tensor(np.array([data[i:i + seq_len + 1] for i in range(0, len(data) - seq_len - 1, 1)]))
    model = MiniGPT(len(chars)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2); crit = nn.CrossEntropyLoss()
    losses = []
    for step in range(300):
        b = idx[torch.randperm(len(idx))[:16]].to(DEVICE)
        x, y = b[:, :-1], b[:, 1:]
        loss = crit(model(x).view(-1, len(chars)), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print("final loss:", round(losses[-1], 3))
    plt.figure(); plt.plot(losses); plt.xlabel("step"); plt.ylabel("cross-entropy")
    plt.title("pe04 GPT training loss"); plt.savefig(f"{FIG}/pe04_loss.png"); plt.close()

    # generate
    model.eval(); ctx = torch.tensor([stoi["t"]]).unsqueeze(0).to(DEVICE)
    out = []
    for _ in range(40):
        with torch.no_grad():
            logits = model(ctx[:, -seq_len:])[:, -1]
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        out.append(itos[nxt.item()]); ctx = torch.cat([ctx, nxt], 1)
    print("generated:", "".join(out))


if __name__ == "__main__":
    main()
