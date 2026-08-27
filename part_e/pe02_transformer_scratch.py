"""part_e / pe02 - Transformer encoder from scratch (PyTorch).

Implements positional encoding, multi-head self-attention, and a small encoder
Transformer for sequence classification, trained on a synthetic variable-length
sequence task. Reports train/val accuracy, loss, and visualizes attention. All
pure PyTorch (no HF dependency).
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


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        den = torch.exp(torch.arange(0, d, 2).float() * -(np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * den); pe[:, 1::2] = torch.cos(pos * den)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttn(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.h = h; self.dk = d // h
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk)
        q = qkv[:, :, 0].transpose(1, 2)  # (B, h, T, dk)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / self.dk ** 0.5  # (B, h, T, T)
        attn = scores.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)   # (B, T, h*dk)
        return self.proj(out), attn


class TransformerEncoder(nn.Module):
    def __init__(self, d=32, h=4):
        super().__init__()
        self.pe = PositionalEncoding(d)
        self.attn = MultiHeadAttn(d, h)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        x = self.pe(x)
        x, attn = self.attn(x)
        x = x + self.ff(x)
        return self.head(x.mean(1)), attn


def gen_seq(n=1500, T=16, d=32, seed=0):
    rng = np.random.RandomState(seed)
    nn_ = n // 2
    normal = rng.normal(0, 1, (nn_, T, d))
    attack = rng.normal(2.0, 1, (n - nn_, T, d))
    X = np.vstack([normal, attack]); y = np.hstack([np.zeros(nn_), np.ones(n - nn_)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def main():
    from sklearn.model_selection import train_test_split
    X, y = gen_seq()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=1)
    model = TransformerEncoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
    tr_a, va_a, tr_l, va_l = [], [], [], []
    Xtr_t, ytr_t, Xva_t, yva_t = (Xtr.to(DEVICE), ytr.to(DEVICE), Xva.to(DEVICE), yva.to(DEVICE))
    for _ in range(30):
        model.train(); opt.zero_grad(); loss = crit(model(Xtr_t)[0], ytr_t); loss.backward(); opt.step()
        tr_l.append(loss.item()); tr_a.append((model(Xtr_t)[0].argmax(1) == ytr_t).float().mean().item())
        model.eval()
        with torch.no_grad():
            o = model(Xva_t)[0]
        va_l.append(crit(o, yva_t).item()); va_a.append((o.argmax(1) == yva_t).float().mean().item())
    print(f"final tr_acc={tr_a[-1]:.3f} val_acc={va_a[-1]:.3f}")

    plt.figure()
    plt.subplot(1, 2, 1); plt.plot(tr_a, label="train"); plt.plot(va_a, label="val"); plt.ylabel("acc"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(tr_l, label="train"); plt.plot(va_l, label="val"); plt.ylabel("loss"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/pe02_curves.png"); plt.close()

    # visualize attention of one head on a sample
    model.eval()
    with torch.no_grad():
        _, attn = model(Xva_t[:1])
    plt.figure(); plt.imshow(attn[0, 0].cpu().numpy(), cmap="viridis")
    plt.title("pe02 attention (head 0)"); plt.colorbar(); plt.savefig(f"{FIG}/pe02_attention.png"); plt.close()
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
