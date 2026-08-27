"""part_d / pd09 - Transformer-based cloud/IoT anomaly detection (deep learning).

Uses a Transformer encoder with self-attention over a time-series of telemetry
snapshots to flag anomalies (e.g., crypto-mining, exfiltration). Mean-pooled
[CLS]-style representation feeds a classifier; reports ROC and inference latency.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gen_seq(n=3000, T=16, F=5, seed=0):
    rng = np.random.RandomState(seed)
    nn_ = n // 2
    normal = rng.normal(0, 1, (nn_, T, F))
    na = n - nn_
    anom = np.copy(normal[:0]) if False else rng.normal(0, 1, (na, T, F))
    anom[:, :, 0] += np.linspace(0, 4, T)        # drifting CPU spike
    anom[:, :, 3] += rng.normal(3, 1, (na, T))
    X = np.vstack([normal, anom]); y = np.hstack([np.zeros(nn_), np.ones(na)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class TransformerIDS(nn.Module):
    def __init__(self, F=5, d_model=32, nhead=4, layers=2):
        super().__init__()
        self.in_proj = nn.Linear(F, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.in_proj(x)                       # (N,T,d_model)
        h = self.tf(h)
        return self.head(h.mean(dim=1))          # mean-pool over time


def main():
    X, y = gen_seq()
    ds = TensorDataset(X, y)
    tr, va, te = torch.utils.data.random_split(ds, [1800, 600, 600])
    tr_dl = DataLoader(tr, 64, shuffle=True); va_dl = DataLoader(va, 128); te_dl = DataLoader(te, 128)
    model = TransformerIDS().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(8):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval(); vc = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                vc += (model(xb.to(DEVICE)).argmax(1) == yb.to(DEVICE)).sum().item()
    model.eval(); all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            all_p.append(torch.softmax(model(xb.to(DEVICE)), 1).cpu()); all_y.append(yb)
    proba = torch.cat(all_p).numpy(); yte = torch.cat(all_y).numpy()
    auc = roc_auc_score(yte, proba[:, 1])
    fpr, tpr, _ = roc_curve(yte, proba[:, 1])
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("pd09 Transformer anomaly ROC"); plt.legend()
    plt.savefig(f"{FIG}/pd09_roc.png"); plt.close()
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in te_dl:
            model(xb.to(DEVICE))
    dt = time.perf_counter() - t0
    print(f"Transformer AUC={auc:.4f}; infer {dt*1000:.2f} ms / {len(te.dataset)} samples")
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
