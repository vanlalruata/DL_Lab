"""part_d / pd07 - LSTM/GRU sequential intrusion detection (deep learning).

Sequence of flow records is fed to an LSTM (optionally GRU) to detect attacks
that only become evident over time (slow scans, low-rate DoS). Reports accuracy,
loss, ROC, and inference latency.
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


def gen_seq(n=2000, T=24, F=5, seed=0):
    rng = np.random.RandomState(seed)
    nn_ = n // 2
    normal = rng.normal(0, 1, (nn_, T, F))
    na = n - nn_
    # attack emerges in the second half (low-rate, only visible over time)
    base = rng.normal(0, 1, (na, T, F))
    base[:, T // 2:, :] += rng.normal(1.5, 0.5, (na, T - T // 2, F))
    attack = base
    X = np.vstack([normal, attack]); y = np.hstack([np.zeros(nn_), np.ones(na)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class RNN_IDS(nn.Module):
    def __init__(self, F=5, hidden=32, cell="LSTM"):
        super().__init__()
        self.cell = cell
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn(F, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def run(cell):
    X, y = gen_seq()
    ds = TensorDataset(X, y)
    tr, va, te = torch.utils.data.random_split(ds, [1200, 400, 400])
    tr_dl = DataLoader(tr, 64, shuffle=True); va_dl = DataLoader(va, 128); te_dl = DataLoader(te, 128)
    model = RNN_IDS(cell=cell).to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(8):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
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
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in te_dl:
            model(xb.to(DEVICE))
    dt = time.perf_counter() - t0
    print(f"{cell}: AUC={auc:.4f} infer {dt*1000:.2f} ms / {len(te.dataset)} samples")
    fpr, tpr, _ = roc_curve(yte, proba[:, 1])
    return auc, fpr, tpr


def main():
    res = {c: run(c) for c in ["LSTM", "GRU"]}
    plt.figure()
    for c, (auc, fpr, tpr) in res.items():
        plt.plot(fpr, tpr, label=f"{c} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("pd07 RNN sequential IDS ROC"); plt.legend(); plt.savefig(f"{FIG}/pd07_roc.png"); plt.close()
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
