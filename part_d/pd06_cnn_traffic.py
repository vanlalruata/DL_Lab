"""part_d / pd06 - 1D-CNN network-traffic classification (deep learning).

Treats a flow as a time-series of N feature snapshots and uses a 1D ConvNet
(PyTorch) to classify benign vs malicious traffic. Reports train/val accuracy &
loss, ROC, and per-sample inference latency.
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


def gen_seq(n=2000, T=20, F=5, seed=0):
    rng = np.random.RandomState(seed)
    nn_ = n // 2
    normal = rng.normal(0, 1, (nn_, T, F))
    na = n - nn_
    attack = np.concatenate([rng.normal(0, 1, (na, T // 2, F)),
                            rng.normal(3, 1, (na, T - T // 2, F))], axis=1)
    X = np.vstack([normal, attack]); y = np.hstack([np.zeros(nn_), np.ones(na)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class Conv1DNet(nn.Module):
    def __init__(self, F=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(F, 16, 3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x.transpose(1, 2))  # (N,T,F)->(N,F,T)


def main():
    X, y = gen_seq()
    ds = TensorDataset(X, y)
    tr, va, te = torch.utils.data.random_split(ds, [1200, 400, 400])
    tr_dl = DataLoader(tr, 64, shuffle=True); va_dl = DataLoader(va, 128); te_dl = DataLoader(te, 128)
    model = Conv1DNet().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(); tr_a, va_a, tr_l, va_l = [], [], [], []
    for ep in range(8):
        model.train(); ca = cl = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            cl += loss.item() * len(yb); ca += (model(xb).argmax(1) == yb).sum().item()
        tr_l.append(cl / len(tr)); tr_a.append(ca / len(tr))
        model.eval(); vc = vl = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb); vl += crit(out, yb).item() * len(yb)
                vc += (out.argmax(1) == yb).sum().item()
        va_l.append(vl / len(va)); va_a.append(vc / len(va))
    print(f"final tr_acc={tr_a[-1]:.3f} val_acc={va_a[-1]:.3f}")

    model.eval(); all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            all_p.append(torch.softmax(model(xb.to(DEVICE)), 1).cpu()); all_y.append(yb)
    proba = torch.cat(all_p).numpy(); yte = torch.cat(all_y).numpy()
    auc = roc_auc_score(yte, proba[:, 1])
    fpr, tpr, _ = roc_curve(yte, proba[:, 1])
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("pd06 1D-CNN traffic ROC"); plt.legend()
    plt.savefig(f"{FIG}/pd06_roc.png"); plt.close()
    plt.figure()
    plt.subplot(1, 2, 1); plt.plot(tr_a, label="train"); plt.plot(va_a, label="val"); plt.ylabel("acc"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(tr_l, label="train"); plt.plot(va_l, label="val"); plt.ylabel("loss"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/pd06_curves.png"); plt.close()

    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in te_dl:
            model(xb.to(DEVICE))
    dt = time.perf_counter() - t0
    print(f"1D-CNN AUC={auc:.4f}; infer {dt*1000:.2f} ms / {len(te.dataset)} samples")
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
