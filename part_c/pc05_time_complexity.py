"""part_c / pc05 - Inference time-complexity study (scaling analysis).

Measures how inference latency scales with (a) number of samples and (b) model
width, to demonstrate O(N) / O(params) behaviour empirically. Uses a small MLP on
synthetic tabular data (no download). Also reports accuracy/loss and a ROC curve.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn

FIG = os.path.join(os.path.dirname(__file__), "figures")


def make_data(n, d, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = (X[:, :3].sum(1) + rng.randn(n) * 0.3 > 0).astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=1)


class MLP(nn.Module):
    def __init__(self, d, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, width), nn.ReLU(),
                                nn.Linear(width, width), nn.ReLU(),
                                nn.Linear(width, 2))

    def forward(self, x):
        return self.net(x)


def time_inference(model, X):
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32)
    for _ in range(3):
        model(xb)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(xb)
    return (time.perf_counter() - t0) * 1000  # ms


def main():
    # (a) scaling with number of samples
    sizes = [100, 1000, 5000, 20000]
    latencies = []
    for n in sizes:
        X_tr, X_te, y_tr, y_te = make_data(n, 20)
        m = MLP(20, 64)
        m.fit = None
        # quick train
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        xt, yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr)
        for _ in range(20):
            opt.zero_grad(); loss = nn.CrossEntropyLoss()(m(xt), yt); loss.backward(); opt.step()
        latencies.append(time_inference(m, X_te))

    plt.figure(figsize=(5, 4))
    plt.plot(sizes, latencies, "o-")
    plt.xlabel("test samples"); plt.ylabel("inference time (ms)")
    plt.title("Inference time vs dataset size (linear scaling)")
    plt.savefig(f"{FIG}/pc05_scale_samples.png"); plt.close()
    print("latency vs n:", list(zip(sizes, [round(l, 2) for l in latencies])))

    # (b) scaling with model width
    X_tr, X_te, y_tr, y_te = make_data(4000, 20)
    widths = [16, 64, 256, 1024]
    w_lat, w_params = [], []
    for w in widths:
        m = MLP(20, w)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        xt, yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr)
        for _ in range(20):
            opt.zero_grad(); loss = nn.CrossEntropyLoss()(m(xt), yt); loss.backward(); opt.step()
        w_lat.append(time_inference(m, X_te))
        w_params.append(sum(p.numel() for p in m.parameters()))

    plt.figure(figsize=(5, 4))
    plt.plot(widths, w_lat, "s-"); plt.xlabel("hidden width"); plt.ylabel("inference (ms)")
    plt.title("Inference time vs model width"); plt.savefig(f"{FIG}/pc05_scale_width.png"); plt.close()

    # final ROC on widest model
    m = MLP(20, 256)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    xt, yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr)
    for _ in range(50):
        opt.zero_grad(); loss = nn.CrossEntropyLoss()(m(xt), yt); loss.backward(); opt.step()
    m.eval()
    proba = torch.softmax(m(torch.tensor(X_te, dtype=torch.float32)), 1)[:, 1].detach().numpy()
    auc = roc_auc_score(y_te, proba)
    fpr, tpr, _ = roc_curve(y_te, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.2f}"); plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("pc05 ROC"); plt.legend()
    plt.savefig(f"{FIG}/pc05_roc.png"); plt.close()
    print(f"Final model AUC={auc:.3f}; params={w_params[-1]:,}")
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
