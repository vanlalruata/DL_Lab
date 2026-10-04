"""part_d / pd03 - Cloud workload anomaly detection with an autoencoder.

Synthesises cloud telemetry (CPU, mem, net I/O, API call rate, auth failures)
for benign behaviour, injects anomalies (crypto-mining, brute-force, exfil).
Train an autoencoder on benign only; reconstruction error is the anomaly score.
Plots loss curve, error distribution, and ROC of anomaly detection.
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn

FIG = os.path.join(os.path.dirname(__file__), "figures")


def gen_cloud(n_norm=4000, n_anom=800, seed=0):
    rng = np.random.RandomState(seed)
    norm = np.column_stack([
        rng.normal(40, 10, n_norm),     # CPU %
        rng.normal(50, 8, n_norm),      # mem %
        rng.normal(30, 6, n_norm),      # net I/O
        rng.poisson(50, n_norm),        # API rate
        rng.poisson(1, n_norm),         # auth failures
    ])
    anom = np.column_stack([
        rng.normal(95, 3, n_anom),      # mining: saturated CPU
        rng.normal(80, 5, n_anom),
        rng.normal(85, 5, n_anom),
        rng.poisson(400, n_anom),       # brute/exfil: burst API
        rng.poisson(40, n_anom),
    ])
    X = np.vstack([norm, anom]); y = np.hstack([np.zeros(n_norm), np.ones(n_anom)])
    return X, y


class AE(nn.Module):
    def __init__(self, d=5):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 8), nn.ReLU(), nn.Linear(8, 3), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def main():
    X, y = gen_cloud()
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    # train only on benign
    X_norm = torch.tensor(Xs[y == 0], dtype=torch.float32)
    model = AE(); opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.MSELoss()
    losses = []
    for ep in range(60):
        opt.zero_grad(); rec = model(X_norm); loss = crit(rec, X_norm)
        loss.backward(); opt.step(); losses.append(loss.item())
    plt.figure(); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("recon MSE")
    plt.title("pd03 autoencoder training loss"); plt.savefig(f"{FIG}/pd03_loss.png"); plt.close()

    model.eval()
    with torch.no_grad():
        rec = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
    err = np.mean((rec - Xs) ** 2, axis=1)        # anomaly score
    auc = roc_auc_score(y, err)
    fpr, tpr, _ = roc_curve(y, err)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("pd03 cloud anomaly ROC"); plt.legend()
    plt.savefig(f"{FIG}/pd03_roc.png"); plt.close()

    plt.figure(); plt.hist(err[y == 0], bins=40, alpha=0.6, label="normal")
    plt.hist(err[y == 1], bins=40, alpha=0.6, label="anomaly"); plt.yscale("log")
    plt.xlabel("reconstruction error"); plt.legend(); plt.title("pd03 error distribution")
    plt.savefig(f"{FIG}/pd03_error_dist.png"); plt.close()

    t0 = time.perf_counter()
    with torch.no_grad():
        model(torch.tensor(Xs[:1000], dtype=torch.float32))
    dt = time.perf_counter() - t0
    print(f"Cloud anomaly AUC={auc:.4f}; inference {dt*1000:.2f} ms / 1000 samples")
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
