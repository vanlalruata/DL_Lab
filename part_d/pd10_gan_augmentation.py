"""part_d / pd10 - GAN-based synthetic attack-traffic augmentation (deep learning).

Trains a GAN to generate synthetic attack flow vectors, then augments a
label-limited attack training set and compares classifier ROC-AUC with vs without
augmentation (data-augmentation for security ML where attack samples are scarce).
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gen_feat(n=4000, F=5, seed=0, attack_frac=0.5):
    rng = np.random.RandomState(seed)
    n_a = int(n * attack_frac)
    normal = rng.normal(0, 1, (n - n_a, F))
    attack = rng.normal(2.5, 1, (n_a, F))
    X = np.vstack([normal, attack]); y = np.hstack([np.zeros(n - n_a), np.ones(n_a)])
    return X, y


class Generator(nn.Module):
    def __init__(self, z=8, F=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z, 16), nn.ReLU(),
                                nn.Linear(16, F), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, F=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(F, 16), nn.ReLU(),
                                nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


def train_gan(G, D, real, epochs=300):
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3)
    for _ in range(epochs):
        z = torch.randn(len(real), 8, device=DEVICE)
        fake = G(z).detach()
        opt_d.zero_grad()
        d_loss = nn.functional.binary_cross_entropy_with_logits(D(real), torch.ones(len(real), 1, device=DEVICE)) \
            + nn.functional.binary_cross_entropy_with_logits(D(fake), torch.zeros(len(real), 1, device=DEVICE))
        d_loss.backward(); opt_d.step()
        z = torch.randn(len(real), 8, device=DEVICE)
        opt_g.zero_grad()
        g_loss = nn.functional.binary_cross_entropy_with_logits(D(G(z)), torch.ones(len(real), 1, device=DEVICE))
        g_loss.backward(); opt_g.step()
    return G


def classifier_auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=DEVICE), \
               torch.tensor(sc.transform(Xte), dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    model = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 2)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(60):
        opt.zero_grad(); nn.CrossEntropyLoss()(model(Xtr), ytr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(Xte), 1)[:, 1].cpu().numpy()
    return roc_auc_score(yte, proba), proba


def main():
    # scarce-attack scenario: only 50 attack samples in training
    X, y = gen_feat(4000, 5, seed=1, attack_frac=0.0125)  # ~50 attacks
    Xte, yte = gen_feat(1500, 5, seed=9)
    sc = StandardScaler().fit(X)
    Xr = torch.tensor(sc.transform(X), dtype=torch.float32, device=DEVICE)
    attack_idx = np.where(y == 1)[0]
    G = Generator().to(DEVICE); D = Discriminator().to(DEVICE)
    G = train_gan(G, D, Xr[attack_idx], epochs=300)

    # generate synthetic attacks
    z = torch.randn(500, 8, device=DEVICE)
    synth = G(z).detach().cpu().numpy() * sc.scale_ + sc.mean_  # invert standardization
    Xaug = np.vstack([X, synth]); yaug = np.hstack([y, np.ones(500)])

    auc_base, _ = classifier_auc(X, y, Xte, yte)
    auc_aug, proba_aug = classifier_auc(Xaug, yaug, Xte, yte)
    print(f"Baseline (scarce attacks) AUC = {auc_base:.4f}")
    print(f"GAN-augmented          AUC = {auc_aug:.4f}")

    fpr, tpr, _ = roc_curve(yte, proba_aug)
    plt.figure(); plt.plot(fpr, tpr, label=f"augmented AUC={auc_aug:.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("pd10 GAN-augmented IDS ROC"); plt.legend(); plt.savefig(f"{FIG}/pd10_roc.png"); plt.close()
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
