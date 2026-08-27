"""part_d / pd08 - Adversarial robustness of an ML-based IDS (FGSM attack).

Trains a DNN intrusion classifier, then crafts FGSM adversarial examples to evade
detection and measures the clean vs adversarial accuracy drop — a core deep-learning
security topic (adversarial ML). Plots accuracy under increasing perturbation eps.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gen_feat(n=3000, F=5, seed=0):
    rng = np.random.RandomState(seed)
    nn_ = n // 2
    normal = rng.normal(0, 1, (nn_, F))
    attack = rng.normal(2.5, 1, (n - nn_, F))
    X = np.vstack([normal, attack]); y = np.hstack([np.zeros(nn_), np.ones(n - nn_)])
    return X, y


class DNN(nn.Module):
    def __init__(self, F=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(F, 32), nn.ReLU(),
                                nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    return (x + eps * x.grad.sign()).detach()


def main():
    X, y = gen_feat()
    sc = StandardScaler().fit(X)
    Xt = torch.tensor(sc.transform(X), dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    model = DNN().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(40):
        opt.zero_grad(); nn.CrossEntropyLoss()(model(Xt), yt).backward(); opt.step()

    model.eval()
    clean_acc = (model(Xt).argmax(1) == yt).float().mean().item()
    epsilons = [0.0, 0.05, 0.1, 0.2, 0.4]
    adv_acc = []
    for eps in epsilons:
        xa = fgsm(model, Xt, yt, eps)
        a = (model(xa).argmax(1) == yt).float().mean().item()
        adv_acc.append(a)
        print(f"eps={eps:.2f} -> adversarial accuracy={a:.3f}")
    print(f"clean accuracy={clean_acc:.3f}")

    plt.figure(); plt.plot(epsilons, adv_acc, "o-", label="adversarial")
    plt.axhline(clean_acc, ls="--", label="clean"); plt.xlabel("FGSM epsilon")
    plt.ylabel("accuracy"); plt.title("pd08 IDS adversarial robustness"); plt.legend()
    plt.savefig(f"{FIG}/pd08_adversarial.png"); plt.close()
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
