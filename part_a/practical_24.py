"""Practical 24 - 2-Layer DNN in PyTorch (same architecture as practical_22).

Now using torch.autograd. The same forward/backward logic as practical_22 but
expressed with nn.Module, nn.Linear, BCELoss and torch.optim. We compare the
resulting decision boundary and confirm that autograd gives the same answer.
"""

import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DNN2(nn.Module):
    def __init__(self, in_d=2, h=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, h), nn.Sigmoid(),
            nn.Linear(h, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(1)

    model = DNN2()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    crit = nn.BCELoss()

    losses, accs = [], []
    for ep in range(2000):
        model.train()
        opt.zero_grad()
        p = model(Xtr_t)
        loss = crit(p, ytr_t)
        loss.backward()
        opt.step()
        if ep % 200 == 0:
            acc = ((p > 0.5).float() == ytr_t).float().mean().item()
            print(f"epoch {ep:4d}  loss={loss.item():.4f}  train_acc={acc:.3f}")
            losses.append(loss.item()); accs.append(acc)

    model.eval()
    with torch.no_grad():
        test_acc = ((model(Xte_t) > 0.5).float() == yte_t).float().mean().item()
    print(f"test acc = {test_acc:.3f}")

    # decision boundary
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1, 1.5, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        out = (model(grid) > 0.5).reshape(xx.shape).numpy()

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, out, alpha=0.3, cmap="RdBu")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", s=12)
    plt.title("practical_24: 2-layer DNN (PyTorch)")
    plt.savefig("part_a/figures/practical_24_boundary.png")
    plt.show()

    plt.figure()
    plt.plot(losses, label="loss"); plt.plot(accs, label="acc")
    plt.xlabel("epoch / 200"); plt.legend()
    plt.title("practical_24: training curves")
    plt.savefig("part_a/figures/practical_24_curves.png")
    plt.show()


import numpy as np   # used for meshgrid in main()
if __name__ == "__main__":
    main()
