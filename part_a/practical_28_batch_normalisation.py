"""Practical 28 - DNN with Batch Normalization in PyTorch.

Same MLP trained with and without BatchNorm1d after each hidden layer.
BatchNorm typically yields faster convergence and lets us use a higher learning
rate. The training-loss curves are compared side by side.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DNNBN(nn.Module):
    def __init__(self, use_bn=True, in_d=2, hidden=(32, 32), out_d=1):
        super().__init__()
        self.use_bn = use_bn
        self.fc1 = nn.Linear(in_d, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], out_d)
        self.bn1 = nn.BatchNorm1d(hidden[0]) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(hidden[1]) if use_bn else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.bn1(self.fc1(x)))
        h = self.act(self.bn2(self.fc2(h)))
        return self.fc3(h)


def run(use_bn, Xtr_t, ytr_t, Xte_t, yte_t, lr=0.05, epochs=40):
    torch.manual_seed(0)
    model = DNNBN(use_bn=use_bn)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        l = crit(model(Xtr_t), ytr_t); l.backward(); opt.step()
        losses.append(l.item())
    model.eval()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(Xte_t)) > 0.5).float() == yte_t).float().mean().item()
    return losses, acc


def main():
    torch.manual_seed(0)
    X, y = make_moons(n_samples=800, noise=0.2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(1)

    l_no_bn, a_no_bn = run(False, Xtr_t, ytr_t, Xte_t, yte_t)
    l_bn,   a_bn   = run(True,  Xtr_t, ytr_t, Xte_t, yte_t)

    plt.figure()
    plt.plot(l_no_bn, label="no BN")
    plt.plot(l_bn, label="with BatchNorm")
    plt.xlabel("epoch"); plt.ylabel("train loss"); plt.legend()
    plt.title("practical_28: BatchNorm speeds up convergence")
    plt.savefig("part_a/figures/practical_28_bn.png")
    plt.show()
    print(f"no BN: final loss={l_no_bn[-1]:.4f}  test acc={a_no_bn:.3f}")
    print(f"with BN: final loss={l_bn[-1]:.4f}  test acc={a_bn:.3f}")


if __name__ == "__main__":
    main()
