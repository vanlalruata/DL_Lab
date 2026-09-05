"""Practical 29 - DNN weight initialization (random vs Xavier vs He) in PyTorch.

Compares three initialization strategies on the same 4-hidden-layer ReLU MLP:
random normal, Xavier (Glorot), and Kaiming (He). He init is theoretically the
best match for ReLU and typically yields the lowest training loss.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class InitDNN(nn.Module):
    def __init__(self, init="xavier", in_d=2, hidden=(32, 32, 32, 32), out_d=1):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = in_d
        for h in hidden:
            lin = nn.Linear(prev, h)
            if init == "xavier":
                nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
            elif init == "he":
                nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu"); nn.init.zeros_(lin.bias)
            else:
                nn.init.normal_(lin.weight, std=0.05); nn.init.zeros_(lin.bias)
            self.layers.append(lin)
            prev = h
        self.head = nn.Linear(prev, out_d)
        nn.init.zeros_(self.head.bias)
        self.act = nn.ReLU()

    def forward(self, x):
        h = x
        for l in self.layers:
            h = self.act(l(h))
        return self.head(h)


def run(init):
    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(1)

    model = InitDNN(init=init)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(60):
        model.train()
        opt.zero_grad()
        l = crit(model(Xtr_t), ytr_t); l.backward(); opt.step()
        losses.append(l.item())
    model.eval()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(Xte_t)) > 0.5).float() == yte_t).float().mean().item()
    return losses, acc


def main():
    res = {n: run(n) for n in ("random", "xavier", "he")}
    plt.figure()
    for n, (l, a) in res.items():
        plt.plot(l, label=f"{n} (acc={a:.2f})")
    plt.xlabel("epoch"); plt.ylabel("train loss"); plt.legend()
    plt.title("practical_29: weight initialisation comparison")
    plt.savefig("part_a/figures/practical_29_init.png")
    plt.show()


if __name__ == "__main__":
    main()
