"""Practical 25 - Deeper DNN in PyTorch (4 hidden layers).

A 4-hidden-layer MLP on a more difficult non-linearly-separable dataset
(make_circles). Demonstrates how depth increases representational capacity and
how train/val curves should be monitored to detect over/underfitting.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DeepDNN(nn.Module):
    def __init__(self, in_d=2, hidden=(32, 16, 8, 4), out_d=1):
        super().__init__()
        layers = []
        prev = in_d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    X, y = make_circles(n_samples=600, noise=0.2, factor=0.4, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(1)

    model = DeepDNN(hidden=(32, 16, 8, 4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.BCEWithLogitsLoss()

    tr_loss, va_loss, tr_acc, va_acc = [], [], [], []
    for ep in range(60):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = crit(logits, ytr_t)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(Xte_t), yte_t).item()
            ta = ((torch.sigmoid(model(Xtr_t)) > 0.5).float() == ytr_t).float().mean().item()
            va = ((torch.sigmoid(model(Xte_t)) > 0.5).float() == yte_t).float().mean().item()
        tr_loss.append(loss.item()); va_loss.append(vl)
        tr_acc.append(ta); va_acc.append(va)

    print(f"final train acc {tr_acc[-1]:.3f}  val acc {va_acc[-1]:.3f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(tr_loss, label="train"); plt.plot(va_loss, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("loss")
    plt.subplot(1, 2, 2)
    plt.plot(tr_acc, label="train"); plt.plot(va_acc, label="val")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend(); plt.title("accuracy")
    plt.tight_layout(); plt.savefig("part_a/figures/practical_25_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
