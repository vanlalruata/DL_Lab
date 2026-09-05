"""Practical 27 - DNN with Dropout regularization in PyTorch.

Trains the same MLP on make_moons with and without Dropout, and shows that
dropout reduces the gap between training and validation loss (regularization).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DNNDropout(nn.Module):
    def __init__(self, p=0.0, in_d=2, hidden=(64, 64), out_d=1):
        super().__init__()
        self.p = p
        self.fc1 = nn.Linear(in_d, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], out_d)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = nn.functional.dropout(h, p=self.p, training=self.training)
        h = self.act(self.fc2(h))
        h = nn.functional.dropout(h, p=self.p, training=self.training)
        return self.fc3(h)


def run(p, Xtr_t, ytr_t, Xte_t, yte_t, epochs=80):
    torch.manual_seed(0)
    model = DNNDropout(p=p)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.BCEWithLogitsLoss()
    tr_loss, va_loss = [], []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        l = crit(model(Xtr_t), ytr_t); l.backward(); opt.step()
        tr_loss.append(l.item())
        model.eval()
        with torch.no_grad():
            va_loss.append(crit(model(Xte_t), yte_t).item())
    return tr_loss, va_loss


def main():
    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(1)

    tr0, va0 = run(0.0, Xtr_t, ytr_t, Xte_t, yte_t)
    tr5, va5 = run(0.5, Xtr_t, ytr_t, Xte_t, yte_t)

    plt.figure(figsize=(8, 4))
    plt.plot(tr0, label="train (no dropout)"); plt.plot(va0, label="val (no dropout)")
    plt.plot(tr5, "--", label="train (p=0.5)"); plt.plot(va5, "--", label="val (p=0.5)")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("practical_27: dropout regularisation")
    plt.savefig("part_a/figures/practical_27_dropout.png")
    plt.show()
    print(f"no-dropout final gap = {tr0[-1]-va0[-1]:.4f}")
    print(f"p=0.5   final gap    = {tr5[-1]-va5[-1]:.4f}")


if __name__ == "__main__":
    main()
