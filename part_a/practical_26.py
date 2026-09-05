"""Practical 26 - DNN activation function comparison in PyTorch.

Same 3-hidden-layer MLP trained with Sigmoid, Tanh, ReLU, and LeakyReLU.
Plots validation accuracy and final loss for each activation.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ActDNN(nn.Module):
    def __init__(self, act, in_d=2, hidden=(16, 16, 16), out_d=1):
        super().__init__()
        layers = []
        prev = in_d
        for h in hidden:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(1)

    acts = {"Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh,
            "ReLU": nn.ReLU, "LeakyReLU": lambda: nn.LeakyReLU(0.1)}
    results = {}
    for name, act in acts.items():
        torch.manual_seed(0)
        model = ActDNN(act)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        crit = nn.BCEWithLogitsLoss()
        va_acc_curve = []
        for _ in range(40):
            model.train()
            opt.zero_grad()
            crit(model(Xtr_t), ytr_t).backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                va_acc_curve.append(((torch.sigmoid(model(Xte_t)) > 0.5).float() == yte_t).float().mean().item())
        results[name] = va_acc_curve
        print(f"{name:10s}  final val acc = {va_acc_curve[-1]:.3f}")

    plt.figure()
    for n, c in results.items():
        plt.plot(c, label=n)
    plt.xlabel("epoch"); plt.ylabel("val accuracy")
    plt.title("practical_26: activation function comparison")
    plt.legend()
    plt.savefig("part_a/figures/practical_26_activations.png")
    plt.show()


if __name__ == "__main__":
    main()
