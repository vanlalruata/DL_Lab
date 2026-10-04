"""Practical 30 - DNN with early stopping and learning-rate scheduling.

A 3-hidden-layer MLP trained with:
  * ReduceLROnPlateau scheduler that halves the learning rate when the
    validation loss plateaus, and
  * early stopping that saves the best model state and stops training when
    validation loss stops improving for `patience` epochs.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DNN(nn.Module):
    def __init__(self, in_d=2, hidden=(32, 32, 32), out_d=1):
        super().__init__()
        layers = []; prev = in_d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
        layers.append(nn.Linear(prev, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    X, y = make_moons(n_samples=800, noise=0.2, random_state=0)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=1)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32).unsqueeze(1)

    model = DNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.BCEWithLogitsLoss()

    best_va = float("inf")
    best_state = None
    best_ep = -1
    patience, since = 3, 0
    tr_curve, va_curve, lr_curve = [], [], []

    for ep in range(80):
        model.train()
        opt.zero_grad()
        tl = crit(model(Xtr_t), ytr_t); tl.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva_t), yva_t).item()
        sched.step(vl)
        lr_curve.append(opt.param_groups[0]["lr"])
        tr_curve.append(tl.item()); va_curve.append(vl)

        if vl < best_va - 1e-4:
            best_va = vl; best_ep = ep; since = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= patience:
                print(f"early stop at epoch {ep} (best val loss at epoch {best_ep})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(Xva_t)) > 0.5).float() == yva_t).float().mean().item()
    print(f"best val acc = {acc:.3f} (best epoch {best_ep})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(tr_curve, label="train"); axes[0].plot(va_curve, label="val")
    axes[0].axvline(best_ep, color="k", ls="--"); axes[0].set_title("loss"); axes[0].legend()
    axes[1].plot(lr_curve); axes[1].set_title("learning rate")
    plt.tight_layout(); plt.savefig("part_a/figures/practical_30_earlystop.png")
    plt.show()


if __name__ == "__main__":
    main()
