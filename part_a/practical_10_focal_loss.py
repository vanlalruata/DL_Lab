"""
Practical 10: Advanced Loss Functions in PyTorch (Focal Loss)
Objective: Implement Focal Loss as a custom torch.nn.Module. Train a small
PyTorch classifier on a highly imbalanced binary dataset (95% negative, 5%
positive) and compare accuracy/recall against standard nn.BCEWithLogitsLoss.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (N,) raw scores; targets: (N,) in {0,1}
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class SmallMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_imbalanced(n=4000, n_pos=200, seed=0):
    X, y = make_classification(n_samples=n, n_features=20, n_informative=10,
                               n_redundant=5, random_state=seed)
    # force 95/5 imbalance
    pos_idx = np.where(y == 1)[0][:n_pos]
    neg_idx = np.where(y == 0)[0][:n - n_pos]
    keep = np.concatenate([pos_idx, neg_idx])
    return X[keep], y[keep]


def train_and_eval(criterion, X_tr, y_tr, X_te, y_te, epochs=20, lr=1e-3):
    model = SmallMLP(X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xtr)
        loss = criterion(logits, ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(model(torch.tensor(X_te, dtype=torch.float32))) > 0.5).int().numpy()
    acc = accuracy_score(y_te, pred)
    rec = recall_score(y_te, pred, zero_division=0)
    return acc, rec


def main():
    X, y = make_imbalanced()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)
    print(f"Train class balance: {np.mean(y_tr):.3f} positive")

    acc_b, rec_b = train_and_eval(nn.BCEWithLogitsLoss(), X_tr, y_tr, X_te, y_te)
    acc_f, rec_f = train_and_eval(FocalLoss(alpha=0.75, gamma=2.0), X_tr, y_tr, X_te, y_te)

    print(f"BCEWithLogitsLoss : accuracy={acc_b:.3f}  recall={rec_b:.3f}")
    print(f"FocalLoss        : accuracy={acc_f:.3f}  recall={rec_f:.3f}")
    print("Focal loss typically improves recall on the rare positive class.")


if __name__ == "__main__":
    main()
