"""part_d / pd04 - Edge federated learning for intrusion detection (FedAvg).

Simulates K edge nodes, each holding a local slice of wireless/SDN-style flow
data. Trains local MLPs and averages their weights (FedAvg). Compares a
centrally-trained model vs the federated model on a held-out test set, reporting
accuracy, ROC, and inference latency.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")


def gen_node(n=600, attack_ratio=0.3, seed=0):
    rng = np.random.RandomState(seed)
    n_a = int(n * attack_ratio)
    n_n = n - n_a
    normal = rng.normal(0, 1, (n_n, 5))
    attack = np.column_stack([rng.normal(3, 1, n_a), rng.normal(-2, 1, n_a),
                              rng.normal(4, 1, n_a), rng.poisson(5, n_a),
                              rng.poisson(8, n_a)]).astype(float)
    X = np.vstack([normal, attack]); y = np.hstack([np.zeros(n_n), np.ones(n_a)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)


def local_train(model, X, y, epochs=5, lr=1e-2):
    m = MLP(); m.load_state_dict(model.state_dict())
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = nn.CrossEntropyLoss()(m(X), y); loss.backward(); opt.step()
    return m.state_dict()


def fedavg(global_model, nodes, rounds=3, local_epochs=5):
    for r in range(rounds):
        states = [local_train(global_model, X, y, local_epochs) for X, y in nodes]
        new = {}
        for k in global_model.state_dict():
            new[k] = torch.mean(torch.stack([s[k].float() for s in states]), 0)
        global_model.load_state_dict(new)
    return global_model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(X), 1)[:, 1].numpy()
        pred = proba > 0.5
    return (pred == y.numpy()).mean(), roc_auc_score(y.numpy(), proba), proba


def main():
    torch.manual_seed(0)
    nodes = [gen_node(seed=i) for i in range(5)]
    allX = torch.cat([n[0] for n in nodes]); ally = torch.cat([n[1] for n in nodes])
    # held-out global test set
    teX, teY = gen_node(800, 0.3, seed=99)

    # centralized baseline
    central = MLP()
    opt = torch.optim.Adam(central.parameters(), lr=1e-2)
    for _ in range(20):
        opt.zero_grad(); loss = nn.CrossEntropyLoss()(central(allX), ally); loss.backward(); opt.step()

    # federated
    fed = MLP()
    fed = fedavg(fed, nodes, rounds=4, local_epochs=3)

    c_acc, c_auc, c_p = evaluate(central, teX, teY)
    f_acc, f_auc, f_p = evaluate(fed, teX, teY)
    print(f"Centralized: acc={c_acc:.3f} AUC={c_auc:.3f}")
    print(f"Federated  : acc={f_acc:.3f} AUC={f_auc:.3f}")

    plt.figure()
    for name, p in [("central", c_p), ("federated", f_p)]:
        fpr, tpr, _ = roc_curve(teY.numpy(), p)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(teY.numpy(), p):.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("pd04 Federated vs Centralized ROC"); plt.legend()
    plt.savefig(f"{FIG}/pd04_roc.png"); plt.close()

    t0 = time.perf_counter(); evaluate(fed, teX, teY); dt = time.perf_counter() - t0
    print(f"Federated inference: {dt*1000:.3f} ms / {len(teY)} samples")
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
