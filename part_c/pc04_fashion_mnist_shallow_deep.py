"""part_c / pc04 - Fashion-MNIST exercise: architecture comparison.

Trains two CNNs (Shallow vs Deep) on Fashion-MNIST, compares train/val accuracy
and loss, ROC (OvR macro), inference latency, throughput, and parameter counts
to illustrate time/space complexity trade-offs.
"""
import os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Shallow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(),
                                nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def run(model, tr_dl, va_dl, te_dl, epochs=5):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    tr_a, va_a, tr_l, va_l = [], [], [], []
    for _ in range(epochs):
        model.train(); ca = cl = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            cl += loss.item() * len(yb); ca += (model(xb).argmax(1) == yb).sum().item()
        tr_l.append(cl / len(tr_dl.dataset)); tr_a.append(ca / len(tr_dl.dataset))
        model.eval(); vc = vl = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb); vl += crit(out, yb).item() * len(yb)
                vc += (out.argmax(1) == yb).sum().item()
        va_l.append(vl / len(va_dl.dataset)); va_a.append(vc / len(va_dl.dataset))
    # test
    model.eval(); all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            all_p.append(torch.softmax(model(xb.to(DEVICE)), 1).cpu()); all_y.append(yb)
    proba = torch.cat(all_p).numpy(); y = torch.cat(all_y).numpy()
    acc = (proba.argmax(1) == y).mean()
    auc = roc_auc_score(y, proba, multi_class="ovr")
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in te_dl:
            model(xb.to(DEVICE))
    dt = time.perf_counter() - t0
    return dict(acc=acc, auc=auc, tr_a=tr_a, va_a=va_a, tr_l=tr_l, va_l=va_l,
               params=n_params, infer_ms=dt * 1000, n_test=len(va_dl.dataset) + 0)


def main():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    tr_ds = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
    te_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    tr, va = torch.utils.data.random_split(tr_ds, [50000, 10000])
    tr_dl = DataLoader(tr, 128, shuffle=True)
    va_dl = DataLoader(va, 256)
    te_dl = DataLoader(te_ds, 256)

    results = {n: run(m, tr_dl, va_dl, te_dl) for n, m in [("Shallow", Shallow()), ("Deep", Deep())]}
    for n, r in results.items():
        print(f"{n}: acc={r['acc']:.4f} auc={r['auc']:.4f} params={r['params']:,} "
              f"infer={r['infer_ms']:.0f}ms")

    plt.figure(figsize=(7, 4))
    for n, r in results.items():
        plt.plot(r["tr_a"], label=f"{n} train"); plt.plot(r["va_a"], label=f"{n} val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Fashion-MNIST accuracy")
    plt.savefig(f"{FIG}/pc04_acc.png"); plt.close()
    plt.figure(figsize=(7, 4))
    for n, r in results.items():
        plt.plot(r["tr_l"], label=f"{n} train"); plt.plot(r["va_l"], label=f"{n} val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Fashion-MNIST loss")
    plt.savefig(f"{FIG}/pc04_loss.png"); plt.close()

    names = list(results)
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1); plt.bar(names, [results[n]["params"] for n in names]); plt.title("params")
    plt.subplot(1, 2, 2); plt.bar(names, [results[n]["infer_ms"] for n in names]); plt.title("infer ms")
    plt.tight_layout(); plt.savefig(f"{FIG}/pc04_complexity.png"); plt.close()
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
