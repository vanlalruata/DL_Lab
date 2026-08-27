"""part_c / pc03 - MNIST exercise (CNN in PyTorch).

Full pipeline: EDA, CNN train/val/test, accuracy & loss curves per epoch,
multi-class ROC (one-vs-rest macro), inference latency, throughput, and
parameter count (time/space complexity proxy). Saves figures to ./figures.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, confusion_matrix

FIG = os.path.join(os.path.dirname(__file__), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)


def main():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
    tr, va = torch.utils.data.random_split(train_ds, [50000, 10000])
    tr_dl, va_dl, te_dl = (DataLoader(s, 128, shuffle=(s is tr)) for s in (tr, va, test_ds))

    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    tr_acc, va_acc, tr_loss, va_loss = [], [], [], []

    for epoch in range(5):
        model.train(); ca = cl = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            cl += loss.item() * len(yb); ca += (model(xb).argmax(1) == yb).sum().item()
        tr_loss.append(cl / len(tr_ds)); tr_acc.append(ca / len(tr_ds))
        model.eval(); va_c = va_l = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb); va_l += crit(out, yb).item() * len(yb)
                va_c += (out.argmax(1) == yb).sum().item()
        va_loss.append(va_l / len(va)); va_acc.append(va_c / len(va))
        print(f"ep{epoch+1}: tr_acc={tr_acc[-1]:.3f} va_acc={va_acc[-1]:.3f}")

    # test evaluation
    model.eval(); all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(DEVICE)
            all_p.append(torch.softmax(model(xb), 1).cpu()); all_y.append(yb)
    proba = torch.cat(all_p).numpy(); y_true = torch.cat(all_y).numpy()
    pred = proba.argmax(1)
    acc = (pred == y_true).mean()
    auc = roc_auc_score(y_true, proba, multi_class="ovr")
    print(f"Test acc={acc:.4f}  macro ROC-AUC(OvR)={auc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_true, pred))

    # plots
    plt.figure(figsize=(6, 4))
    plt.plot(tr_acc, label="train"); plt.plot(va_acc, label="val"); plt.ylabel("acc")
    plt.xlabel("epoch"); plt.legend(); plt.title("MNIST accuracy"); plt.savefig(f"{FIG}/pc03_acc.png"); plt.close()
    plt.figure(figsize=(6, 4))
    plt.plot(tr_loss, label="train"); plt.plot(va_loss, label="val"); plt.ylabel("loss")
    plt.xlabel("epoch"); plt.legend(); plt.title("MNIST loss"); plt.savefig(f"{FIG}/pc03_loss.png"); plt.close()
    plt.figure(figsize=(6, 5))
    for i in range(10):
        fpr, tpr, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve((y_true == i).astype(int), proba[:, i])
        plt.plot(fpr, tpr, label=f"digit {i}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("MNIST ROC (OvR)"); plt.legend(fontsize=7); plt.savefig(f"{FIG}/pc03_roc.png"); plt.close()

    # inference time / complexity
    n_params = sum(p.numel() for p in model.parameters())
    xb, _ = next(iter(te_dl)); xb = xb.to(DEVICE)
    for _ in range(5): model(xb)  # warmup
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in te_dl:
            model(xb.to(DEVICE))
    dt = time.perf_counter() - t0
    print(f"Inference: {dt*1000:.1f} ms / {len(test_ds)} imgs "
          f"({dt/len(test_ds)*1e6:.1f} us/img); params={n_params:,}")


if __name__ == "__main__":
    main()
