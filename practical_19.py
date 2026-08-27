"""
Practical 19: PyTorch Custom Optimizer & Loss Benchmarking
Objective: Train identical MLPs on Fashion-MNIST using 5 optimizers: SGD,
SGD(momentum=0.9), Adagrad, RMSprop, AdamW. Plot combined training loss
convergence and validation accuracy across all 5 optimizers.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_optimizer(name, params):
    if name == "SGD":
        return torch.optim.SGD(params, lr=1e-2)
    if name == "SGD_momentum":
        return torch.optim.SGD(params, lr=1e-2, momentum=0.9)
    if name == "Adagrad":
        return torch.optim.Adagrad(params, lr=1e-2)
    if name == "RMSprop":
        return torch.optim.RMSprop(params, lr=1e-3)
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=1e-3)


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.2860,), (0.3530,))])
    train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256)

    names = ["SGD", "SGD_momentum", "Adagrad", "RMSprop", "AdamW"]
    losses_hist = {n: [] for n in names}
    acc_hist = {n: [] for n in names}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name in names:
        model = MLP().to(device)
        opt = get_optimizer(name, model.parameters())
        crit = nn.CrossEntropyLoss()
        for epoch in range(5):
            model.train()
            run = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                run += loss.item()
            losses_hist[name].append(run / len(train_dl))
            model.eval()
            c = n = 0
            with torch.no_grad():
                for xb, yb in test_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb).argmax(1)
                    c += (pred == yb).sum().item(); n += yb.size(0)
            acc_hist[name].append(c / n)
        print(f"{name}: final acc {acc_hist[name][-1]:.4f}")

    plt.figure(figsize=(7, 4))
    for n in names:
        plt.plot(losses_hist[n], label=n)
    plt.title("Training loss convergence")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.show()

    plt.figure(figsize=(7, 4))
    for n in names:
        plt.plot(acc_hist[n], label=n)
    plt.title("Validation accuracy")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
