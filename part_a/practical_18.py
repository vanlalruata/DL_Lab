"""
Practical 18: End-to-End Classification Pipeline using torch.nn
Objective: Build a 3-hidden-layer MLP with nn.Sequential, nn.Linear, nn.ReLU and
nn.CrossEntropyLoss. Train on MNIST using a PyTorch DataLoader.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        correct = total_n = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total_n += yb.size(0)
        print(f"Epoch {epoch+1}: train loss {total/len(train_dl):.4f} | "
              f"test acc {correct/total_n:.4f}")


if __name__ == "__main__":
    main()
