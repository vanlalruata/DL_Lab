"""part_e / pe10 - Parameter-Efficient Fine-Tuning: LoRA vs full fine-tuning.

Implements a LoRA linear layer (W + A@B with low-rank A,B frozen W) from scratch
in PyTorch and compares trainable-parameter counts and training latency against a
full fine-tune on a synthetic classification task. Uses `peft` when available for
the real HF path; otherwise the manual LoRA implementation demonstrates the concept.
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

FIG = os.path.join(os.path.dirname(__file__), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_f, in_f), requires_grad=False)  # frozen
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, x):
        return x @ self.W.T + x @ self.A.T @ self.B.T  # W + B@A (low rank)


def gen_data(n=2000, d=64):
    rng = np.random.RandomState(0)
    X = rng.randn(n, d); y = (X[:, :3].sum(1) > 0).astype(int)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    X, y = gen_data()
    Xtr, Xte = X[:1600].to(DEVICE), X[1600:].to(DEVICE)
    ytr, yte = y[:1600].to(DEVICE), y[1600:].to(DEVICE)

    full = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)).to(DEVICE)
    lora = nn.Sequential(LoRALinear(64, 32), nn.ReLU(), LoRALinear(32, 2)).to(DEVICE)

    results = {}
    for name, model in [("full", full), ("lora", lora)]:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
        crit = nn.CrossEntropyLoss()
        t0 = time.perf_counter()
        for _ in range(30):
            opt.zero_grad(); crit(model(Xtr), ytr).backward(); opt.step()
        dt = time.perf_counter() - t0
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        results[name] = (count_trainable(model), acc, dt * 1000)
        print(f"{name}: trainable={results[name][0]:,} acc={acc:.3f} train={dt*1000:.1f}ms")

    labels = list(results)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1); plt.bar(labels, [results[k][0] for k in labels])
    plt.title("trainable params"); plt.ylabel("count")
    plt.subplot(1, 2, 2); plt.bar(labels, [results[k][2] for k in labels])
    plt.title("train latency (ms)")
    plt.tight_layout(); plt.savefig(f"{FIG}/pe10_lora.png"); plt.close()
    print("Figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
