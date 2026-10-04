"""
Practical 20: Self-Normalizing Networks with SELU and AlphaDropout
Objective: Construct a 10-layer deep feedforward network in PyTorch using nn.SELU
and nn.AlphaDropout. Record mean and variance of hidden activations across all 10
layers during forward passes to verify the self-normalizing property (without BN).
"""

import torch
import torch.nn as nn
import numpy as np


class SELUNet(nn.Module):
    def __init__(self, in_dim=20, hidden=64, n_layers=10):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SELU(), nn.AlphaDropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SELU(), nn.AlphaDropout(0.1)]
        self.net = nn.Sequential(*layers)
        self.n_layers = n_layers

    def forward(self, x):
        stats = []
        h = x
        # collect per-layer stats by walking the sequential blocks
        for i in range(0, len(self.net), 3):
            h = self.net[i](h)            # Linear
            if self.training:
                h = self.net[i + 2](self.net[i + 1](h))  # SELU then AlphaDropout
                a = self.net[i + 1](h)
            else:
                h = self.net[i + 1](h)    # SELU only
                a = h
            stats.append((a.mean().item(), a.var().item()))
        return h, stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SELUNet().to(device).eval()
    rng = torch.randn(1024, 20).to(device)

    means, vars_ = [], []
    with torch.no_grad():
        _, stats = model(rng)
    for m, v in stats:
        means.append(m)
        vars_.append(v)

    print("Layer : mean    variance")
    for i, (m, v) in enumerate(zip(means, vars_)):
        print(f"  {i+1:2d}  : {m:+.4f}  {v:.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(means) + 1), means, "o-", label="mean")
    plt.axhline(0.0, color="gray", ls="--")
    plt.plot(range(1, len(vars_) + 1), vars_, "s-", label="variance")
    plt.axhline(1.0, color="gray", ls="--")
    plt.xlabel("hidden layer")
    plt.title("SELU self-normalizing: activations stay near mean=0, var=1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
