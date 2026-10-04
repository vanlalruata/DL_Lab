"""ext12: Why residual connections mitigate vanishing gradients (gradient flow demo)."""
import torch
import torch.nn as nn


def grad_flow(use_skip, depth=20):
    torch.manual_seed(0)
    x = torch.randn(2, 4, requires_grad=True)
    h = x
    layers = [nn.Linear(4, 4) for _ in range(depth)]
    for L in layers:
        y = L(h)
        h = y + h if use_skip else y  # skip adds identity path
        h = torch.relu(h)
    loss = h.sum()
    loss.backward()
    return x.grad.abs().mean().item()


if __name__ == "__main__":
    print("mean grad (no skip):", grad_flow(False))
    print("mean grad (skip)   :", grad_flow(True))
    print("Skip connections add an identity path so gradients do not vanish to ~0.")
