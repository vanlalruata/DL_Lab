"""ext38: WGAN-GP (gradient penalty) and stability vs vanilla GAN."""
import torch
import torch.nn as nn


def gradient_penalty(critic, real, fake):
    a = torch.rand(real.size(0), 1)
    interp = a * real + (1 - a) * fake
    interp.requires_grad_(True)
    d = critic(interp)
    grad = torch.autograd.grad(d.sum(), interp, create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    C = Critic()
    real = torch.randn(64, 2)
    fake = torch.randn(64, 2)
    gp = gradient_penalty(C, real, fake)
    print("gradient penalty (target ~0):", gp.item())
