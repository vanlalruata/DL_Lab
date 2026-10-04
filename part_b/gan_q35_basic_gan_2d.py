"""ext35: Basic GAN (generator + discriminator MLP) on a 2D Gaussian mixture."""
import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z=2, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z, h), nn.ReLU(),
                                nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                                nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


def sample_real(n=128):
    a = np.random.randn(n // 2, 2) * 0.3 + np.array([2, 2])
    b = np.random.randn(n - n // 2, 2) * 0.3 + np.array([-2, -2])
    return torch.tensor(np.vstack([a, b]).astype("float32"))


if __name__ == "__main__":
    G, D = Generator(), Discriminator()
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3)
    for step in range(200):
        real = sample_real()
        z = torch.randn(128, 2)
        fake = G(z).detach()
        d_loss = nn.functional.binary_cross_entropy_with_logits(
            D(real), torch.ones(128, 1)) + nn.functional.binary_cross_entropy_with_logits(
            D(fake), torch.zeros(128, 1))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        z = torch.randn(128, 2)
        g_loss = nn.functional.binary_cross_entropy_with_logits(D(G(z)), torch.ones(128, 1))
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
    print("final d_loss:", d_loss.item())
