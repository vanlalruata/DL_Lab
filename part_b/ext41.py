"""ext41: CycleGAN (two generators, two discriminators) for unpaired translation."""
import torch
import torch.nn as nn


class ResGen(nn.Module):
    def __init__(self, c=3, f=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(c, f, 4, 2, 1), nn.ReLU(),
                                nn.Conv2d(f, f * 2, 4, 2, 1), nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(f * 2, f, 4, 2, 1), nn.ReLU(),
                                nn.ConvTranspose2d(f, c, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        return self.dec(self.enc(x))


class Disc(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, 32, 4, 2, 1), nn.LeakyReLU(0.2),
                                nn.Conv2d(32, 1, 4, 1, 0))

    def forward(self, x):
        return self.net(x).view(-1)


if __name__ == "__main__":
    G_AB, G_BA = ResGen(), ResGen()
    D_A, D_B = Disc(), Disc()
    a = torch.randn(2, 3, 32, 32)
    b = torch.randn(2, 3, 32, 32)
    # cycle-consistency: a -> G_AB -> b_hat -> G_BA -> a_recon
    recon_a = G_BA(G_AB(a))
    recon_b = G_AB(G_BA(b))
    print("cycle recon A shape:", recon_a.shape, "B shape:", recon_b.shape)
