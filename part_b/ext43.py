"""ext43: Pix2Pix (cGAN, U-Net generator + patch discriminator) for paired translation."""
import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(2 * c, 32, 4, 2, 1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU())
        self.up = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                               nn.ConvTranspose2d(32, c, 4, 2, 1), nn.Tanh())

    def forward(self, x, y):
        return self.up(self.down(torch.cat([x, y], 1)))


class PatchDisc(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2 * c, 32, 4, 2, 1), nn.LeakyReLU(0.2),
                                nn.Conv2d(32, 1, 4, 1, 0))  # patch logits

    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1)).view(-1)


if __name__ == "__main__":
    G = UNetGenerator(); D = PatchDisc()
    a = torch.randn(2, 3, 32, 32)
    b = torch.randn(2, 3, 32, 32)
    fake_b = G(a, b)
    print("pix2pix fake:", fake_b.shape, "patch D:", D(a, fake_b).shape)
