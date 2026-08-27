"""ext39: DCGAN for generating MNIST/Fashion-MNIST images (conv generator/discriminator)."""
import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    def __init__(self, z=100, f=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z, f * 4, 4, 1, 0), nn.BatchNorm2d(f * 4), nn.ReLU(),
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1), nn.BatchNorm2d(f * 2), nn.ReLU(),
            nn.ConvTranspose2d(f * 2, f, 4, 2, 1), nn.BatchNorm2d(f), nn.ReLU(),
            nn.ConvTranspose2d(f, 1, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        return self.net(x.view(x.size(0), -1, 1, 1))


class DCGANDiscriminator(nn.Module):
    def __init__(self, f=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, f, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(f, f * 2, 4, 2, 1), nn.BatchNorm2d(f * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(f * 2, f * 4, 4, 2, 1), nn.BatchNorm2d(f * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(f * 4, 1, 4, 1, 0))

    def forward(self, x):
        return self.net(x).view(-1)


if __name__ == "__main__":
    z = torch.randn(8, 100)
    fake = DCGANGenerator()(z)
    print("generated image shape:", fake.shape, "-> D score:",
          DCGANDiscriminator()(fake).shape)
