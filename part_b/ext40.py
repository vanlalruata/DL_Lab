"""ext40: Conditional GAN (cGAN) generating class-conditioned samples."""
import torch
import torch.nn as nn


class cGenerator(nn.Module):
    def __init__(self, z=100, n_classes=10, f=64):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z + n_classes, f * 4, 4, 1, 0), nn.BatchNorm2d(f * 4), nn.ReLU(),
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1), nn.BatchNorm2d(f * 2), nn.ReLU(),
            nn.ConvTranspose2d(f * 2, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z, y):
        y = self.label_emb(y).unsqueeze(-1).unsqueeze(-1)
        z = torch.cat([z.view(z.size(0), -1, 1, 1), y], 1)
        return self.net(z)


if __name__ == "__main__":
    G = cGenerator()
    z = torch.randn(8, 100)
    y = torch.randint(0, 10, (8,))
    print("cGAN generated shape (conditioned on labels):", G(z, y).shape)
