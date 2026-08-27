"""ext18: U-Net encoder-decoder with skip connections for segmentation (synthetic)."""
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_c=1, out_c=1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 32, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(32, out_c, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.up(e2)
        d = torch.cat([d, e1], 1)  # skip connection
        return self.dec(d)


if __name__ == "__main__":
    print(UNet()(torch.randn(2, 1, 64, 64)).shape)
