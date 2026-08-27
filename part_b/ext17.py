"""ext17: Global average pooling as a regularizer (GoogLeNet-style)."""
import torch
import torch.nn as nn


class GAPNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 10, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        # global average pooling over spatial dims -> (N, C, 1, 1)
        return x.mean(dim=(2, 3))  # no fc weights -> fewer params, acts as regularizer


if __name__ == "__main__":
    m = GAPNet()
    # only 1x1 conv params + bias, no large FC classifier
    print("params:", sum(p.numel() for p in m.parameters()))
    print("out:", m(torch.randn(2, 3, 32, 32)).shape)
