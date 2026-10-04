"""ext14: DenseNet dense block with concatenation (feature reuse)."""
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_c, growth=32):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, 4 * growth, 1)
        self.bn2 = nn.BatchNorm2d(4 * growth)
        self.conv2 = nn.Conv2d(4 * growth, growth, 3, padding=1)

    def forward(self, x):
        out = torch.relu(self.conv1(self.bn1(x)))
        out = torch.relu(self.conv2(self.bn2(out)))
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_c, n_layers, growth=32):
        super().__init__()
        self.layers = nn.ModuleList()
        c = in_c
        for _ in range(n_layers):
            self.layers.append(DenseLayer(c, growth))
            c += growth

    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return x


if __name__ == "__main__":
    print(DenseBlock(64, 4, 32)(torch.randn(2, 64, 16, 16)).shape)  # 64+4*32=192
