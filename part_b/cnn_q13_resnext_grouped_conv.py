"""ext13: ResNeXt grouped convolution vs standard ResNet."""
import torch
import torch.nn as nn


class ResNeXtBlock(nn.Module):
    def __init__(self, in_c, out_c, groups=32, stride=1):
        super().__init__()
        mid = out_c // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=groups), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, out_c, 1), nn.BatchNorm2d(out_c))
        self.shortcut = (nn.Conv2d(in_c, out_c, 1, stride) if in_c != out_c or stride != 1
                         else nn.Identity())
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


if __name__ == "__main__":
    m = ResNeXtBlock(64, 64, groups=32)
    print(m(torch.randn(2, 64, 32, 32)).shape)
