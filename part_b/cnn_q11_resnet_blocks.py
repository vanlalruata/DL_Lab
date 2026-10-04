"""ext11: ResNet basic-block and bottleneck-block with skip connections."""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = (nn.Conv2d(in_c, out_c, 1, stride) if in_c != out_c or stride != 1
                         else nn.Identity())
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        mid = out_c // 4
        self.b1 = nn.Sequential(nn.Conv2d(in_c, mid, 1), nn.BatchNorm2d(mid), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(mid, mid, 3, stride, 1), nn.BatchNorm2d(mid), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(mid, out_c, 1), nn.BatchNorm2d(out_c))
        self.shortcut = (nn.Conv2d(in_c, out_c, 1, stride) if in_c != out_c or stride != 1
                         else nn.Identity())
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.b3(self.b2(self.b1(x)))
        return self.relu(out + self.shortcut(x))


if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    print("basic:", BasicBlock(64, 64)(x).shape)
    print("bottleneck:", Bottleneck(64, 256)(x).shape)
