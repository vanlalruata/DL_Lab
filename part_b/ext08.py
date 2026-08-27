"""ext08: GoogLeNet / Inception module with parallel branches."""
import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, in_c, c1, c3r, c3, c5r, c5, cp):
        super().__init__()
        self.b1 = nn.Conv2d(in_c, c1, 1)
        self.b2 = nn.Sequential(nn.Conv2d(in_c, c3r, 1), nn.ReLU(),
                               nn.Conv2d(c3r, c3, 3, padding=1), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(in_c, c5r, 1), nn.ReLU(),
                               nn.Conv2d(c5r, c5, 5, padding=2), nn.ReLU())
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, padding=1),
                               nn.Conv2d(in_c, cp, 1), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


if __name__ == "__main__":
    m = InceptionModule(192, 64, 96, 128, 16, 32, 32)
    print(m(torch.randn(2, 192, 28, 28)).shape)
