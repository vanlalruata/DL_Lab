"""ext06: VGG-16 block design and parameter-count growth analysis."""
import torch
import torch.nn as nn


def vgg_block(in_c, out_c, n_convs):
    layers = []
    for _ in range(n_convs):
        layers += [nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU()]
        in_c = out_c
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            vgg_block(3, 64, 2), vgg_block(64, 128, 2), vgg_block(128, 256, 3),
            vgg_block(256, 512, 3), vgg_block(512, 512, 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = VGG16()
    print("VGG16 params:", count_params(m))
    print("out:", m(torch.randn(2, 3, 224, 224)).shape)
