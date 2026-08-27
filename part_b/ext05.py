"""ext05: AlexNet in PyTorch with synthetic images; note ReLU+Dropout rationale."""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2), nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, 1, 2), nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    x = torch.randn(2, 3, 227, 227)
    print(AlexNet()(x).shape)
    # ReLU avoids saturating sigmoid; Dropout reduces co-adaptation -> generalization.
