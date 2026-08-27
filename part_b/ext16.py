"""ext16: Compare parameter counts and FLOPs of AlexNet, VGG-16, ResNet-50."""
import torch
import torch.nn as nn
from torchvision import models


def stats(model):
    params = sum(p.numel() for p in model.parameters())
    return params


if __name__ == "__main__":
    for name, ctor in [("alexnet", models.alexnet),
                       ("vgg16", models.vgg16),
                       ("resnet50", models.resnet50)]:
        try:
            m = ctor(num_classes=10)
            print(f"{name}: {stats(m):,} params")
        except Exception as e:
            print(name, "unavailable:", e)
