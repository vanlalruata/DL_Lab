"""ext20: Visualize CNN filters / feature maps of a pretrained model using hooks."""
import torch
import torch.nn as nn
import torchvision.models as models


def capture_feature_maps(layer, x):
    acts = {}
    hook = lambda m, i, o: acts.setdefault("out", o.detach())
    h = layer.register_forward_hook(hook)
    _ = layer(x)
    h.remove()
    return acts["out"]


if __name__ == "__main__":
    model = models.vgg16(pretrained=False)
    conv1 = model.features[0]  # first conv
    x = torch.randn(1, 3, 224, 224)
    fm = capture_feature_maps(conv1, x)
    print("conv1 feature maps shape:", fm.shape)  # (1, 64, H, W)
    print("mean activation per filter (first 5):", fm[0, :5].mean(dim=(1, 2)).numpy())
