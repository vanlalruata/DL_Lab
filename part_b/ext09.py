"""ext09: 1x1 convolutions: channel mixing and dimensionality reduction."""
import torch
import torch.nn as nn


def count_conv_params(k, in_c, out_c):
    return k * k * in_c * out_c + out_c


if __name__ == "__main__":
    x = torch.randn(2, 256, 28, 28)
    # 1x1 reduces channels before an expensive 3x3, cutting cost drastically.
    reduce = nn.Conv2d(256, 64, 1)
    costly = nn.Conv2d(64, 64, 3, padding=1)
    out = costly(reduce(x))
    print("reduced-path out:", out.shape)
    print("3x3 direct params:", count_conv_params(3, 256, 64),
          "| 1x1+3x3 params:", count_conv_params(1, 256, 64) + count_conv_params(3, 64, 64))
