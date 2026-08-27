"""ext15: Depthwise-separable convolution (MobileNet) and FLOP savings."""
import torch
import torch.nn as nn


def dw_sep_block(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c),  # depthwise
        nn.BatchNorm2d(in_c), nn.ReLU(),
        nn.Conv2d(in_c, out_c, 1),  # pointwise
        nn.BatchNorm2d(out_c), nn.ReLU())


def standard_params(k, in_c, out_c):
    return k * k * in_c * out_c


def dw_sep_params(k, in_c, out_c):
    return k * k * in_c + in_c * out_c


if __name__ == "__main__":
    print("standard 3x3 64->64 params:", standard_params(3, 64, 64))
    print("dw-sep params:", dw_sep_params(3, 64, 64))
    print("out:", dw_sep_block(64, 64)(torch.randn(2, 64, 32, 32)).shape)
