"""ext19: Transpose convolution (fractional strided) for upsampling."""
import torch
import torch.nn as nn


def output_size_convT(H, kernel, stride=2, padding=0):
    return (H - 1) * stride - 2 * padding + kernel


if __name__ == "__main__":
    up = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
    x = torch.randn(2, 16, 8, 8)
    y = up(x)
    print("upsampled:", y.shape)
    print("formula 8 ->", output_size_convT(8, 4, 2, 1))
