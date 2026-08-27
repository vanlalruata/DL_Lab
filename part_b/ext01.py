"""ext01: 2D convolution from scratch in NumPy (no nn.Conv2d)."""
import numpy as np


def conv2d(img, kernel, stride=1, padding=0):
    """img: (C_in, H, W), kernel: (C_out, C_in, k, k). Returns (C_out, H', W')."""
    C_in, H, W = img.shape
    C_out, _, k, _ = kernel.shape
    if padding:
        img = np.pad(img, ((0, 0), (padding, padding), (padding, padding)),
                     mode="constant")
    H2 = (img.shape[1] - k) // stride + 1
    W2 = (img.shape[2] - k) // stride + 1
    out = np.zeros((C_out, H2, W2))
    for co in range(C_out):
        for i in range(0, H2 * stride, stride):
            for j in range(0, W2 * stride, stride):
                for ci in range(C_in):
                    patch = img[ci, i:i + k, j:j + k]
                    out[co, i // stride, j // stride] += np.sum(patch * kernel[co, ci])
    return out


if __name__ == "__main__":
    x = np.random.rand(3, 8, 8)
    k = np.random.rand(2, 3, 3, 3)
    print("conv output shape:", conv2d(x, k, stride=1, padding=1).shape)
