"""ext02: same / valid / causal padding for 1D and 2D signals."""
import numpy as np


def pad_1d(x, kernel_size, mode="same"):
    k = kernel_size
    if mode == "valid":
        return x
    if mode == "same":
        p = k // 2
    elif mode == "causal":
        p = k - 1  # pad only on the left
    else:
        raise ValueError(mode)
    if mode == "causal":
        return np.pad(x, (p, 0), mode="constant")
    return np.pad(x, (p, p), mode="constant")


def pad_2d(x, kernel_size, mode="same"):
    k = kernel_size
    if mode == "valid":
        return x
    p = k // 2
    return np.pad(x, ((0, 0), (p, p), (p, p)), mode="constant")


if __name__ == "__main__":
    x = np.arange(5)
    for m in ["valid", "same", "causal"]:
        print(m, pad_1d(x, 3, m))
