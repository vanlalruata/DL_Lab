"""ext04: max-pooling and average-pooling from scratch in NumPy."""


def pool2d(x, size=2, stride=2, mode="max"):
    C, H, W = x.shape
    H2 = (H - size) // stride + 1
    W2 = (W - size) // stride + 1
    out = np.zeros((C, H2, W2))
    fn = np.max if mode == "max" else np.mean
    for c in range(C):
        for i in range(0, H2 * stride, stride):
            for j in range(0, W2 * stride, stride):
                out[c, i // stride, j // stride] = fn(x[c, i:i + size, j:j + size])
    return out


if __name__ == "__main__":
    x = np.random.rand(2, 4, 4)
    print("max pool:", pool2d(x, 2, 2, "max").shape)
    print("avg pool:", pool2d(x, 2, 2, "avg").shape)
