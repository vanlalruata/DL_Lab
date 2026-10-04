"""ext03: Derive output spatial size of a conv layer."""


def output_size(H, kernel=3, stride=1, padding=0, dilation=1):
    k = dilation * (kernel - 1) + 1
    return (H + 2 * padding - k) // stride + 1


if __name__ == "__main__":
    for s, p in [(1, 0), (1, 1), (2, 1)]:
        print(f"kernel=3 stride={s} padding={p} -> "
              f"{output_size(224, 3, s, p)}x{output_size(224, 3, s, p)}")
