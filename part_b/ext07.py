"""ext07: 7x7 vs 5x5 vs 3x3 convolutions: receptive field vs parameter cost."""


def receptive_field(n_layers, kernel):
    return 1 + n_layers * (kernel - 1)


def params_per_layer(kernel, in_c, out_c):
    return kernel * kernel * in_c * out_c + out_c


if __name__ == "__main__":
    print("Receptive field of 3 stacked 3x3 =", receptive_field(3, 3), "= single 7x7")
    print("Params (64->64): 7x7 =", params_per_layer(7, 64, 64),
          "| 3x3 x3 =", 3 * params_per_layer(3, 64, 64))
