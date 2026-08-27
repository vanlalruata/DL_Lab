"""ext10: Batch Normalization from scratch + integrate into a CNN loop."""
import numpy as np


class BatchNorm2d:
    def __init__(self, c, eps=1e-5, momentum=0.9):
        self.gamma = np.ones(c)
        self.beta = np.zeros(c)
        self.running_mean = np.zeros(c)
        self.running_var = np.zeros(c)
        self.eps, self.momentum = eps, momentum

    def forward(self, x, train=True):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        xr = x.transpose(1, 0, 2, 3).reshape(C, -1)
        if train:
            mean = xr.mean(1)
            var = xr.var(1)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean, var = self.running_mean, self.running_var
        xr = (xr - mean[:, None]) / np.sqrt(var[:, None] + self.eps)
        xr = self.gamma[:, None] * xr + self.beta[:, None]
        return xr.reshape(C, N, H, W).transpose(1, 0, 2, 3)


if __name__ == "__main__":
    bn = BatchNorm2d(3)
    x = np.random.randn(4, 3, 8, 8)
    print("BN out shape:", bn.forward(x).shape)
