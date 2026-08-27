"""
Practical 7: Regression Loss Functions (MSE, MAE, and Huber Loss)
Objective: Implement MSE, MAE, and Huber Loss (delta=1.0) in NumPy. Create a
synthetic regression dataset with high-magnitude outliers. Plot the loss curves
and compare how each loss penalizes outliers.
"""

import numpy as np
import matplotlib.pyplot as plt


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def huber(y_true, y_pred, delta=1.0):
    diff = np.abs(y_true - y_pred)
    quad = 0.5 * diff ** 2
    lin = delta * (diff - 0.5 * delta)
    return np.mean(np.where(diff <= delta, quad, lin))


def main():
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y_true = 2 * x + 1 + np.random.randn(n) * 0.5
    # inject high-magnitude outliers
    outlier_idx = [20, 50, 80]
    y_true[outlier_idx] += [40, -50, 60]

    # candidate predictions: shift a perfect-fit line to scan loss vs offset
    offsets = np.linspace(-5, 5, 200)
    losses_mse, losses_mae, losses_huber = [], [], []
    for o in offsets:
        pred = 2 * x + 1 + o
        losses_mse.append(mse(y_true, pred))
        losses_mae.append(mae(y_true, pred))
        losses_huber.append(huber(y_true, pred))

    plt.figure(figsize=(8, 5))
    plt.plot(offsets, losses_mse, label="MSE", lw=2)
    plt.plot(offsets, losses_mae, label="MAE", lw=2)
    plt.plot(offsets, losses_huber, label="Huber (delta=1)", lw=2)
    plt.xlabel("prediction offset")
    plt.ylabel("loss")
    plt.title("Regression Loss Comparison (with outliers)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    best = np.argmin(losses_mae)
    print("MSE most sensitive to outliers (high, steep); MAE/Huber more robust.")


if __name__ == "__main__":
    main()
