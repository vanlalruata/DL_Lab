"""
Practical 6: Softmax and Stable Numerical Implementations
Objective: Implement standard Softmax vs Numerically Stable Softmax (subtracting
max(z) from logits). Pass extreme logits like z = [1000, 1001, 1002] through both
to demonstrate how vanilla Softmax produces NaN/inf overflow while stable Softmax
handles it cleanly.
"""

import numpy as np


def softmax_vanilla(z):
    exp_z = np.exp(z)               # overflow for large z
    return exp_z / np.sum(exp_z)


def softmax_stable(z):
    z = z - np.max(z)               # shift by max for numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def main():
    test_cases = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1000.0, 1001.0, 1002.0]),
        np.array([-1000.0, -999.0, -1002.0]),
    ]

    for z in test_cases:
        print(f"\nLogits z = {z}")
        try:
            sv = softmax_vanilla(z)
            print(f"  vanilla : {sv}  (sum={np.sum(sv):.4f}, any NaN={np.any(np.isnan(sv))})")
        except Exception as e:
            print(f"  vanilla : ERROR {e}")
        ss = softmax_stable(z)
        print(f"  stable  : {ss}  (sum={np.sum(ss):.4f}, any NaN={np.any(np.isnan(ss))})")


if __name__ == "__main__":
    main()
