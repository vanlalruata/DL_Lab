"""
Practical 9: Multi-Class Categorical Cross-Entropy (CCE) & Softmax Coupling
Objective: Implement CCE from scratch using NumPy for multi-class predictions.
Compare performance and memory consumption between One-Hot Encoded Targets
(Categorical Cross-Entropy) and Integer Targets (Sparse Categorical Cross-Entropy).
"""

import numpy as np
import time
import os
import psutil  # optional; falls back gracefully if missing


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def categorical_ce(probs, one_hot):
    eps = 1e-12
    return -np.mean(np.sum(one_hot * np.log(probs + eps), axis=1))


def sparse_ce(probs, int_labels):
    eps = 1e-12
    n = probs.shape[0]
    return -np.mean(np.log(probs[np.arange(n), int_labels] + eps))


def rss_usage_mb():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return float("nan")


def main():
    np.random.seed(0)
    n_classes = 1000
    n_samples = 20000
    logits = np.random.randn(n_samples, n_classes)
    int_labels = np.random.randint(0, n_classes, size=n_samples)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), int_labels] = 1

    probs = softmax(logits)

    # One-hot / Categorical CE
    t0 = time.perf_counter()
    mem0 = rss_usage_mb()
    loss_oh = categorical_ce(probs, one_hot)
    t_oh = time.perf_counter() - t0
    mem_oh = rss_usage_mb() - mem0

    # Integer / Sparse CE
    t0 = time.perf_counter()
    mem0 = rss_usage_mb()
    loss_sp = sparse_ce(probs, int_labels)
    t_sp = time.perf_counter() - t0
    mem_sp = rss_usage_mb() - mem0

    print(f"Categorical CE (one-hot): loss={loss_oh:.4f}  time={t_oh*1000:.3f} ms")
    print(f"Sparse CE (integer)     : loss={loss_sp:.4f}  time={t_sp*1000:.3f} ms")
    print(f"one-hot target array size: {one_hot.nbytes/1e6:.2f} MB "
          f"(sparse needs only integer labels)")


if __name__ == "__main__":
    main()
