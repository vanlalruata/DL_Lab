"""part_e / pe03 - Multi-head self-attention from scratch (NumPy) + visualization.

Builds scaled dot-product multi-head attention with nothing but NumPy and plots
the attention heatmaps for a short sentence, illustrating what each head focuses on.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")


def random_qkv(seq_len, d, h=4, seed=0):
    rng = np.random.RandomState(seed)
    s = np.random.randn(seq_len, d)
    Wq = rng.randn(d, d); Wk = rng.randn(d, d); Wv = rng.randn(d, d)
    Wo = rng.randn(d, d)
    Wv = rng.randn(d, d)
    Q, K, V = s @ Wq, s @ Wk, s @ Wv  # V uses a learned projection of the input
    return Q, K, V, Wo, h


def attention_head(Q, K, V, dk):
    scores = (Q @ K.T) / np.sqrt(dk)
    weights = np.exp(scores - scores.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    return weights @ V, weights


def main():
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    Q, K, V, Wo, h = random_qkv(len(tokens), 16, h=4)
    dk = 16 // h
    fig, axes = plt.subplots(1, h, figsize=(14, 3))
    for i in range(h):
        Qh = Q[:, i * dk:(i + 1) * dk]; Kh = K[:, i * dk:(i + 1) * dk]; Vh = V[:, i * dk:(i + 1) * dk]
        out, w = attention_head(Qh, Kh, Vh, dk)
        ax = axes[i]; ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens))); ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90); ax.set_yticklabels(tokens)
        ax.set_title(f"head {i}")
    plt.tight_layout(); plt.savefig(f"{FIG}/pe03_attention_heads.png"); plt.close()
    print("Saved multi-head attention heatmaps for:", tokens)
    print("Figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
