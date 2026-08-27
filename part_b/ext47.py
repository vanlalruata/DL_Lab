"""ext47: Graph Attention Networks (GAT) with learned attention coefficients."""
import numpy as np


def softmax_row(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def gat_layer(A, X, W, a):
    """A: (N,N) adjacency, X: (N,F), W: (F,F'), a: (2F',). LeakyReLU attention."""
    N = X.shape[0]
    H = X @ W
    attn = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if A[i, j] == 0 and i != j:
                continue
            e = np.dot(a, np.concatenate([H[i], H[j]]))
            attn[i, j] = np.maximum(e, 0.2 * e)  # LeakyReLU slope 0.2
    for i in range(N):
        attn[i] = softmax_row(attn[i])
    return attn @ H


if __name__ == "__main__":
    N = 5
    A = (np.random.rand(N, N) < 0.5).astype(float)
    np.fill_diagonal(A, 1)
    X = np.random.randn(N, 3)
    W = np.random.randn(3, 4)
    a = np.random.randn(8)
    out = gat_layer(A, X, W, a)
    print("GAT output shape:", out.shape)
