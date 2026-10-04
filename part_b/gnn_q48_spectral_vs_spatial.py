"""ext48: Compare spectral (GCN) vs spatial (GraphSAGE/GAT) convolution approaches.

Theoretical comparison implementation: build both message-passing operators on a
synthetic graph and report computational structure (spectral uses full-graph
Laplacian; spatial samples/local aggregates).
"""
import numpy as np


def spectral_gcn(A, X, W):
    D = np.diag(A.sum(1) ** -0.5)
    return D @ A @ D @ X @ W


def spatial_sage(A, X, W):
    N = X.shape[0]
    out = np.zeros((N, W.shape[1]))
    for i in range(N):
        neigh = X[A[i] > 0]
        agg = np.mean(np.vstack([X[i], neigh]), 0)
        out[i] = np.maximum(0, W @ agg)
    return out


if __name__ == "__main__":
    N = 6
    A = (np.random.rand(N, N) < 0.5).astype(float)
    np.fill_diagonal(A, 1)
    X = np.random.randn(N, 3)
    W = np.random.randn(3, 4)
    print("spectral GCN :", spectral_gcn(A, X, W).shape, "(uses full graph Laplacian)")
    print("spatial SAGE :", spatial_sage(A, X, W).shape, "(local neighbor sampling)")
