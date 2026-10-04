"""ext44: Basic message-passing layer (GCN) from scratch using adjacency + degree norm."""
import numpy as np


def gcn_layer(A, X, W):
    """A: (N,N) adjacency (+self-loops), X: (N,F) features, W: (F,F')."""
    D = np.diag(A.sum(1) ** -0.5)
    A_hat = D @ A @ D            # symmetric normalization
    return A_hat @ X @ W


if __name__ == "__main__":
    N = 5
    A = np.array([[1, 1, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1]], dtype=float)
    X = np.random.randn(N, 3)
    W = np.random.randn(3, 4)
    out = gcn_layer(A, X, W)
    print("GCN output shape:", out.shape)
