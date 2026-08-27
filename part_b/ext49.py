"""ext49: GNN for link prediction with negative sampling (PyTorch geometric-style)."""
import numpy as np


def negative_sample(edge_index, num_nodes, n):
    """Sample `n` node pairs not present in edge_index."""
    edges = set(map(tuple, edge_index.T))
    neg = []
    while len(neg) < n:
        i, j = np.random.randint(0, num_nodes, 2)
        if (i, j) not in edges and i != j:
            neg.append((i, j))
    return np.array(neg)


def dot_score(emb, pairs):
    return np.einsum("ij,ij->i", emb[pairs[:, 0]], emb[pairs[:, 1]])


if __name__ == "__main__":
    num_nodes = 10
    edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])  # undirected edges
    emb = np.random.randn(num_nodes, 8)
    pos = edge_index.T
    neg = negative_sample(edge_index, num_nodes, 4)
    print("pos scores:", dot_score(emb, pos))
    print("neg scores:", dot_score(emb, neg))
    print("Higher pos vs neg scores -> successful link prediction.")
