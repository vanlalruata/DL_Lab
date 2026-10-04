"""ext50: Readout/pooling layer for graph-level classification (molecular property)."""
import numpy as np


def global_mean_pool(node_feats, batch):
    """node_feats: (N, F), batch: (N,) graph id -> (G, F)."""
    G = int(batch.max()) + 1
    out = np.zeros((G, node_feats.shape[1]))
    counts = np.zeros(G)
    for n in range(len(batch)):
        out[batch[n]] += node_feats[n]
        counts[batch[n]] += 1
    return out / counts[:, None]


def global_max_pool(node_feats, batch):
    G = int(batch.max()) + 1
    out = np.full((G, node_feats.shape[1]), -np.inf)
    for n in range(len(batch)):
        out[batch[n]] = np.maximum(out[batch[n]], node_feats[n])
    return out


if __name__ == "__main__":
    nodes = np.random.randn(7, 5)
    batch = np.array([0, 0, 0, 1, 1, 2, 2])  # 3 graphs
    print("mean-pool graphs:", global_mean_pool(nodes, batch).shape)
    print("max-pool graphs :", global_max_pool(nodes, batch).shape)
