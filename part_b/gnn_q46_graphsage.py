"""ext46: GraphSAGE layer with neighbor sampling (inductive learning)."""
import numpy as np


def aggregate(neigh_feats):
    return np.mean(neigh_feats, axis=0)  # mean aggregator


def sage_layer(W, x_node, neigh_feats, bias=0.0):
    """Concat self + aggregated neighbors, then transform."""
    agg = aggregate(neigh_feats)
    h = np.concatenate([x_node, agg])
    return np.maximum(0, W @ h + bias)  # ReLU


if __name__ == "__main__":
    x_node = np.random.randn(4)
    neigh = np.random.randn(3, 4)        # 3 neighbors
    W = np.random.randn(8, 8)
    out = sage_layer(W, x_node, neigh)
    print("GraphSAGE output shape:", out.shape, "(inductive: works on unseen nodes)")
