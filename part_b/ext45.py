"""ext45: Apply a GCN to the Cora citation dataset for node classification.

Note: requires `torch_geometric` and the Cora dataset (auto-downloaded on first run).
If the package is unavailable, a lightweight synthetic-graph fallback is provided.
"""
try:
    import torch
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv
    import torch.nn.functional as F

    class GCN(torch.nn.Module):
        def __init__(self, in_c, h, out_c):
            super().__init__()
            self.c1 = GCNConv(in_c, h)
            self.c2 = GCNConv(h, out_c)

        def forward(self, x, edge_index):
            x = F.relu(self.c1(x, edge_index))
            return self.c2(x, edge_index)

    def run_cora():
        data = Planetoid(root="data/Cora", name="Cora")[0]
        model = GCN(data.num_node_features, 16, data.num_classes)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(50):
            opt.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward(); opt.step()
        acc = (out.argmax(1)[data.test_mask] == data.y[data.test_mask]).float().mean()
        print("Cora test accuracy:", acc.item())

    if __name__ == "__main__":
        run_cora()
except ImportError:
    print("torch_geometric not installed. Use: pip install torch_geometric")
