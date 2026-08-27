"""ext34: Stacked (multi-layer) LSTM and representational depth."""
import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    def __init__(self, in_s, hidden, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(in_s, hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


if __name__ == "__main__":
    for n in [1, 2, 4]:
        m = StackedLSTM(4, 16, n)
        params = sum(p.numel() for p in m.parameters())
        print(f"layers={n}: params={params:,} out={m(torch.randn(8,20,4)).shape}")
