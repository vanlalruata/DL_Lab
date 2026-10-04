"""ext26: Compare LSTM vs GRU on a toy sequence task (PyTorch)."""
import torch
import torch.nn as nn


def build(model_type):
    if model_type == "lstm":
        return nn.LSTM(1, 16, batch_first=True)
    return nn.GRU(1, 16, batch_first=True)


if __name__ == "__main__":
    x = torch.randn(8, 20, 1)
    for name, m in [("LSTM", build("lstm")), ("GRU", build("gru"))]:
        out, _ = m(x)
        print(f"{name}: out {tuple(out.shape)} (GRU merges cell into hidden)")
