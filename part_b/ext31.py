"""ext31: Masking padded sequences with pack_padded_sequence."""
import torch
import torch.nn as nn


def demo():
    lengths = torch.tensor([5, 3, 4])           # variable lengths in batch of 3
    x = torch.randn(3, 5, 8)                    # padded to max length 5
    x = torch.nn.utils.rnn.pack_padded_sequence(
        x, lengths, batch_first=True, enforce_sorted=False)
    rnn = nn.RNN(8, 16, batch_first=True)
    out, _ = rnn(x)
    out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    return out.shape


if __name__ == "__main__":
    print("padded output:", demo())
