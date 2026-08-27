"""ext23: Gradient clipping and its effect on RNN training stability."""
import torch
import torch.nn as nn


def train_with_clip(clip_value):
    torch.manual_seed(0)
    rnn = nn.RNN(1, 16, batch_first=True)
    opt = torch.optim.SGD(rnn.parameters(), lr=1.0)  # large lr -> unstable
    loss_fn = nn.MSELoss()
    x = torch.randn(4, 10, 1)
    y = torch.randn(4, 10, 1)
    opt.zero_grad()
    out, _ = rnn(x)
    loss = loss_fn(out, y)
    loss.backward()
    if clip_value:
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip_value)
    opt.step()
    return max(p.grad.abs().max().item() for p in rnn.parameters() if p.grad is not None)


if __name__ == "__main__":
    print("max grad no clip:", f"{train_with_clip(0):.2e}")
    print("max grad clipped (1.0):", f"{train_with_clip(1.0):.2e}")
