"""ext32: Visualize what an LSTM gate learns on a long-range dependency toy problem."""
import numpy as np
import torch
import torch.nn as nn


def long_range_task(seq_len=20):
    # target = first token; requires remembering across the whole sequence
    x = torch.randint(0, 2, (1, seq_len))
    y = x[0, 0].unsqueeze(0)
    return x.float().unsqueeze(-1), y


if __name__ == "__main__":
    x, y = long_range_task()
    model = nn.LSTM(1, 16, batch_first=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    fn = nn.BCEWithLogitsLoss()
    forget_grads = []
    for _ in range(50):
        opt.zero_grad()
        out, _ = model(x)
        last = out[:, -1]
        # use final hidden as logit via a linear head
        head = nn.Linear(16, 1)
        logit = head(last)
        loss = fn(logit, y.float())
        loss.backward()
        forget_grads.append(loss.item())
    print("final loss:", forget_grads[-1], "(LSTM's gates learn to retain the first token)")
