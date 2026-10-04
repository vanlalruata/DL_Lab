"""ext33: Teacher forcing vs scheduled sampling in sequence generation."""
import random
import torch
import torch.nn as nn


class SeqModel(nn.Module):
    def __init__(self, vocab=10, hidden=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.rnn = nn.RNN(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, y, teacher_forcing, tf_ratio):
        x = self.emb(x)
        h = None
        logits = []
        inp = x[:, 0:1]
        T = y.size(1)
        for t in range(T):
            out, h = self.rnn(inp, h)
            logit = self.fc(out[:, -1])
            logits.append(logit)
            if t == T - 1:
                break
            use_tf = teacher_forcing and (random.random() < tf_ratio)
            nxt = y[:, t:t + 1] if use_tf else logit.argmax(1, keepdim=True)
            inp = self.emb(nxt)
        return torch.stack(logits, 1)


if __name__ == "__main__":
    m = SeqModel()
    x = torch.randint(0, 10, (4, 8))
    y = torch.randint(0, 10, (4, 8))
    print("teacher forcing:", m(x, y, True, 1.0).shape)
    print("scheduled (ratio 0.5):", m(x, y, True, 0.5).shape)
