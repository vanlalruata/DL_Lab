"""ext27: Train an LSTM for sentiment classification (synthetic IMDB-like data)."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMClassifier(nn.Module):
    def __init__(self, vocab=1000, emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


if __name__ == "__main__":
    # synthetic sequences of token ids (length 30)
    X = torch.randint(0, 1000, (200, 30))
    y = torch.randint(0, 2, (200,))
    dl = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    model = LSTMClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for xb, yb in dl:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    print("trained one epoch, final batch loss:", loss.item())
