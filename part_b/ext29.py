"""ext29: Bidirectional RNN/LSTM and when future context helps."""
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, in_s, hidden):
        super().__init__()
        self.lstm = nn.LSTM(in_s, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        out, _ = self.lstm(x)           # out contains both directions
        return self.fc(out[:, -1])       # final forward + backward concat


if __name__ == "__main__":
    x = torch.randn(8, 20, 4)
    print("bidirectional out:", BiLSTMClassifier(4, 16)(x).shape)
    print("Future context (right-to-left) helps when full sequence is available.")
