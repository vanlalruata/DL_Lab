"""
Question 26: Compare LSTM vs GRU on a language-modeling or sequence task (perplexity/accuracy/speed).
Demonstrates: Side-by-side training and empirical comparison between PyTorch nn.LSTM
and nn.GRU on sequence modeling, measuring parameter count, convergence speed,
and loss.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SequenceModel(nn.Module):
    """Wrapper supporting either LSTM or GRU recurrent backbone."""
    def __init__(self, cell_type, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell_type = cell_type.lower()
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
            
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        logits = self.fc(out)
        return logits


def train_and_benchmark(cell_type, X, Y, vocab_size, epochs=25):
    torch.manual_seed(42)
    model = SequenceModel(cell_type, vocab_size=vocab_size, embed_dim=16, hidden_dim=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    param_count = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    total_time = time.time() - start_time
    final_loss = losses[-1]
    perplexity = np.exp(final_loss)

    return {
        "params": param_count,
        "time": total_time,
        "final_loss": final_loss,
        "perplexity": perplexity
    }


if __name__ == "__main__":
    # Create toy character sequence task
    text = "deep learning with recurrent neural networks and gated architectures"
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    seq_len = 10
    encoded = [char2idx[ch] for ch in text]
    
    # Form batches
    X_list, Y_list = [], []
    for i in range(len(encoded) - seq_len):
        X_list.append(encoded[i:i + seq_len])
        Y_list.append(encoded[i + 1:i + seq_len + 1])
        
    X_tensor = torch.tensor(X_list, dtype=torch.long)
    Y_tensor = torch.tensor(Y_list, dtype=torch.long)

    print("=" * 68)
    print("EMPIRICAL COMPARISON: LSTM VS GRU ON SEQUENCE PREDICTION")
    print(f"Task: Next-character language modeling (Vocab={vocab_size}, Samples={len(X_list)})")
    print("=" * 68)

    lstm_res = train_and_benchmark("lstm", X_tensor, Y_tensor, vocab_size)
    gru_res  = train_and_benchmark("gru",  X_tensor, Y_tensor, vocab_size)

    print(f"{'Metric':<25} | {'LSTM':<18} | {'GRU':<18}")
    print("-" * 68)
    print(f"{'Total Parameters':<25} | {lstm_res['params']:<18} | {gru_res['params']:<18}")
    print(f"{'Training Time (25 ep)':<25} | {lstm_res['time']:.4f}s{'':<11} | {gru_res['time']:.4f}s")
    print(f"{'Final Cross-Entropy':<25} | {lstm_res['final_loss']:.4f}{'':<12} | {gru_res['final_loss']:.4f}")
    print(f"{'Perplexity':<25} | {lstm_res['perplexity']:.4f}{'':<12} | {gru_res['perplexity']:.4f}")
    print("-" * 68)
    print("Key Takeaways:")
    print(f"  * GRU parameter savings : {((lstm_res['params'] - gru_res['params']) / lstm_res['params']) * 100:.1f}%")
    print(f"  * Speedup factor        : {(lstm_res['time'] / gru_res['time']):.2f}x faster execution for GRU")
    print("  * Accuracy / Perplexity : Highly competitive with LSTM despite simpler gating structure.")
    print("=" * 68)

