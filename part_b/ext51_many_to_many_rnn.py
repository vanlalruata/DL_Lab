"""
Part B — Question 51: Many-to-Many RNN Architecture (Sequence-to-Sequence / POS Tagging)
Demonstrates: Many-to-Many (output at every time step)
Real-world use cases: Machine Translation, POS Tagging, Named Entity Recognition, Video Frame Classification
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# VANILLA RNN FROM SCRATCH (NumPy) — Many-to-Many
# =============================================================================

class VanillaRNNManyToMany:
    """Vanilla RNN cell for Many-to-Many: produces output at each time step."""
    
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        
        # Xavier initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(1.0 / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.by = np.zeros((output_size, 1))
        
        # For BPTT
        self.cache = {}
    
    def forward(self, x_seq):
        """Forward pass through entire sequence.
        x_seq: (seq_len, input_size, batch_size)
        Returns: outputs (seq_len, output_size, batch_size), final_hidden
        """
        seq_len = x_seq.shape[0]
        batch_size = x_seq.shape[2]
        h = np.zeros((self.hidden_size, batch_size))
        outputs = np.zeros((seq_len, self.output_size, batch_size))
        
        # Store for BPTT
        hs = [h.copy()]
        
        for t in range(seq_len):
            x = x_seq[t]  # (input_size, batch_size)
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            outputs[t] = y
            hs.append(h.copy())
        
        self.cache = {'hs': hs, 'x_seq': x_seq}
        return outputs, h
    
    def backward(self, d_outputs):
        """Backward pass through time (BPTT).
        d_outputs: (seq_len, output_size, batch_size)
        """
        seq_len = d_outputs.shape[0]
        batch_size = d_outputs.shape[2]
        hs = self.cache['hs']
        x_seq = self.cache['x_seq']
        
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros((self.hidden_size, batch_size))
        
        for t in reversed(range(seq_len)):
            dy = d_outputs[t]  # (output_size, batch_size)
            h = hs[t + 1]
            h_prev = hs[t]
            x = x_seq[t]
            
            # Output layer gradients
            dWhy += dy @ h.T
            dby += np.sum(dy, axis=1, keepdims=True)
            dh = self.Why.T @ dy + dh_next
            
            # Tanh derivative
            dtanh = (1 - h * h) * dh
            
            # Hidden layer gradients
            dbh += np.sum(dtanh, axis=1, keepdims=True)
            dWxh += dtanh @ x.T
            dWhh += dtanh @ h_prev.T
            
            dh_next = self.Whh.T @ dtanh
        
        # Gradient clipping
        for grad in [dWxh, dWhh, dbh, dWhy, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        # Update parameters
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.bh -= self.lr * dbh
        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby
    
    def train_step(self, x_seq, y_seq):
        """Single training step."""
        outputs, _ = self.forward(x_seq)
        loss = np.mean((outputs - y_seq) ** 2)
        d_outputs = 2 * (outputs - y_seq) / outputs.size
        self.backward(d_outputs)
        return loss


# =============================================================================
# PYTORCH RNN — Many-to-Many
# =============================================================================

class PyTorchManyToManyRNN(nn.Module):
    """PyTorch RNN for Many-to-Many tasks (e.g., POS tagging)."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.bidirectional = bidirectional
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        out, _ = self.rnn(embedded)   # (batch, seq_len, hidden_size * num_dirs)
        out = self.fc(out)            # (batch, seq_len, output_size)
        return out


# =============================================================================
# REAL-WORLD EXAMPLE: POS TAGGING (Many-to-Many)
# =============================================================================

def create_pos_tagging_data():
    """Create synthetic POS tagging data."""
    # Vocabulary
    words = ['the', 'cat', 'dog', 'chased', 'ate', 'ran', 'fast', 'slowly', 'big', 'small']
    tags = ['DET', 'NOUN', 'VERB', 'ADV', 'ADJ']
    
    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    
    # Synthetic sentences with POS tags
    sentences = [
        (['the', 'big', 'cat', 'chased', 'the', 'small', 'dog'], ['DET', 'ADJ', 'NOUN', 'VERB', 'DET', 'ADJ', 'NOUN']),
        (['the', 'dog', 'ate', 'fast'], ['DET', 'NOUN', 'VERB', 'ADV']),
        (['the', 'small', 'cat', 'ran', 'slowly'], ['DET', 'ADJ', 'NOUN', 'VERB', 'ADV']),
        (['big', 'dog', 'chased', 'small', 'cat'], ['ADJ', 'NOUN', 'VERB', 'ADJ', 'NOUN']),
    ]
    
    X = [[word2idx[w] for w in sent] for sent, _ in sentences]
    y = [[tag2idx[t] for t in tags] for _, tags in sentences]
    
    max_len = max(len(s) for s in X)
    X_padded = np.array([s + [0] * (max_len - len(s)) for s in X])
    y_padded = np.array([t + [0] * (max_len - len(t)) for t in y])
    
    return X_padded, y_padded, word2idx, tag2idx, words, tags


def run_vanilla_many_to_many():
    """Run vanilla RNN Many-to-Many on POS tagging."""
    print("=" * 60)
    print("VANILLA RNN (NumPy) — Many-to-Many: POS Tagging")
    print("=" * 60)
    
    X, y, word2idx, tag2idx, words, tags = create_pos_tagging_data()
    vocab_size = len(words)
    tag_size = len(tags)
    
    # One-hot encode
    seq_len, batch_size = X.shape
    x_seq = np.zeros((seq_len, vocab_size, batch_size))
    y_seq = np.zeros((seq_len, tag_size, batch_size))
    
    for t in range(seq_len):
        for b in range(batch_size):
            x_seq[t, X[t, b], b] = 1
            y_seq[t, y[t, b], b] = 1
    
    # Train
    rnn = VanillaRNNManyToMany(vocab_size, 32, tag_size, lr=0.1)
    
    print(f"Vocab size: {vocab_size}, Tag size: {tag_size}")
    print(f"Sequence length: {seq_len}, Batch size: {batch_size}")
    
    for epoch in range(100):
        loss = rnn.train_step(x_seq, y_seq)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    # Test
    outputs, _ = rnn.forward(x_seq)
    preds = np.argmax(outputs, axis=1)
    
    print("\nPredictions vs Ground Truth:")
    for b in range(batch_size):
        sent_words = [words[X[t, b]] for t in range(seq_len) if X[t, b] != 0]
        true_tags = [tags[y[t, b]] for t in range(seq_len) if X[t, b] != 0]
        pred_tags = [tags[preds[t, b]] for t in range(seq_len) if X[t, b] != 0]
        print(f"  Words: {sent_words}")
        print(f"  True:  {true_tags}")
        print(f"  Pred:  {pred_tags}")
        print()


def run_pytorch_many_to_many():
    """Run PyTorch RNN Many-to-Many on POS tagging."""
    print("=" * 60)
    print("PYTORCH RNN — Many-to-Many: POS Tagging")
    print("=" * 60)
    
    X, y, word2idx, tag2idx, words, tags = create_pos_tagging_data()
    vocab_size = len(words) + 1  # +1 for padding
    tag_size = len(tags) + 1     # +1 for padding
    
    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    model = PyTorchManyToManyRNN(vocab_size, 16, 32, tag_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Vocab size: {vocab_size}, Tag size: {tag_size}")
    print(f"Model: {model}")
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)  # (batch, seq_len, tag_size)
        loss = criterion(outputs.view(-1, tag_size), y_tensor.view(-1))
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = outputs.argmax(dim=2)
    
    print("\nPredictions vs Ground Truth:")
    for b in range(X.shape[0]):
        sent_words = [words[X[b, t]] for t in range(X.shape[1]) if X[b, t] != 0]
        true_tags = [tags[y[b, t]] for t in range(X.shape[1]) if X[b, t] != 0]
        pred_tags = [tags[preds[b, t].item()] for t in range(X.shape[1]) if X[b, t] != 0]
        print(f"  Words: {sent_words}")
        print(f"  True:  {true_tags}")
        print(f"  Pred:  {pred_tags}")
        print()


if __name__ == "__main__":
    run_vanilla_many_to_many()
    run_pytorch_many_to_many()