"""
Part B — Question 53: Many-to-One RNN Architecture (Sequence Classification)
Demonstrates: Many-to-One (sequence input, single output)
Real-world use cases: Sentiment Analysis, Speech Recognition, Anomaly Detection, Handwriting Recognition
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# VANILLA RNN FROM SCRATCH (NumPy) — Many-to-One
# =============================================================================

class VanillaRNNManyToOne:
    """Vanilla RNN for Many-to-One: sequence input produces single output."""
    
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
        
        self.cache = {}
    
    def forward(self, x_seq):
        """Forward pass: process sequence, return single output.
        x_seq: (seq_len, input_size, batch_size)
        Returns: output (output_size, batch_size), final_hidden
        """
        seq_len = x_seq.shape[0]
        batch_size = x_seq.shape[2]
        h = np.zeros((self.hidden_size, batch_size))
        
        xs = []
        for t in range(seq_len):
            x = x_seq[t]
            xs.append(x)
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
        
        y = self.Why @ h + self.by
        self.cache = {'xs': xs, 'h': h, 'x_seq': x_seq}
        return y, h
    
    def backward(self, dy):
        """BPTT for Many-to-One."""
        xs = self.cache['xs']
        h = self.cache['h']
        x_seq = self.cache['x_seq']
        batch_size = dy.shape[1]
        
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        
        dh = self.Why.T @ dy
        dh_next = np.zeros((self.hidden_size, batch_size))
        
        for t in reversed(range(len(xs))):
            x = xs[t]
            h_prev = np.zeros((self.hidden_size, batch_size)) if t == 0 else self.cache['h_prev'][t - 1]
            
            dtanh = (1 - h * h) * dh + dh_next
            
            dWhy += dy @ h.T
            dby += dy
            dbh += dtanh
            dWxh += dtanh @ x.T
            dWhh += dtanh @ h_prev.T
            
            dh_next = self.Whh.T @ dtanh
        
        for grad in [dWxh, dWhh, dbh, dWhy, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.bh -= self.lr * dbh
        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby
    
    def train_step(self, x_seq, y_target):
        """Train on target label."""
        # Store h_prev for backward pass
        seq_len = x_seq.shape[0]
        batch_size = x_seq.shape[2]
        h_prev_list = []
        h = np.zeros((self.hidden_size, batch_size))
        for t in range(seq_len):
            h_prev_list.append(h.copy())
            h = np.tanh(self.Wxh @ x_seq[t] + self.Whh @ h + self.bh)
        self.cache['h_prev'] = h_prev_list
        
        y, _ = self.forward(x_seq)
        loss = np.mean((y - y_target) ** 2)
        dy = 2 * (y - y_target) / y.size
        self.backward(dy)
        return loss


# =============================================================================
# PYTORCH RNN — Many-to-One
# =============================================================================

class PyTorchManyToOneRNN(nn.Module):
    """PyTorch RNN for Many-to-One tasks (e.g., sentiment classification)."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.bidirectional = bidirectional
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        out, h_n = self.rnn(embedded)  # out: (batch, seq_len, hidden), h_n: (num_layers*dirs, batch, hidden)
        
        # Use the last hidden state for classification
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # forward + backward
        else:
            h_last = h_n[-1]  # (batch, hidden)
        
        output = self.fc(h_last)  # (batch, output_size)
        return output


# =============================================================================
# REAL-WORLD EXAMPLE: SENTIMENT ANALYSIS (Many-to-One)
# =============================================================================

def create_sentiment_data():
    """Create synthetic sentiment analysis data."""
    # Vocabulary
    words = ['the', 'movie', 'was', 'great', 'terrible', 'amazing', 'awful', 'good', 'bad',
             'i', 'love', 'hate', 'film', 'it', 'very', 'not', 'fun', 'boring', 'excellent', 'poor']
    pos_words = {'great', 'amazing', 'good', 'love', 'fun', 'excellent', 'movie', 'film', 'i', 'it'}
    neg_words = {'terrible', 'awful', 'bad', 'hate', 'boring', 'poor', 'not', 'was'}
    
    word2idx = {w: i for i, w in enumerate(words)}
    
    # Sentences labeled: 1=positive, 0=negative
    data = [
        (['the', 'movie', 'was', 'great'], 1),
        (['the', 'film', 'was', 'terrible'], 0),
        (['i', 'love', 'this', 'amazing', 'movie'], 1),
        (['it', 'was', 'awful', 'and', 'boring'], 0),
        (['the', 'good', 'film', 'is', 'great'], 1),
        (['i', 'hate', 'this', 'terrible', 'movie'], 0),
        (['it', 'was', 'very', 'good', 'and', 'fun'], 1),
        (['the', 'awful', 'film', 'is', 'poor'], 0),
        (['not', 'great', 'but', 'not', 'awful'], 0),
        (['amazing', 'excellent', 'love', 'it'], 1),
    ]
    
    # Replace 'this' and 'and' with known word indices
    extra_words = {'this': 0, 'and': 0}  # padding/unknown
    
    X = []
    y = []
    for sent, label in data:
        x = [word2idx.get(w, 0) for w in sent]
        X.append(x)
        y.append(label)
    
    max_len = max(len(s) for s in X)
    X_padded = np.array([s + [0] * (max_len - len(s)) for s in X])
    y_labels = np.array(y)
    
    return X_padded, y_labels, word2idx, words, max_len


def run_vanilla_many_to_one():
    """Run vanilla RNN Many-to-One on sentiment analysis."""
    print("=" * 60)
    print("VANILLA RNN (NumPy) — Many-to-One: Sentiment Analysis")
    print("=" * 60)
    
    X, y, word2idx, words, max_len = create_sentiment_data()
    vocab_size = len(words)
    
    # One-hot encode
    batch_size = X.shape[0]
    x_seq = np.zeros((max_len, vocab_size, batch_size))
    y_target = np.zeros((1, batch_size))
    
    for t in range(max_len):
        for b in range(batch_size):
            if X[b, t] < vocab_size:
                x_seq[t, X[b, t], b] = 1
    
    for b in range(batch_size):
        y_target[0, b] = y[b]
    
    # Train
    rnn = VanillaRNNManyToOne(vocab_size, 32, 1, lr=0.1)
    
    print(f"Vocab size: {vocab_size}, Max seq len: {max_len}")
    print(f"Batch size: {batch_size}")
    
    for epoch in range(200):
        loss = rnn.train_step(x_seq, y_target)
        if epoch % 40 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    # Evaluate
    y_pred, _ = rnn.forward(x_seq)
    preds = (y_pred > 0.5).astype(int)
    accuracy = np.mean(preds == y_target)
    
    print(f"\nAccuracy: {accuracy * 100:.1f}%")
    print("\nPredictions vs Ground Truth:")
    for b in range(batch_size):
        sent = [words[X[b, t]] for t in range(max_len) if X[b, t] != 0 and words[X[b, t]] != 'this' and words[X[b, t]] != 'and']
        print(f"  Sentence: {' '.join(sent)}")
        print(f"  True: {int(y[b])} | Pred: {preds[0, b]}")
        print()


def run_pytorch_many_to_one():
    """Run PyTorch RNN Many-to-One on sentiment analysis."""
    print("\n" + "=" * 60)
    print("PYTORCH RNN — Many-to-One: Sentiment Analysis")
    print("=" * 60)
    
    X, y, word2idx, words, max_len = create_sentiment_data()
    vocab_size = len(words)
    
    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    model = PyTorchManyToOneRNN(vocab_size, 16, 32, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Vocab size: {vocab_size}, Max seq len: {max_len}")
    print(f"Model: {model}")
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tensor).squeeze()
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            preds = (logits > 0).float()
            acc = (preds == y_tensor).float().mean().item()
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).squeeze()
        preds = (logits > 0).float()
        accuracy = (preds == y_tensor).float().mean().item()
    
    print(f"\nFinal Accuracy: {accuracy * 100:.1f}%")
    print("\nPredictions vs Ground Truth:")
    for b in range(X.shape[0]):
        sent = [words[X[b, t]] for t in range(max_len) if X[b, t] != 0 and words[X[b, t]] not in ('this', 'and')]
        true_label = int(y[b])
        pred_label = int(preds[b].item())
        print(f"  Sentence: {' '.join(sent)}")
        print(f"  True: {true_label} | Pred: {pred_label}")
        print()


# =============================================================================
# REAL-WORLD EXAMPLE: SPEECH RECOGNITION (Many-to-One)
# =============================================================================

def demo_speech_recognition():
    """Demonstrate speech recognition as Many-to-One."""
    print("\n" + "=" * 60)
    print("SPEECH RECOGNITION (Many-to-One Real-World Use Case)")
    print("=" * 60)
    print("""
    Architecture: Audio Waveform (many time steps) → Text Transcript (single output)
    
    Many-to-One flow:
    Audio frames [t=1...T] → RNN processes all frames → Final hidden state → 
    → CTC Loss / Classification → Transcribed text
    
    Real examples:
    - Google Speech-to-Text
    - Apple Siri transcription
    - Amazon Alexa wake word detection
    """)


# =============================================================================
# REAL-WORLD EXAMPLE: ANOMALY DETECTION (Many-to-One)
# =============================================================================

def demo_anomaly_detection():
    """Demonstrate anomaly detection as Many-to-One."""
    print("=" * 60)
    print("TIME SERIES ANOMALY DETECTION (Many-to-One Real-World Use Case)")
    print("=" * 60)
    print("""
    Architecture: Sensor Readings (many time steps) → Normal/Anomaly label (single output)
    
    Many-to-One flow:
    Sensor readings [t=1...T] → RNN processes all readings → Final hidden state →
    → Sigmoid/Softmax → Normal (0) or Anomaly (1)
    
    Real examples:
    - Industrial equipment monitoring (vibration sensors)
    - Network intrusion detection (packet rates over time)
    - ECG heartbeat classification (cardiac signals)
    """)


if __name__ == "__main__":
    run_vanilla_many_to_one()
    run_pytorch_many_to_one()
    demo_speech_recognition()
    demo_anomaly_detection()