"""
Part B — Question 55: PyTorch RNN — All Architectures (Practical Guide)
Demonstrates: Many-to-Many, One-to-Many, Many-to-One using nn.RNN
Real-world examples included for each architecture
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# =============================================================================
# ARCHITECTURE 1: MANY-TO-MANY (POS Tagging)
# =============================================================================

class ManyToManyRNN(nn.Module):
    """Many-to-Many RNN for POS Tagging.
    
    Input:  Sequence of word embeddings  (batch, seq_len, embed_dim)
    Output: Sequence of tag probabilities (batch, seq_len, num_tags)
    
    Real-world: POS tagging, NER, Machine Translation
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_tags, num_layers=1, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_tags)
    
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = self.fc(out)
        return out


# =============================================================================
# ARCHITECTURE 2: ONE-TO-MANY (Music Generation)
# =============================================================================

class OneToManyRNN(nn.Module):
    """One-to-Many RNN for Music/Sequence Generation.
    
    Input:  Single conditioning vector (batch, feature_dim)
    Output: Generated sequence (batch, seq_len, vocab_size)
    
    Real-world: Music generation, Image captioning, Text generation
    """
    
    def __init__(self, feature_dim, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(feature_dim, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x_init, seq_len):
        batch_size = x_init.size(0)
        h0 = self.input_proj(x_init).unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        outputs = []
        x = torch.zeros(batch_size, 1, self.hidden_size, device=x_init.device)
        
        for _ in range(seq_len):
            out, h0 = self.rnn(x, h0)
            out = self.fc(out)
            outputs.append(out)
            x = out
        
        return torch.cat(outputs, dim=1)


# =============================================================================
# ARCHITECTURE 3: MANY-TO-ONE (Sentiment Classification)
# =============================================================================

class ManyToOneRNN(nn.Module):
    """Many-to-One RNN for Sentiment Classification.
    
    Input:  Sequence of word embeddings (batch, seq_len, embed_dim)
    Output: Single classification logits (batch, num_classes)
    
    Real-world: Sentiment analysis, Speech recognition, Anomaly detection
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=1, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        out, h_n = self.rnn(embedded)
        h_last = h_n[-1]
        output = self.fc(h_last)
        return output


# =============================================================================
# REAL-WORLD DEMO 1: SENTIMENT ANALYSIS (Many-to-One)
# =============================================================================

def demo_sentiment_analysis():
    """Many-to-One: Sentiment classification with PyTorch."""
    print("=" * 60)
    print("MANY-TO-ONE: Sentiment Analysis (PyTorch)")
    print("=" * 60)
    
    # Vocabulary
    words = ['<pad>', 'the', 'movie', 'was', 'great', 'terrible', 'amazing',
             'awful', 'good', 'bad', 'i', 'love', 'hate', 'film', 'very',
             'not', 'fun', 'boring', 'excellent', 'poor']
    word2idx = {w: i for i, w in enumerate(words)}
    vocab_size = len(words)
    
    # Data: (sentence, label)
    data = [
        (['the', 'movie', 'was', 'great'], 1),
        (['the', 'film', 'was', 'terrible'], 0),
        (['i', 'love', 'this', 'amazing'], 1),
        (['it', 'was', 'awful', 'and', 'boring'], 0),
        (['good', 'and', 'great', 'film'], 1),
        (['hate', 'this', 'poor', 'movie'], 0),
        (['very', 'good', 'excellent'], 1),
        (['terrible', 'boring', 'awful'], 0),
        (['not', 'great', 'not', 'bad'], 1),
        (['love', 'amazing', 'excellent'], 1),
    ]
    
    max_len = 6
    
    X = []
    y = []
    for sent, label in data:
        x = [word2idx.get(w, 0) for w in sent]
        x = x[:max_len] + [0] * (max_len - len(x[:max_len]))
        X.append(x)
        y.append(label)
    
    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    model = ManyToOneRNN(vocab_size, 16, 32, 1, bidirectional=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: Embedding -> Bidirectional RNN -> Linear -> Sigmoid")
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tensor).squeeze()
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = (logits > 0).float()
                acc = (preds == y_tensor).float().mean().item()
            print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.4f}, Acc: {acc:.4f}")
    
    print("\nPredictions:")
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).squeeze()
        preds = (logits > 0).float()
        for i in range(len(data)):
            sent = ' '.join([w for w in data[i][0]])
            print(f"  '{sent}' | True: {data[i][1]} | Pred: {int(preds[i].item())}")


# =============================================================================
# REAL-WORLD DEMO 2: TEXT SEQUENCE GENERATION (One-to-Many)
# =============================================================================

def demo_text_generation():
    """One-to-Many: Character-level text generation with PyTorch."""
    print("\n" + "=" * 60)
    print("ONE-TO-MANY: Text Generation (PyTorch)")
    print("=" * 60)
    
    chars = "abcdefghijklmnopqrstuvwxyz .,"
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(chars)
    
    corpus = "hello world, hi there. hello again."
    
    seq_len = 5
    X = []
    Y = []
    for i in range(len(corpus) - seq_len):
        X.append([char2idx[c] for c in corpus[i:i + seq_len]])
        Y.append([char2idx[c] for c in corpus[i + 1:i + seq_len + 1]])
    
    X_tensor = torch.tensor(X, dtype=torch.long)
    Y_tensor = torch.tensor(Y, dtype=torch.long)
    
    model = OneToManyRNN(vocab_size, 32, vocab_size, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Corpus: '{corpus}'")
    print(f"Sequence length: {seq_len}")
    print(f"Architecture: Input Projection -> RNN -> Linear (autoregressive)")
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor[:, 0].float().unsqueeze(1).float(), seq_len)
        loss = criterion(outputs.view(-1, vocab_size), Y_tensor.view(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100 - Loss: {loss.item():.4f}")
    
    print("\nNote: For proper training, input should be embedded characters.")
    print("This demonstrates the One-to-Many architecture pattern.")


# =============================================================================
# REAL-WORLD DEMO 3: MACHINE TRANSLATION (Many-to-Many)
# =============================================================================

def demo_machine_translation():
    """Many-to-Many: Machine Translation concept with PyTorch."""
    print("\n" + "=" * 60)
    print("MANY-TO-MANY: Machine Translation (PyTorch)")
    print("=" * 60)
    
    src_words = ['<pad>', 'the', 'cat', 'dog', 'chased', 'ate', 'is', 'on', 'mat']
    tgt_words = ['<pad>', 'le', 'chat', 'chien', 'a', 'mange', 'est', 'sur', 'tapis']
    
    src2idx = {w: i for i, w in enumerate(src_words)}
    tgt2idx = {w: i for i, w in enumerate(tgt_words)}
    
    vocab_size = len(src_words)
    max_len = 5
    
    pairs = [
        (['the', 'cat', 'is', 'on', 'mat'], ['le', 'chat', 'est', 'sur', 'tapis']),
        (['the', 'dog', 'is', 'on', 'mat'], ['le', 'chien', 'est', 'sur', 'tapis']),
    ]
    
    print(f"Source vocab: {vocab_size}")
    print(f"Target vocab: {vocab_size}")
    print(f"Architecture: Embedding -> RNN -> Linear -> Softmax")
    print("Task: Translate English to French (sequence to sequence)")
    print()
    
    src_sentence = pairs[0][0]
    tgt_sentence = pairs[0][1]
    print(f"English:  {' '.join(src_sentence)}")
    print(f"French:   {' '.join(tgt_sentence)}")
    
    model = ManyToManyRNN(vocab_size, 16, 32, vocab_size, bidirectional=False)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nThis is the Many-to-Many pattern: each input token produces an output token.")


# =============================================================================
# RNN vs LSTM vs GRU COMPARISON
# =============================================================================

def compare_rnn_variants():
    """Compare vanilla RNN, LSTM, and GRU."""
    print("\n" + "=" * 60)
    print("RNN VARIANT COMPARISON")
    print("=" * 60)
    print()
    
    seq_len, batch, features = 20, 4, 8
    x = torch.randn(batch, seq_len, features)
    
    rnn = nn.RNN(features, 16, batch_first=True)
    lstm = nn.LSTM(features, 16, batch_first=True)
    gru = nn.GRU(features, 16, batch_first=True)
    
    with torch.no_grad():
        out_rnn, h_rnn = rnn(x)
        out_lstm, (h_lstm, c_lstm) = lstm(x)
        out_gru, h_gru = gru(x)
    
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"RNN output:   {tuple(out_rnn.shape)}, hidden: {tuple(h_rnn.shape)}")
    print(f"LSTM output:  {tuple(out_lstm.shape)}, hidden: {tuple(h_lstm.shape)}, cell: {tuple(c_lstm.shape)}")
    print(f"GRU output:   {tuple(out_gru.shape)}, hidden: {tuple(h_gru.shape)}")
    print()
    print("Vanilla RNN:  Simple, fast, but suffers from vanishing/exploding gradients")
    print("LSTM:         3 gates (forget, input, output) + cell state; handles long-range deps")
    print("GRU:          2 gates (reset, update); similar to LSTM but fewer parameters")
    print()
    print("Parameter counts:")
    print(f"  RNN:  {sum(p.numel() for p in rnn.parameters()):,}")
    print(f"  LSTM: {sum(p.numel() for p in lstm.parameters()):,}  (4x weights for gates + cell)")
    print(f"  GRU:  {sum(p.numel() for p in gru.parameters()):,}  (3x weights for gates)")


# =============================================================================
# TRAINING WITH TEACHER FORCING (One-to-Many)
# =============================================================================

def demonstrate_teacher_forcing():
    """Show teacher forcing vs no teacher forcing for sequence generation."""
    print("\n" + "=" * 60)
    print("TEACHER FORCING vs SCHEDULED SAMPLING")
    print("=" * 60)
    print("""
    Teacher Forcing:
      During training, feed the GROUND TRUTH as next input.
      - Faster convergence
      - But causes exposure mismatch at test time (using own predictions)
    
    Scheduled Sampling:
      Gradually switch from ground truth to model's own predictions during training.
      - Slower convergence
      - Better alignment between train and test time
    
    No Teacher Forcing (autoregressive):
      Always use model's own previous output as next input.
      - Most realistic
      - Hardest to train (errors accumulate)
    
    Usage: One-to-Many tasks (text gen, music gen, machine translation)
    """)


if __name__ == "__main__":
    demo_sentiment_analysis()
    demo_text_generation()
    demo_machine_translation()
    compare_rnn_variants()
    demonstrate_teacher_forcing()