"""
Part B — Question 52: One-to-Many RNN Architecture (Sequence Generation)
Demonstrates: One-to-Many (single input, sequence of outputs)
Real-world use cases: Image Captioning, Text Generation, Music Generation, Time Series Forecasting
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# VANILLA RNN FROM SCRATCH (NumPy) — One-to-Many
# =============================================================================

class VanillaRNNOneToMany:
    """Vanilla RNN for One-to-Many: single input generates sequence of outputs."""
    
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
    
    def forward(self, x_init, seq_len):
        """Generate sequence from single input.
        x_init: (input_size, 1) - initial input (e.g., image features, start token)
        Returns: outputs (seq_len, output_size, 1), hidden_states
        """
        h = np.zeros((self.hidden_size, 1))
        outputs = np.zeros((seq_len, self.output_size, 1))
        hs = [h.copy()]
        
        x = x_init
        for t in range(seq_len):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            outputs[t] = y
            hs.append(h.copy())
            # Next input is the previous output (for autoregressive generation)
            x = y
        
        self.cache = {'hs': hs, 'x_init': x_init}
        return outputs, h
    
    def backward(self, d_outputs):
        """BPTT for One-to-Many."""
        seq_len = d_outputs.shape[0]
        hs = self.cache['hs']
        x_init = self.cache['x_init']
        
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros((self.hidden_size, 1))
        
        for t in reversed(range(seq_len)):
            dy = d_outputs[t]
            h = hs[t + 1]
            h_prev = hs[t]
            x = x_init if t == 0 else self.Why @ hs[t] + self.by
            
            dWhy += dy @ h.T
            dby += dy
            dh = self.Why.T @ dy + dh_next
            dtanh = (1 - h * h) * dh
            
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
    
    def train_step(self, x_init, y_seq):
        """Train on target sequence."""
        outputs, _ = self.forward(x_init, y_seq.shape[0])
        loss = np.mean((outputs - y_seq) ** 2)
        d_outputs = 2 * (outputs - y_seq) / outputs.size
        self.backward(d_outputs)
        return loss


# =============================================================================
# PYTORCH RNN — One-to-Many
# =============================================================================

class PyTorchOneToManyRNN(nn.Module):
    """PyTorch RNN for One-to-Many tasks (e.g., image captioning, music generation)."""
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initial projection from input (e.g., image features) to hidden
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x_init, seq_len):
        # x_init: (batch, input_size) - single input per batch
        batch_size = x_init.size(0)
        
        # Project initial input to hidden state
        h0 = self.input_proj(x_init).unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # Generate sequence autoregressively
        outputs = []
        x = torch.zeros(batch_size, 1, self.hidden_size, device=x_init.device)
        
        for _ in range(seq_len):
            out, h0 = self.rnn(x, h0)
            out = self.fc(out)  # (batch, 1, output_size)
            outputs.append(out)
            x = out  # Next input is previous output (autoregressive)
        
        return torch.cat(outputs, dim=1)  # (batch, seq_len, output_size)


# =============================================================================
# REAL-WORLD EXAMPLE: MUSIC GENERATION (One-to-Many)
# =============================================================================

def create_music_data():
    """Create simple music sequence data (note indices)."""
    # Simple melody: C D E F G F E D C (C major scale)
    notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'F4', 'E4', 'D4', 'C4']
    note2idx = {n: i for i, n in enumerate(sorted(set(notes)))}
    idx2note = {i: n for n, i in note2idx.items()}
    vocab_size = len(note2idx)
    
    # Training sequences: each sequence starts with a "start" token and generates melody
    sequences = []
    for i in range(len(notes) - 4):
        seq = [note2idx[notes[j]] for j in range(i, i + 5)]
        sequences.append(seq)
    
    return sequences, note2idx, idx2note, vocab_size


def run_vanilla_one_to_many():
    """Run vanilla RNN One-to-Many on music generation."""
    print("=" * 60)
    print("VANILLA RNN (NumPy) — One-to-Many: Music Generation")
    print("=" * 60)
    
    sequences, note2idx, idx2note, vocab_size = create_music_data()
    hidden_size = 32
    
    # Prepare data: first note as input, rest as targets
    X_init = []
    Y_seq = []
    for seq in sequences:
        x = np.zeros((vocab_size, 1))
        x[seq[0], 0] = 1
        X_init.append(x)
        
        y = np.zeros((len(seq) - 1, vocab_size, 1))
        for t, note_idx in enumerate(seq[1:]):
            y[t, note_idx, 0] = 1
        Y_seq.append(y)
    
    rnn = VanillaRNNOneToMany(vocab_size, hidden_size, vocab_size, lr=0.1)
    
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}")
    print(f"Training sequences: {len(sequences)}")
    
    for epoch in range(200):
        total_loss = 0
        for x_init, y_seq in zip(X_init, Y_seq):
            loss = rnn.train_step(x_init, y_seq)
            total_loss += loss
        if epoch % 40 == 0:
            print(f"Epoch {epoch}: Avg Loss = {total_loss/len(sequences):.4f}")
    
    # Generate
    print("\nGenerated sequences:")
    for i, (x_init, y_seq) in enumerate(zip(X_init, Y_seq)):
        outputs, _ = rnn.forward(x_init, y_seq.shape[0])
        preds = np.argmax(outputs, axis=1).flatten()
        start_note = idx2note[np.argmax(x_init)]
        true_notes = [idx2note[np.argmax(y_seq[t])] for t in range(y_seq.shape[0])]
        pred_notes = [idx2note[p] for p in preds]
        print(f"  Seed: {start_note} | True: {true_notes} | Pred: {pred_notes}")


def run_pytorch_one_to_many():
    """Run PyTorch RNN One-to-Many on music generation."""
    print("\n" + "=" * 60)
    print("PYTORCH RNN — One-to-Many: Music Generation")
    print("=" * 60)
    
    sequences, note2idx, idx2note, vocab_size = create_music_data()
    hidden_size = 32
    
    # Prepare data
    X_init = torch.tensor([s[0] for s in sequences], dtype=torch.long)
    Y_seq = torch.tensor([s[1:] for s in sequences], dtype=torch.long)
    
    # One-hot encode initial input
    X_init_onehot = torch.nn.functional.one_hot(X_init, vocab_size).float()
    
    model = PyTorchOneToManyRNN(vocab_size, hidden_size, vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Vocab size: {vocab_size}, Hidden size: {hidden_size}")
    print(f"Training sequences: {len(sequences)}")
    print(f"Model: {model}")
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_init_onehot, Y_seq.shape[1])  # (batch, seq_len, vocab_size)
        loss = criterion(outputs.view(-1, vocab_size), Y_seq.view(-1))
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model(X_init_onehot, Y_seq.shape[1])
        preds = outputs.argmax(dim=2)
    
    print("\nGenerated sequences:")
    for i in range(len(sequences)):
        start_note = idx2note[sequences[i][0]]
        true_notes = [idx2note[n] for n in sequences[i][1:]]
        pred_notes = [idx2note[p.item()] for p in preds[i]]
        print(f"  Seed: {start_note} | True: {true_notes} | Pred: {pred_notes}")


# =============================================================================
# IMAGE CAPTIONING EXAMPLE (One-to-Many)
# =============================================================================

class ImageCaptioningRNN(nn.Module):
    """One-to-Many RNN for image captioning: CNN features -> caption."""
    
    def __init__(self, feature_size, embed_dim, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.feature_proj = nn.Linear(feature_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # features: (batch, feature_size)
        # captions: (batch, seq_len) - teacher forcing during training
        batch_size = features.size(0)
        
        # Initialize hidden state from image features
        h0 = self.feature_proj(features).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        
        # Embed captions
        embeds = self.embedding(captions)  # (batch, seq_len, embed_dim)
        
        # LSTM forward
        out, _ = self.rnn(embeds, (h0, c0))
        out = self.fc(out)
        return out


def demo_image_captioning():
    """Demo image captioning architecture."""
    print("\n" + "=" * 60)
    print("IMAGE CAPTIONING ARCHITECTURE (One-to-Many)")
    print("=" * 60)
    
    vocab = ['<start>', '<end>', 'a', 'cat', 'dog', 'on', 'mat', 'the', 'is']
    vocab_size = len(vocab)
    feature_size = 2048  # ResNet-50 features
    embed_dim = 128
    hidden_size = 256
    
    model = ImageCaptioningRNN(feature_size, embed_dim, hidden_size, vocab_size)
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simulate forward pass
    batch_size = 2
    features = torch.randn(batch_size, feature_size)
    captions = torch.tensor([[0, 2, 3, 5, 6, 1], [0, 2, 4, 5, 7, 1]])  # "a cat on mat", "a dog on the"
    
    outputs = model(features, captions)
    print(f"Input features: {features.shape}")
    print(f"Input captions: {captions.shape}")
    print(f"Output logits: {outputs.shape}")
    print("\nThis demonstrates One-to-Many: single image features -> sequence of words")


if __name__ == "__main__":
    run_vanilla_one_to_many()
    run_pytorch_one_to_many()
    demo_image_captioning()