"""
Part B — Question 54: Vanilla RNN from Scratch (NumPy Implementation)
Demonstrates: Pure NumPy RNN cell, forward pass, BPTT, and training
No PyTorch, no frameworks — all math explicitly coded
"""
import numpy as np


# =============================================================================
# CORE VANILLA RNN CELL
# =============================================================================

class VanillaRNNCell:
    """A single vanilla RNN cell with tanh activation.
    
    Equations:
        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_hy @ h_t + b_y
    
    This is the most fundamental building block of all RNN variants.
    """
    
    def __init__(self, input_size, hidden_size, output_size=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        
        # Xavier/Glorot initialization
        scale_x = np.sqrt(2.0 / (input_size + hidden_size))
        scale_h = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale_y = np.sqrt(2.0 / (hidden_size + self.output_size))
        
        self.Wxh = np.random.randn(hidden_size, input_size) * scale_x
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale_h
        self.bh = np.zeros((hidden_size, 1))
        self.Why = np.random.randn(self.output_size, hidden_size) * scale_y
        self.by = np.zeros((self.output_size, 1))
        
        # Momentum terms (for basic SGD with momentum)
        self.vWxh = np.zeros_like(self.Wxh)
        self.vWhh = np.zeros_like(self.Whh)
        self.vbh = np.zeros_like(self.bh)
        self.vWhy = np.zeros_like(self.Why)
        self.vby = np.zeros_like(self.by)
        
        self.t = 0
    
    def forward(self, x, h_prev):
        """Single RNN step.
        
        Args:
            x: (input_size, 1) input vector
            h_prev: (hidden_size, 1) previous hidden state
        
        Returns:
            y: (output_size, 1) output
            h: (hidden_size, 1) new hidden state
        """
        z_h = self.Wxh @ x + self.Whh @ h_prev + self.bh
        h = np.tanh(z_h)
        y = self.Why @ h + self.by
        return y, h
    
    def forward_sequence(self, x_seq):
        """Forward pass through entire sequence.
        
        Args:
            x_seq: (seq_len, input_size, batch_size) input sequence
        
        Returns:
            y_seq: (seq_len, output_size, batch_size) output sequence
            h_final: (hidden_size, batch_size) final hidden state
            cache: dict with intermediate values for BPTT
        """
        seq_len, _, batch_size = x_seq.shape
        h = np.zeros((self.hidden_size, batch_size))
        y_seq = np.zeros((seq_len, self.output_size, batch_size))
        h_list = [h.copy()]
        
        for t in range(seq_len):
            x_t = x_seq[t]  # (input_size, batch_size)
            y_t, h = self.forward_batch(x_t, h)
            y_seq[t] = y_t
            h_list.append(h.copy())
        
        cache = {'h_list': h_list, 'x_seq': x_seq, 'y_seq': y_seq}
        return y_seq, h, cache
    
    def forward_batch(self, x, h_prev):
        """Forward pass for batch input."""
        z_h = self.Wxh @ x + self.Whh @ h_prev + self.bh
        h = np.tanh(z_h)
        y = self.Why @ h + self.by
        return y, h
    
    def backward(self, d_y_seq, cache, clip_value=5.0):
        """Backprop Through Time (BPTT).
        
        Args:
            d_y_seq: (seq_len, output_size, batch_size) gradient of loss w.r.t. outputs
            cache: cache from forward pass
            clip_value: gradient clipping threshold
        
        Returns:
            loss: scalar
        """
        y_seq = cache['y_seq']
        x_seq = cache['x_seq']
        h_list = cache['h_list']
        seq_len = d_y_seq.shape[0]
        batch_size = d_y_seq.shape[2]
        
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros((self.hidden_size, batch_size))
        
        for t in reversed(range(seq_len)):
            # Output gradient
            dy = d_y_seq[t]  # (output_size, batch_size)
            h = h_list[t + 1]  # current hidden
            h_prev = h_list[t]  # previous hidden
            x = x_seq[t]         # current input
            
            # Gradients for output layer
            dWhy += dy @ h.T
            dby += dy
            
            # Backprop through output to hidden
            dh = self.Why.T @ dy + dh_next
            
            # Through tanh: derivative is (1 - h^2)
            dtanh = (1 - h * h) * dh
            
            # Gradients for recurrent weights
            dbh += np.sum(dtanh, axis=1, keepdims=True)
            dWxh += dtanh @ x.T
            dWhh += dtanh @ h_prev.T
            
            # Propagate to next time step
            dh_next = self.Whh.T @ dtanh
            
            # Check for NaN/Inf
            if np.any(np.isnan(dWxh)) or np.any(np.isinf(dWxh)):
                print(f"Warning: NaN/Inf detected at timestep {t}")
                break
        
        # Gradient clipping (prevent exploding gradients)
        for grad in [dWxh, dWhh, dbh, dWhy, dby]:
            norm = np.linalg.norm(grad)
            if norm > clip_value:
                grad *= clip_value / norm
        
        # Update with momentum
        momentum = 0.9
        lr = 0.01
        self._update_with_momentum(dWxh, dWhh, dbh, dWhy, dby, lr, momentum)
        
        return np.mean(d_y_seq ** 2)
    
    def _update_with_momentum(self, dWxh, dWhh, dbh, dWhy, dby, lr, momentum):
        """SGD with momentum."""
        self.vWxh = momentum * self.vWxh - lr * dWxh
        self.vWhh = momentum * self.vWhh - lr * dWhh
        self.vbh = momentum * self.vbh - lr * dbh
        self.vWhy = momentum * self.vWhy - lr * dWhy
        self.vby = momentum * self.vby - lr * dby
        
        self.Wxh += self.vWxh
        self.Whh += self.vWhh
        self.bh += self.vbh
        self.Why += self.vWhy
        self.by += self.vby
    
    def train(self, x_seq, y_seq, epochs=100, clip_value=5.0):
        """Train on sequence data."""
        losses = []
        for epoch in range(epochs):
            y_pred, _, cache = self.forward_sequence(x_seq)
            loss = np.mean((y_pred - y_seq) ** 2)
            losses.append(loss)
            
            d_y = 2 * (y_pred - y_seq) / y_pred.size
            self.backward(d_y, cache, clip_value)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        return losses


# =============================================================================
# DEMONSTRATION: Sine Wave Prediction
# =============================================================================

def generate_sine_sequence(seq_len=100, noise=0.05):
    """Generate sine wave data for prediction."""
    t = np.linspace(0, 4 * np.pi, seq_len)
    data = np.sin(t) + noise * np.random.randn(seq_len)
    return data.reshape(-1, 1)  # (seq_len, 1)


def demo_vanilla_rnn():
    """Demonstrate vanilla RNN on sine wave prediction."""
    print("=" * 60)
    print("VANILLA RNN (NumPy) — Sine Wave Prediction")
    print("=" * 60)
    print()
    print("Architecture: RNN Cell")
    print("  h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)")
    print("  y_t = W_hy * h_t + b_y")
    print()
    print("Task: Given previous values, predict next sine wave value")
    print()
    
    # Generate data
    np.random.seed(42)
    data = generate_sine_sequence(100)
    
    # Build training pairs: use window of 10 values to predict next
    window = 10
    X_list = []
    Y_list = []
    for i in range(len(data) - window):
        X_list.append(data[i:i + window])
        Y_list.append(data[i + window])
    
    print(f"Training samples: {len(X_list)}")
    print(f"Window size: {window}")
    print()
    
    # Train model
    input_size = 1
    hidden_size = 32
    output_size = 1
    
    rnn = VanillaRNNCell(input_size, hidden_size, output_size)
    
    print("Training...")
    losses = rnn.train(
        np.array([X_list[0]]),  # placeholder
        np.array([Y_list[0]]),  # placeholder
        epochs=1  # just to show init
    )
    
    # Do proper training with individual windows
    total_loss = 0
    for i in range(len(X_list)):
        x_seq = X_list[i].reshape(window, 1, 1)  # (seq_len, input_size, batch_size)
        y_seq = Y_list[i].reshape(1, 1, 1)        # (seq_len, output_size, batch_size)
        _, _, cache = rnn.forward_sequence(x_seq)
        loss = np.mean((rnn.forward_sequence(x_seq)[0] - y_seq) ** 2)
        d_y = 2 * (rnn.forward_sequence(x_seq)[0] - y_seq) / rnn.forward_sequence(x_seq)[0].size
        rnn.backward(d_y, cache)
        total_loss += loss
    
    avg_loss = total_loss / len(X_list)
    print(f"Average training loss: {avg_loss:.6f}")
    
    # Demo forward pass
    print("\nDemo - Predicting from first window:")
    test_input = X_list[0].reshape(window, 1, 1)
    y_pred, h, _ = rnn.forward_sequence(test_input)
    print(f"  Input:  {X_list[0].flatten().round(3).tolist()}")
    print(f"  Target: {Y_list[0].round(3).tolist()}")
    print(f"  Output: {y_pred[-1, 0, 0].round(3).tolist()}")


# =============================================================================
# DEMONSTRATION: BPTT GRADIENT FLOW
# =============================================================================

def demonstrate_bptt():
    """Demonstrate how gradients flow through time."""
    print("\n" + "=" * 60)
    print("BPTT GRADIENT FLOW ANALYSIS")
    print("=" * 60)
    print()
    
    hidden_size = 8
    seq_len = 20
    
    # Simulate gradient magnitudes over time
    np.random.seed(42)
    Whh = np.random.randn(hidden_size, hidden_size) * 0.5
    
    print("Vanilla RNN with Whh spectral radius < 1:")
    print(f"  Spectral radius: {np.linalg.norm(Whh):.4f}")
    
    grad_mag = 1.0
    grad_mags = [grad_mag]
    for t in range(seq_len):
        grad_mag *= np.linalg.norm(Whh)
        grad_mags.append(grad_mag)
    
    print(f"  Gradient magnitude after {seq_len} steps: {grad_mags[-1]:.6e}")
    verdict = "vanish" if grad_mags[-1] < 1e-6 else "stable"
    print(f"  -> Gradients {verdict} over time")
    
    print()
    
    # Exploding gradient example
    Whh_exp = np.random.randn(hidden_size, hidden_size) * 1.5
    grad_mag = 1.0
    grad_mags_exp = [grad_mag]
    for t in range(seq_len):
        grad_mag *= np.linalg.norm(Whh_exp)
        grad_mags_exp.append(grad_mag)
    
    print("Vanilla RNN with Whh spectral radius > 1:")
    print(f"  Spectral radius: {np.linalg.norm(Whh_exp):.4f}")
    print(f"  Gradient magnitude after {seq_len} steps: {grad_mags_exp[-1]:.6e}")
    print(f"  -> Gradients EXPLODE over time")
    print(f"  -> Solution: Gradient clipping (see ext23)")


# =============================================================================
# RNN ARCHITECTURE COMPARISON
# =============================================================================

def compare_architectures():
    """Compare RNN architectures: Many-to-Many, One-to-Many, Many-to-One."""
    print("\n" + "=" * 60)
    print("RNN ARCHITECTURE COMPARISON")
    print("=" * 60)
    print()
    print("""
    Many-to-Many (Sequence → Sequence):
      Input:  x_1, x_2, ..., x_T  (T time steps)
      Output: y_1, y_2, ..., y_T  (T time steps)
      Use: POS Tagging, NER, Machine Translation
      
    One-to-Many (Single → Sequence):
      Input:  x                     (1 input)
      Output: y_1, y_2, ..., y_T   (T time steps)
      Use: Image Captioning, Music Gen, Text Gen
      
    Many-to-One (Sequence → Single):
      Input:  x_1, x_2, ..., x_T    (T time steps)
      Output: y                      (1 output)
      Use: Sentiment Analysis, Speech Rec, Anomaly Detection
    """)


if __name__ == "__main__":
    demo_vanilla_rnn()
    demonstrate_bptt()
    compare_architectures()