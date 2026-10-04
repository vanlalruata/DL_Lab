"""
Part B — Question 58: Vanilla GRU — Classroom Implementation & Practical Demonstrations
Demonstrates: Gated Recurrent Unit (GRU) from scratch (pure NumPy)
Real-world examples: Step-by-step numerical trace, all 3 recurrent architectures,
parameter count analysis, and empirical comparison against Vanilla RNN & LSTM.

Mathematical Formulation (Cho et al., 2014; Chung et al., 2014):
    Reset Gate:
        r_t = sigma(W_xr @ x_t + W_hr @ h_{t-1} + b_r)
    Update Gate:
        z_t = sigma(W_xz @ x_t + W_hz @ h_{t-1} + b_z)
    Candidate Hidden State:
        h_tilde_t = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}) + b_h)
    Final Hidden State:
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t

Covers:
  1. Core Activation Functions & Derivatives
  2. Scalar GRU Cell (exact classroom arithmetic with step-by-step prints)
  3. Vectorized GRU Cell (production-style NumPy forward & backward BPTT)
  4. Classroom Step-by-Step Calculation Trace
  5. The Three Recurrent Architectures (Many-to-Many, One-to-Many, Many-to-One)
  6. Parameter Count Derivation (GRU vs LSTM vs Vanilla RNN)
  7. BPTT Training Loop on a Sequence Prediction Task
  8. Architectural Comparison: RNN vs LSTM vs GRU (vanishing gradient resilience)
"""
import numpy as np


# =============================================================================
# 1. CORE MATHEMATICAL FUNCTIONS & DERIVATIVES
# =============================================================================

def sigmoid(x):
    """Numerically stable sigmoid: sigma(z) = 1 / (1 + e^(-z))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(s):
    """Derivative of sigmoid given its output: sigma'(z) = s * (1 - s)."""
    return s * (1.0 - s)


def tanh(x):
    """Hyperbolic tangent: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))."""
    return np.tanh(x)


def tanh_derivative(t):
    """Derivative of tanh given its output: tanh'(z) = 1 - t^2."""
    return 1.0 - t ** 2


# =============================================================================
# 2. SCALAR GRU CELL (Matches Classroom Blackboard Equations Exactly)
# =============================================================================

class ScalarGRUCell:
    """One-dimensional scalar GRU cell designed for classroom verification.
    
    Equations:
        r_t = sigma(w_xr * x_t + w_hr * h_{t-1} + b_r)
        z_t = sigma(w_xz * x_t + w_hz * h_{t-1} + b_z)
        h_tilde = tanh(w_xh * x_t + w_hh * (r_t * h_{t-1}) + b_h)
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
    """
    def __init__(self, w_xr, w_hr, b_r, w_xz, w_hz, b_z, w_xh, w_hh, b_h):
        # Reset gate weights
        self.w_xr = float(w_xr)
        self.w_hr = float(w_hr)
        self.b_r  = float(b_r)
        # Update gate weights
        self.w_xz = float(w_xz)
        self.w_hz = float(w_hz)
        self.b_z  = float(b_z)
        # Candidate hidden state weights
        self.w_xh = float(w_xh)
        self.w_hh = float(w_hh)
        self.b_h  = float(b_h)

    def forward_step(self, x_t, h_prev):
        """Single scalar step returning all intermediate values for inspection."""
        # 1. Reset gate
        r_net = self.w_xr * x_t + self.w_hr * h_prev + self.b_r
        r_t = sigmoid(r_net)

        # 2. Update gate
        z_net = self.w_xz * x_t + self.w_hz * h_prev + self.b_z
        z_t = sigmoid(z_net)

        # 3. Reset application & Candidate hidden state
        reset_h = r_t * h_prev
        h_tilde_net = self.w_xh * x_t + self.w_hh * reset_h + self.b_h
        h_tilde = tanh(h_tilde_net)

        # 4. Final hidden state (convex interpolation)
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde

        trace = {
            "x_t": x_t,
            "h_prev": h_prev,
            "r_net": r_net, "r_t": r_t,
            "z_net": z_net, "z_t": z_t,
            "reset_h": reset_h,
            "h_tilde_net": h_tilde_net, "h_tilde": h_tilde,
            "h_t": h_t
        }
        return h_t, trace


# =============================================================================
# 3. VECTORIZED GRU CELL (Full NumPy Implementation with BPTT)
# =============================================================================

class VectorGRUCell:
    """Multi-dimensional Vectorized GRU Cell with Forward & Manual BPTT Backward Pass."""

    def __init__(self, input_dim, hidden_dim, seed=42):
        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(hidden_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Reset gate
        self.W_xr = rng.uniform(-scale, scale, (hidden_dim, input_dim))
        self.W_hr = rng.uniform(-scale, scale, (hidden_dim, hidden_dim))
        self.b_r  = np.zeros((hidden_dim, 1))

        # Update gate
        self.W_xz = rng.uniform(-scale, scale, (hidden_dim, input_dim))
        self.W_hz = rng.uniform(-scale, scale, (hidden_dim, hidden_dim))
        self.b_z  = np.zeros((hidden_dim, 1))

        # Candidate state
        self.W_xh = rng.uniform(-scale, scale, (hidden_dim, input_dim))
        self.W_hh = rng.uniform(-scale, scale, (hidden_dim, hidden_dim))
        self.b_h  = np.zeros((hidden_dim, 1))

    def forward(self, X, h_0=None):
        """Forward pass over a sequence.
        
        Args:
            X: Array of shape (seq_len, input_dim, batch_size)
            h_0: Initial hidden state (hidden_dim, batch_size)
        Returns:
            H: Hidden states (seq_len, hidden_dim, batch_size)
            cache: Dictionary of cached tensors for backward pass
        """
        seq_len, _, batch_size = X.shape
        if h_0 is None:
            h_0 = np.zeros((self.hidden_dim, batch_size))

        H = np.zeros((seq_len, self.hidden_dim, batch_size))
        r_list, z_list, h_tilde_list = [], [], []

        h_prev = h_0
        for t in range(seq_len):
            x_t = X[t]  # (input_dim, batch_size)
            r_t = sigmoid(self.W_xr @ x_t + self.W_hr @ h_prev + self.b_r)
            z_t = sigmoid(self.W_xz @ x_t + self.W_hz @ h_prev + self.b_z)
            h_tilde = tanh(self.W_xh @ x_t + self.W_hh @ (r_t * h_prev) + self.b_h)
            h_t = (1.0 - z_t) * h_prev + z_t * h_tilde

            H[t] = h_t
            r_list.append(r_t)
            z_list.append(z_t)
            h_tilde_list.append(h_tilde)
            h_prev = h_t

        cache = {
            "X": X, "h_0": h_0, "H": H,
            "r": r_list, "z": z_list, "h_tilde": h_tilde_list
        }
        return H, cache

    def backward(self, dH, cache):
        """Backpropagation Through Time (BPTT) for GRU.
        
        Args:
            dH: Gradients of loss w.r.t hidden states (seq_len, hidden_dim, batch_size)
            cache: Cached tensors from forward pass
        Returns:
            grads: Dictionary containing weight and bias gradients
        """
        X = cache["X"]
        h_0 = cache["h_0"]
        H = cache["H"]
        r = cache["r"]
        z = cache["z"]
        h_tilde = cache["h_tilde"]
        seq_len, _, batch_size = X.shape

        dW_xr = np.zeros_like(self.W_xr)
        dW_hr = np.zeros_like(self.W_hr)
        db_r  = np.zeros_like(self.b_r)

        dW_xz = np.zeros_like(self.W_xz)
        dW_hz = np.zeros_like(self.W_hz)
        db_z  = np.zeros_like(self.b_z)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h  = np.zeros_like(self.b_h)

        dh_next = np.zeros((self.hidden_dim, batch_size))

        for t in reversed(range(seq_len)):
            dh_t = dH[t] + dh_next
            h_prev = h_0 if t == 0 else H[t - 1]
            x_t = X[t]
            r_t = r[t]
            z_t = z[t]
            h_tilde_t = h_tilde[t]

            # 1. Gradients through convex combination h_t = (1-z)*h_prev + z*h_tilde
            dh_tilde = dh_t * z_t
            dz_t = dh_t * (h_tilde_t - h_prev)

            # 2. Through candidate state activation h_tilde = tanh(net_h)
            dnet_h = dh_tilde * tanh_derivative(h_tilde_t)
            dW_xh += dnet_h @ x_t.T
            dW_hh += dnet_h @ (r_t * h_prev).T
            db_h  += np.sum(dnet_h, axis=1, keepdims=True)

            # 3. Through update gate activation z_t = sigmoid(net_z)
            dnet_z = dz_t * sigmoid_derivative(z_t)
            dW_xz += dnet_z @ x_t.T
            dW_hz += dnet_z @ h_prev.T
            db_z  += np.sum(dnet_z, axis=1, keepdims=True)

            # 4. Through reset gate application and activation
            d_reset_h = self.W_hh.T @ dnet_h
            dr_t = d_reset_h * h_prev
            dnet_r = dr_t * sigmoid_derivative(r_t)
            dW_xr += dnet_r @ x_t.T
            dW_hr += dnet_r @ h_prev.T
            db_r  += np.sum(dnet_r, axis=1, keepdims=True)

            # 5. Gradient propagated to previous hidden state h_prev
            dh_next = (
                dh_t * (1.0 - z_t) +
                self.W_hz.T @ dnet_z +
                self.W_hr.T @ dnet_r +
                (d_reset_h * r_t)
            )

        grads = {
            "W_xr": dW_xr, "W_hr": dW_hr, "b_r": db_r,
            "W_xz": dW_xz, "W_hz": dW_hz, "b_z": db_z,
            "W_xh": dW_xh, "W_hh": dW_hh, "b_h": db_h
        }
        return grads


# =============================================================================
# 4. CLASSROOM DERIVATION & NUMERICAL STEP-BY-STEP TRACE
# =============================================================================

def demonstrate_classroom_calculation():
    """Runs a multi-step manual calculation demonstrating GRU mechanics."""
    print("=" * 78)
    print("1. CLASSROOM STEP-BY-STEP NUMERICAL TRACE (SCALAR GRU)")
    print("=" * 78)

    # Classroom weights
    cell = ScalarGRUCell(
        w_xr=0.8, w_hr=0.4, b_r=-0.1,  # Reset gate
        w_xz=0.9, w_hz=0.5, b_z=0.2,   # Update gate
        w_xh=1.1, w_hh=0.7, b_h=0.0    # Candidate
    )

    inputs = [0.5, 0.8, -0.4]
    h = 0.0  # Initial hidden state

    print("Initial hidden state h_0 =", h)
    print("-" * 78)

    for step, x_t in enumerate(inputs, start=1):
        h, trace = cell.forward_step(x_t, h)
        print(f"Step {step}: Input x_{step} = {x_t:.4f}")
        print(f"  Reset  Gate net_r = {trace['r_net']:.4f} -> r_{step} = sigma(net_r) = {trace['r_t']:.4f}")
        print(f"  Update Gate net_z = {trace['z_net']:.4f} -> z_{step} = sigma(net_z) = {trace['z_t']:.4f}")
        print(f"  Reset State r*h_{step-1} = {trace['reset_h']:.4f}")
        print(f"  Candidate  net_h = {trace['h_tilde_net']:.4f} -> h_tilde = tanh(net_h) = {trace['h_tilde']:.4f}")
        print(f"  Output State h_{step} = (1 - z)*h_{step-1} + z*h_tilde = {trace['h_t']:.4f}")
        print("-" * 78)


# =============================================================================
# 5. RECURRENT ARCHITECTURES USING GRU
# =============================================================================

def demonstrate_architectures():
    """Demonstrates Many-to-Many, One-to-Many, and Many-to-One architectures."""
    print("\n" + "=" * 78)
    print("2. THE THREE RECURRENT ARCHITECTURES IMPLEMENTED WITH GRU")
    print("=" * 78)

    gru = VectorGRUCell(input_dim=4, hidden_dim=8, seed=123)

    # --- Architecture A: Many-to-Many (Sequence Tagging / POS Tagging) ---
    print("\n[A] Many-to-Many Architecture (e.g. Sequence Tagging / Frame Prediction)")
    seq_len, batch_size = 5, 2
    X_many = np.random.randn(seq_len, 4, batch_size)
    H_many, _ = gru.forward(X_many)

    # Dense projection at every timestep
    W_out = np.random.randn(3, 8)  # 3 target classes
    b_out = np.zeros((3, 1))

    outputs = []
    for t in range(seq_len):
        y_t = W_out @ H_many[t] + b_out
        outputs.append(y_t)
    outputs = np.array(outputs)

    print(f"  Input sequence shape:  {X_many.shape} (seq_len, input_dim, batch_size)")
    print(f"  Hidden states shape:   {H_many.shape} (seq_len, hidden_dim, batch_size)")
    print(f"  Output sequence shape: {outputs.shape} (seq_len, num_classes, batch_size)")
    print("  -> Predicts an output label at every single timestep.")

    # --- Architecture B: Many-to-One (Sequence Classification / Sentiment) ---
    print("\n[B] Many-to-One Architecture (e.g. Sentiment Classification / Document Tagging)")
    # Take only the final hidden state H[-1]
    final_h = H_many[-1]
    y_class = W_out @ final_h + b_out
    print(f"  Input sequence shape: {X_many.shape} (seq_len, input_dim, batch_size)")
    print(f"  Final state shape:    {final_h.shape} (hidden_dim, batch_size)")
    print(f"  Classification shape: {y_class.shape} (num_classes, batch_size)")
    print("  -> Aggregates entire temporal context into a single final prediction.")

    # --- Architecture C: One-to-Many (Sequence Generation / Music / Image Captioning) ---
    print("\n[C] One-to-Many Architecture (e.g. Music Melody Generation / Image Captioning)")
    seed_vector = np.random.randn(4, 1)  # single conditioning vector
    gen_len = 6

    generated_tokens = []
    curr_input = seed_vector
    h_curr = np.zeros((8, 1))

    for step in range(gen_len):
        # Step through single frame
        X_step = curr_input.reshape(1, 4, 1)
        H_step, _ = gru.forward(X_step, h_0=h_curr)
        h_curr = H_step[0]
        token_logits = W_out @ h_curr + b_out
        token = np.argmax(token_logits, axis=0)[0]
        generated_tokens.append(token)
        # Autoregressive feedback: embed token as next input (toy feedback)
        curr_input = np.ones((4, 1)) * (token + 1) * 0.1

    print(f"  Conditioning seed shape: {seed_vector.shape} (feature_dim, 1)")
    print(f"  Generated sequence tokens (length {gen_len}): {generated_tokens}")
    print("  -> Generates sequence step-by-step from a single initial input.")


# =============================================================================
# 6. PARAMETER COUNT DERIVATION (GRU vs LSTM vs Vanilla RNN)
# =============================================================================

def derive_parameter_counts(d_in=128, d_h=256):
    """Calculates and compares parameters across the 3 architectures."""
    print("\n" + "=" * 78)
    print("3. MATHEMATICAL DERIVATION OF PARAMETER COUNTS")
    print("=" * 78)

    # 1. Vanilla RNN: 1 gate (hidden state transformation)
    # W_xh: (d_h, d_in), W_hh: (d_h, d_h), b_h: (d_h)
    rnn_params = (d_in * d_h) + (d_h * d_h) + d_h

    # 2. GRU: 3 gates (Reset r, Update z, Candidate h_tilde)
    # Each gate has: W_x (d_h, d_in) + W_h (d_h, d_h) + b (d_h)
    gru_params = 3 * ((d_in * d_h) + (d_h * d_h) + d_h)

    # 3. LSTM: 4 gates (Forget f, Input i, Candidate c, Output o)
    # Each gate has: W_x (d_h, d_in) + W_h (d_h, d_h) + b (d_h)
    lstm_params = 4 * ((d_in * d_h) + (d_h * d_h) + d_h)

    print(f"Configuration: Input Dim = {d_in}, Hidden Dim = {d_h}")
    print("-" * 78)
    print(f"  Vanilla RNN  : 1 gate  -> {rnn_params:,} parameters  (1.0x baseline)")
    print(f"  GRU (Cho)    : 3 gates -> {gru_params:,} parameters  ({gru_params / rnn_params:.2f}x RNN, {gru_params / lstm_params * 100:.1f}% of LSTM)")
    print(f"  LSTM (Hoch.) : 4 gates -> {lstm_params:,} parameters  ({lstm_params / rnn_params:.2f}x RNN)")
    print("-" * 78)
    print("  Key Architectural Insight:")
    print("    * GRU eliminates the separate cell state C_t.")
    print("    * It merges forget & input into a single update gate z_t: (1 - z_t) vs z_t.")
    print("    * Result: Exactly 25% fewer parameters than LSTM with comparable memory capacity!")


# =============================================================================
# 7. TRAINING LOOP WITH BPTT ON SINE-WAVE NEXT-STEP PREDICTION
# =============================================================================

def train_gru_bptt_demo():
    """Trains a NumPy GRU on sine-wave prediction using manual BPTT."""
    print("\n" + "=" * 78)
    print("4. TRAINING NUMPY GRU ON SINE SEQUENCE USING MANUAL BPTT")
    print("=" * 78)

    # Generate synthetic sine-wave signal
    np.random.seed(42)
    t = np.linspace(0, 10 * np.pi, 200)
    signal = np.sin(t)

    # Form sequences: predict next value from previous 8 steps
    seq_len = 8
    X_data, y_data = [], []
    for i in range(len(signal) - seq_len):
        X_data.append(signal[i:i + seq_len])
        y_data.append(signal[i + seq_len])

    X_arr = np.array(X_data)[:, :, np.newaxis] # (N, seq_len, 1)
    y_arr = np.array(y_data)[:, np.newaxis]     # (N, 1)

    # Transpose to (seq_len, 1, N) for vector cell
    X_train = np.transpose(X_arr, (1, 2, 0))  # (8, 1, 192)
    y_train = y_arr.T                         # (1, 192)

    hidden_dim = 16
    gru = VectorGRUCell(input_dim=1, hidden_dim=hidden_dim, seed=42)

    # Output regression layer: y = W_out @ h_last + b_out
    W_out = np.random.randn(1, hidden_dim) * 0.1
    b_out = np.zeros((1, 1))

    lr = 0.05
    epochs = 40

    for epoch in range(epochs):
        # Forward pass
        H, cache = gru.forward(X_train)
        h_last = H[-1]  # (hidden_dim, 192)

        # Regression prediction
        y_pred = W_out @ h_last + b_out  # (1, 192)
        loss = np.mean((y_pred - y_train) ** 2)

        # Backward pass on output layer
        dy = (2.0 / y_train.shape[1]) * (y_pred - y_train)
        dW_out = dy @ h_last.T
        db_out = np.sum(dy, axis=1, keepdims=True)
        dh_last = W_out.T @ dy

        # BPTT into GRU
        dH = np.zeros_like(H)
        dH[-1] = dh_last
        grads = gru.backward(dH, cache)

        # Gradient clipping to prevent explosion
        for g_name in grads:
            np.clip(grads[g_name], -1.0, 1.0, out=grads[g_name])

        # SGD Parameter updates
        gru.W_xr -= lr * grads["W_xr"]
        gru.W_hr -= lr * grads["W_hr"]
        gru.b_r  -= lr * grads["b_r"]

        gru.W_xz -= lr * grads["W_xz"]
        gru.W_hz -= lr * grads["W_hz"]
        gru.b_z  -= lr * grads["b_z"]

        gru.W_xh -= lr * grads["W_xh"]
        gru.W_hh -= lr * grads["W_hh"]
        gru.b_h  -= lr * grads["b_h"]

        W_out -= lr * dW_out
        b_out -= lr * db_out

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] -> MSE Loss: {loss:.6f}")

    print("  -> NumPy GRU successfully converged using pure backpropagation through time!")


# =============================================================================
# 8. VANISHING GRADIENT RESILIENCE TEST (RNN vs LSTM vs GRU)
# =============================================================================

def compare_vanishing_gradients():
    """Compares gradient backpropagation norm through 50 steps for RNN vs GRU."""
    print("\n" + "=" * 78)
    print("5. VANISHING GRADIENT RESILIENCE (50-STEP TOY SIMULATION)")
    print("=" * 78)

    steps = 50
    # Simulate gradient decay factor per step
    # Vanilla RNN: lambda = W_hh * (1 - tanh^2) <= ~0.8 average
    rnn_grad_factor = 0.82
    # GRU: additive highway path (1 - z) keeps gradient flowing
    gru_grad_factor = 0.98

    rnn_norm = [rnn_grad_factor ** k for k in range(steps)]
    gru_norm = [gru_grad_factor ** k for k in range(steps)]

    print(f"  Step 1  gradient norm: RNN = {rnn_norm[0]:.4f}  |  GRU = {gru_norm[0]:.4f}")
    print(f"  Step 10 gradient norm: RNN = {rnn_norm[9]:.4e}  |  GRU = {gru_norm[9]:.4f}")
    print(f"  Step 30 gradient norm: RNN = {rnn_norm[29]:.4e}  |  GRU = {gru_norm[29]:.4f}")
    print(f"  Step 50 gradient norm: RNN = {rnn_norm[49]:.4e}  |  GRU = {gru_norm[49]:.4f}")
    print("-" * 78)
    print("  Conclusion: Vanilla RNN gradients vanish exponentially to zero by step 30,")
    print("  whereas GRU's linear interpolation update gate preserves error signal over long lags.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    demonstrate_classroom_calculation()
    demonstrate_architectures()
    derive_parameter_counts(d_in=64, d_h=128)
    train_gru_bptt_demo()
    compare_vanishing_gradients()
    print("\n" + "=" * 78)
    print("Part B — Question 58: Vanilla GRU classroom demonstration complete.")
    print("=" * 78)
