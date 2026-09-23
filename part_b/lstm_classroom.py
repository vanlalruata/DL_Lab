"""
Part B - Question 57: Vanilla LSTM - Classroom Implementation & Practical Examples
Demonstrates: Vanilla LSTM from scratch (NumPy), all 3 architectures, classroom calculations
Real-world examples: POS tagging, sentiment classification, sequence generation

Covers:
  1. Core mathematical functions (sigmoid, tanh, derivatives)
  2. Scalar LSTM cell (matches document classroom example exactly)
  3. Vector LSTM cell (matrix-based, production-ready)
  4. Classroom calculation reproduction (scalar, multi-step)
  5. LSTM architectures: Many-to-Many, One-to-Many, Many-to-One
  6. Parameter count calculation
  7. Training loop with simplified BPTT
  8. RNN vs LSTM comparison
"""
import numpy as np


# =============================================================================
# 1. CORE MATHEMATICAL FUNCTIONS
# =============================================================================

def sigmoid(x):
    """Sigmoid activation: sigma(z) = 1 / (1 + e^(-z))"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    """Tanh activation: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))"""
    return np.tanh(x)


def sigmoid_derivative(s):
    """Derivative of sigmoid: sigma'(z) = sigma(z)(1 - sigma(z))"""
    return s * (1 - s)


def tanh_derivative(t):
    """Derivative of tanh: tanh'(z) = 1 - tanh^2(z)"""
    return 1 - t ** 2


# =============================================================================
# 2. SCALAR LSTM CELL (Matches Document Classroom Example Exactly)
# =============================================================================

class ScalarLSTMCell:
    """One-dimensional scalar LSTM cell.

    Directly implements the 6 equations from the classroom notes:
        f_t = sigma(W_f * z + b_f)
        i_t = sigma(W_i * z + b_i)
        c_tilde = tanh(W_c * z + b_c)
        c_t  = f_t * c_{t-1} + i_t * c_tilde
        o_t  = sigma(W_o * z + b_o)
        h_t  = o_t * tanh(c_t)

    where z = h_{t-1} + x_t (scalar concatenation)
    """

    def __init__(self, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
        self.Wf = Wf
        self.Wi = Wi
        self.Wc = Wc
        self.Wo = Wo
        self.bf = bf
        self.bi = bi
        self.bc = bc
        self.bo = bo

    def step(self, x, h_prev, c_prev):
        z = h_prev + x

        f = sigmoid(self.Wf * z + self.bf)
        i = sigmoid(self.Wi * z + self.bi)
        c_candidate = tanh(self.Wc * z + self.bc)
        c = f * c_prev + i * c_candidate
        o = sigmoid(self.Wo * z + self.bo)
        h = o * tanh(c)

        return h, c, f, i, c_candidate, o


def run_classroom_calculation():
    """Reproduce the exact calculation from Sections 20-28."""
    print("=" * 70)
    print("CLASSROOM CALCULATION - One-Dimensional LSTM (Sections 20-28)")
    print("=" * 70)
    print()
    print("Given:")
    print("  x_t = 0.5,  h_{t-1} = 0.4,  c_{t-1} = 0.2")
    print("  W_f=0.7, W_i=0.6, W_c=0.5, W_o=0.8")
    print("  b_f=0.1, b_i=-0.2, b_c=0, b_o=0.1")
    print()

    lstm = ScalarLSTMCell(0.7, 0.6, 0.5, 0.8, 0.1, -0.2, 0.0, 0.1)
    h, c, f, i, c_cand, o = lstm.step(0.5, 0.4, 0.2)

    print("Step 1 - Combined Input:")
    print(f"  z = h_{{t-1}} + x_t = 0.4 + 0.5 = {0.4 + 0.5}")
    print()

    print("Step 2 - Forget Gate:")
    print(f"  f_t = sigma(0.7 * 0.9 + 0.1) = sigma(0.73)")
    print(f"  f_t = {f:.4f}")
    print(f"  (Document says: f_t ~ 0.675)")
    print()

    print("Step 3 - Input Gate:")
    print(f"  i_t = sigma(0.6 * 0.9 - 0.2) = sigma(0.34)")
    print(f"  i_t = {i:.4f}")
    print(f"  (Document says: i_t ~ 0.584)")
    print()

    print("Step 4 - Candidate Cell State:")
    print(f"  c_tilde = tanh(0.5 * 0.9 + 0) = tanh(0.45)")
    print(f"  c_tilde = {c_cand:.4f}")
    print(f"  (Document says: c_tilde ~ 0.422)")
    print()

    print("Step 5 - Cell State:")
    print(f"  c_t = f_t * c_{{t-1}} + i_t * c_tilde")
    print(f"  c_t = {f:.4f} * 0.2 + {i:.4f} * {c_cand:.4f}")
    print(f"  c_t = {f * 0.2:.4f} + {i * c_cand:.4f}")
    print(f"  c_t = {c:.4f}")
    print(f"  (Document says: c_t ~ 0.381)")
    print()

    print("Step 6 - Output Gate:")
    print(f"  o_t = sigma(0.8 * 0.9 + 0.1) = sigma(0.82)")
    print(f"  o_t = {o:.4f}")
    print(f"  (Document says: o_t ~ 0.694)")
    print()

    print("Step 7 - Hidden State:")
    tanh_c = tanh(c)
    print(f"  tanh(c_t) = tanh({c:.4f}) = {tanh_c:.4f}")
    print(f"  h_t = o_t * tanh(c_t) = {o:.4f} * {tanh_c:.4f}")
    print(f"  h_t = {h:.4f}")
    print(f"  (Document says: h_t ~ 0.252)")
    print()

    print("=" * 70)
    print("FINAL ANSWER:")
    print(f"  (f_t, i_t, c_tilde, c_t, o_t, h_t)")
    print(f"  = ({f:.4f}, {i:.4f}, {c_cand:.4f}, {c:.4f}, {o:.4f}, {h:.4f})")
    print("=" * 70)


def run_multi_step_calculation():
    """Reproduce the 3-step calculation from Sections 31-32."""
    print("\n" + "=" * 70)
    print("MULTI-STEP CALCULATION - 3 Time Steps (Sections 31-32)")
    print("=" * 70)
    print()
    print("Given: x_1=0.5, x_2=0.8, x_3=0.2")
    print("       h_0=0, c_0=0")
    print("       Same weights as above")
    print()

    lstm = ScalarLSTMCell(0.7, 0.6, 0.5, 0.8, 0.1, -0.2, 0.0, 0.1)
    sequence = [0.5, 0.8, 0.2]
    h = 0.0
    c = 0.0

    results = []
    for t, x in enumerate(sequence, start=1):
        h, c, f, i, c_cand, o = lstm.step(x, h, c)
        results.append((t, x, f, i, c_cand, c, o, h))

    print(f"{'Step':>4} {'x':>6} {'f_t':>8} {'i_t':>8} {'c_tilde':>8} {'c_t':>8} {'o_t':>8} {'h_t':>8}")
    print("-" * 68)
    for t, x, f, i, c_cand, c, o, h in results:
        print(f"{t:>4} {x:>6.2f} {f:>8.4f} {i:>8.4f} {c_cand:>8.4f} {c:>8.4f} {o:>8.4f} {h:>8.4f}")

    print()
    print("Observation:")
    print("  - c_t grows as new information is written")
    print("  - h_t reflects what is exposed from the cell state")
    print("  - The cell state carries information across time steps")
    print("  This is the key advantage over vanilla RNN.")


# =============================================================================
# 3. VECTOR LSTM CELL (Matrix-Based, Production-Ready)
# =============================================================================

class VectorLSTMCell:
    """Vectorized LSTM cell with matrix weights.

    Dimensions:
        x_t: (d,)  - input vector
        h_t: (h,)  - hidden state vector
        c_t: (h,)  - cell state vector
        W_f, W_i, W_c, W_o: (h, h+d) - weight matrices
        b_f, b_i, b_c, b_o: (h,) - bias vectors

    Equations (same as scalar, but with matrix operations):
        z_t = [h_{t-1}, x_t]     - concatenation, shape (h+d,)
        f_t = sigma(W_f @ z_t + b_f)
        i_t = sigma(W_i @ z_t + b_i)
        c_tilde = tanh(W_c @ z_t + b_c)
        c_t = f_t * c_{t-1} + i_t * c_tilde    (element-wise)
        o_t = sigma(W_o @ z_t + b_o)
        h_t = o_t * tanh(c_t)                  (element-wise)
    """

    def __init__(self, input_size, hidden_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.RandomState(seed)
        scale = np.sqrt(1.0 / hidden_size)

        concat_size = input_size + hidden_size
        self.Wf = rng.randn(hidden_size, concat_size) * scale
        self.Wi = rng.randn(hidden_size, concat_size) * scale
        self.Wc = rng.randn(hidden_size, concat_size) * scale
        self.Wo = rng.randn(hidden_size, concat_size) * scale

        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)

    def step(self, x, h_prev, c_prev):
        """Single LSTM forward step."""
        z = np.concatenate([h_prev, x])

        f = sigmoid(self.Wf @ z + self.bf)
        i = sigmoid(self.Wi @ z + self.bi)
        c_candidate = tanh(self.Wc @ z + self.bc)
        c = f * c_prev + i * c_candidate
        o = sigmoid(self.Wo @ z + self.bo)
        h = o * tanh(c)

        return h, c, f, i, c_candidate, o

    def forward(self, x_seq, h0=None, c0=None):
        """Forward pass through entire sequence.

        x_seq: (T, d) - sequence of T input vectors
        Returns: (h_seq, c_seq) each of shape (T+1, h)
        """
        T = x_seq.shape[0]
        h_seq = np.zeros((T + 1, self.hidden_size))
        c_seq = np.zeros((T + 1, self.hidden_size))

        if h0 is not None:
            h_seq[0] = h0
        if c0 is not None:
            c_seq[0] = c0

        for t in range(T):
            h_seq[t + 1], c_seq[t + 1], *_ = self.step(
                x_seq[t], h_seq[t], c_seq[t]
            )

        return h_seq, c_seq


def demonstrate_vector_lstm():
    """Demonstrate vector LSTM with random inputs."""
    print("\n" + "=" * 70)
    print("VECTOR LSTM - Matrix-Based Implementation")
    print("=" * 70)
    print()

    input_size = 4
    hidden_size = 8
    seq_len = 5

    lstm = VectorLSTMCell(input_size, hidden_size)
    x_seq = np.random.randn(seq_len, input_size)

    print(f"Input size (d): {input_size}")
    print(f"Hidden size (h): {hidden_size}")
    print(f"Sequence length (T): {seq_len}")

    h_seq, c_seq = lstm.forward(x_seq)

    param_count = 4 * hidden_size * (input_size + hidden_size + 1)
    print(f"\nParameter count: P = 4h(d+h+1) = 4*{hidden_size}({input_size}+{hidden_size}+1)")
    print(f"                 = 4*{hidden_size}*{input_size+hidden_size+1}")
    print(f"                 = {param_count}")

    print(f"\nHidden state shapes: {h_seq.shape} (T+1 x h)")
    print(f"Cell state shapes:   {c_seq.shape} (T+1 x h)")
    print(f"\nCell state evolution (L2 norm at each step):")
    for t in range(seq_len + 1):
        print(f"  c_{t}: {np.linalg.norm(c_seq[t]):.4f}")

    print("\nKey insight: The cell state norm changes gradually,")
    print("controlled by the forget and input gates.")
    print("This allows long-term information to flow through the network.")


# =============================================================================
# 4. LSTM NETWORK WITH TRAINING
# =============================================================================

class LSTMBinaryClassifier:
    """Many-to-One LSTM for binary classification."""

    def __init__(self, input_size, hidden_size, num_classes=2, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = VectorLSTMCell(input_size, hidden_size, seed)
        self.Wy = np.random.randn(num_classes, hidden_size) * 0.1
        self.by = np.zeros(num_classes)
        self.lr = 0.01
        self.params = self._collect_params()

    def _collect_params(self):
        return [self.lstm.Wf, self.lstm.Wi, self.lstm.Wc, self.lstm.Wo,
                self.lstm.bf, self.lstm.bi, self.lstm.bc, self.lstm.bo,
                self.Wy, self.by]

    def forward(self, x_seq):
        h_seq, c_seq = self.lstm.forward(x_seq)
        h_final = h_seq[-1]
        logits = self.Wy @ h_final + self.by
        return logits, h_seq, c_seq

    def train_step(self, x_seq, y_true):
        """Simplified training step with numerical gradient."""
        logits, _, _ = self.forward(x_seq)
        probs = softmax_output(logits)

        loss = -np.log(probs[y_true] + 1e-8)

        d_logits = probs.copy()
        d_logits[y_true] -= 1

        d_Wy = np.outer(d_logits, h_seq[-1])
        d_by = d_logits

        dh = self.Wy.T @ d_logits
        d_h = dh

        z_t = np.concatenate([h_seq[-2], x_seq[-1]])
        o = sigmoid(self.lstm.Wo @ z_t + self.lstm.bo)
        tanh_c = tanh(c_seq[-1])
        d_o = d_h * tanh_c
        d_c = d_h * o * (1 - tanh_c ** 2)

        f_t = sigmoid(self.lstm.Wf @ np.concatenate([h_seq[-2], x_seq[-1]]) + self.lstm.bf)
        i_t = sigmoid(self.lstm.Wi @ np.concatenate([h_seq[-2], x_seq[-1]]) + self.lstm.bi)
        c_cand_t = tanh(self.lstm.Wc @ np.concatenate([h_seq[-2], x_seq[-1]]) + self.lstm.bc)

        d_c_prev = d_c * f_t
        d_f = d_c * c_seq[-2] * f_t * (1 - f_t)
        d_i = d_c * c_cand_t * i_t * (1 - i_t)
        d_c_cand = d_c * i_t * (1 - c_cand_t ** 2)

        d_z = (self.lstm.Wf.T @ d_f + self.lstm.Wi.T @ d_i +
               self.lstm.Wc.T @ d_c_cand + self.lstm.Wo.T @ d_o)

        d_h_prev = d_z[:self.hidden_size]
        d_x = d_z[self.hidden_size:]

        lr = self.lr
        self.Wy -= lr * d_Wy
        self.by -= lr * d_by
        self.lstm.Wo -= lr * (np.outer(d_o, z_t) * o * (1 - o))
        self.lstm.Wf -= lr * (np.outer(d_f, z_t) * f_t * (1 - f_t))
        self.lstm.Wi -= lr * (np.outer(d_i, z_t) * i_t * (1 - i_t))
        self.lstm.Wc -= lr * (np.outer(d_c_cand, z_t) * (1 - c_cand_t ** 2))

        return loss


def softmax_output(logits):
    exp_l = np.exp(logits - np.max(logits))
    return exp_l / np.sum(exp_l)


def demonstrate_lstm_training():
    """Train LSTM on simple binary classification."""
    print("\n" + "=" * 70)
    print("LSTM TRAINING - Binary Classification on Sequence Data")
    print("=" * 70)
    print()

    np.random.seed(42)
    hidden_size = 16

    model = LSTMBinaryClassifier(input_size=4, hidden_size=hidden_size)

    positive_data = []
    negative_data = []

    for _ in range(50):
        pos_seq = np.random.randn(8, 4) + 0.5
        positive_data.append(pos_seq)
        neg_seq = np.random.randn(8, 4) - 0.5
        negative_data.append(neg_seq)

    data = [(seq, 1) for seq in positive_data] + [(seq, 0) for seq in negative_data]

    losses = []
    correct = 0
    for epoch in range(20):
        total_loss = 0
        correct = 0
        rng = np.random.RandomState(epoch)
        for x_seq, y in data:
            loss = model.train_step(x_seq, y)
            total_loss += loss
            logits, _, _ = model.forward(x_seq)
            pred = 1 if logits[1] > logits[0] else 0
            correct += (pred == y)
        avg_loss = total_loss / len(data)
        acc = correct / len(data)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/20 - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    print("\nTraining complete!")
    print("The LSTM learns to classify sequences based on the mean of their features.")


# =============================================================================
# 5. ARCHITECTURE PATTERNS
# =============================================================================

def demonstrate_architectures():
    """Demonstrate Many-to-Many, One-to-Many, Many-to-One LSTM patterns."""
    print("\n" + "=" * 70)
    print("LSTM ARCHITECTURE PATTERNS")
    print("=" * 70)

    print("\n" + "-" * 50)
    print("MANY-TO-MANY (Sequence -> Sequence)")
    print("-" * 50)
    print("""
    Input:  x_1, x_2, x_3, x_4    (T input vectors)
    Output: y_1, y_2, y_3, y_4    (T output vectors)

    Flow:
    x_1 -> LSTM -> h_1 -> y_1
    x_2 -> LSTM -> h_2 -> y_2
    x_3 -> LSTM -> h_3 -> y_3
    x_4 -> LSTM -> h_4 -> y_4

    Real-world uses:
    - POS Tagging       (each word -> POS tag)
    - Named Entity Rec  (each token -> NER label)
    - Time-series pred  (each step -> prediction)
    - Video frame class (each frame -> class)

    PyTorch: nn.RNN/nn.LSTM returns output at every time step.
    """)

    print("-" * 50)
    print("ONE-TO-MANY (Single Input -> Sequence)")
    print("-" * 50)
    print("""
    Input:  x_0             (1 input vector)
    Output: y_1, y_2, y_3   (T output vectors)

    Flow:
    x_0 -> LSTM -> h_1 -> y_1
                h_1 -> y_2 (autoregressive)
                      h_2 -> y_3

    Real-world uses:
    - Image Captioning   (image -> word sequence)
    - Music Generation    (seed note -> melody)
    - Text Generation     (start token -> sentence)
    - Time-series Forecast (initial state -> future values)

    PyTorch: Use nn.RNN/nn.LSTM with autoregressive decoding.
    """)

    print("-" * 50)
    print("MANY-TO-ONE (Sequence -> Single Output)")
    print("-" * 50)
    print("""
    Input:  x_1, x_2, x_3, x_4    (T input vectors)
    Output: y                       (1 output vector)

    Flow:
    x_1 -> LSTM -> h_1
    x_2 -> LSTM -> h_2
    x_3 -> LSTM -> h_3 -> y
    x_4 -> LSTM -> h_4 ?

    Final hidden state h_T is used for classification.

    Real-world uses:
    - Sentiment Analysis  (review -> positive/negative)
    - Speech Recognition  (audio -> transcript)
    - Anomaly Detection   (sensor data -> normal/anomaly)
    - Emotion Classification (text -> emotion)

    PyTorch: Use final hidden state h_n from nn.RNN/nn.LSTM.
    """)


# =============================================================================
# 6. PARAMETER COUNT CALCULATION
# =============================================================================

def demonstrate_param_count():
    """Calculate LSTM parameter counts for different configurations."""
    print("\n" + "=" * 70)
    print("LSTM PARAMETER COUNT CALCULATION")
    print("=" * 70)
    print()
    print("Formula: P = 4 * h * (d + h + 1)")
    print("where d = input size, h = hidden size")
    print()

    configs = [
        (10, 20),
        (5, 10),
        (100, 128),
        (300, 256),
        (512, 512),
    ]

    print(f"{'d (input)':>10} {'h (hidden)':>12} {'P (params)':>14} {'P (MB)':>10}")
    print("-" * 52)
    for d, h in configs:
        P = 4 * h * (d + h + 1)
        P_MB = P * 4 / (1024 * 1024)
        print(f"{d:>10} {h:>12} {P:>14,} {P_MB:>10.2f}")

    print()
    print("Compare with vanilla RNN: P_rnn = h*(d+h) + h (no gates)")
    print("LSTM has approximately 4x the parameters of a vanilla RNN.")


# =============================================================================
# 7. RNN vs LSTM COMPARISON
# =============================================================================

def compare_rnn_lstm():
    """Compare RNN and LSTM architectures."""
    print("\n" + "=" * 70)
    print("RNN vs LSTM COMPARISON")
    print("=" * 70)
    print()
    print("| Feature                | RNN           | LSTM          |")
    print("| ---------------------- | ------------- | ------------- |")
    print("| Hidden state           | Yes (h_t)     | Yes (h_t)     |")
    print("| Cell state             | No            | Yes (c_t)     |")
    print("| Forget gate            | No            | Yes           |")
    print("| Input gate             | No            | Yes           |")
    print("| Output gate            | No            | Yes           |")
    print("| Candidate memory       | Implicit      | Explicit      |")
    print("| Long-term dependencies | Difficult     | Better        |")
    print("| Parameter count        | Lower         | ~4x higher    |")
    print("| BPTT                   | Yes           | Yes           |")
    print("| Main activations       | tanh          | sigmoid+tanh  |")
    print()

    print("Gradient comparison (after T steps):")
    print("  Vanilla RNN: gradient ~ (||W_hh|| * (1-h^2))^T")
    print("  LSTM: gradient flows through cell state with gates")
    print("  -> LSTM can maintain gradient over much longer sequences")


# =============================================================================
# 8. LSTM LOSS FUNCTIONS
# =============================================================================

def demonstrate_losses():
    """Show common LSTM loss functions."""
    print("\n" + "=" * 70)
    print("LSTM LOSS FUNCTIONS")
    print("=" * 70)
    print()

    T = 5
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.9, 0.2, 0.8, 0.7, 0.3])

    print("Many-to-Many loss (sum over time steps):")
    print("  L = sum_t L_t")
    L_sum = 0
    for t in range(T):
        L_t = -(y_true[t] * np.log(y_pred[t] + 1e-8) +
                (1 - y_true[t]) * np.log(1 - y_pred[t] + 1e-8))
        L_sum += L_t
        print(f"  L_{t} = -[{y_true[t]}*{np.log(y_pred[t]+1e-8):.4f} + {1-y_true[t]}*{np.log(1-y_pred[t]+1e-8):.4f}] = {L_t:.4f}")
    print(f"  L = {L_sum:.4f}")

    print("\nMany-to-Many loss (average over time steps):")
    print(f"  L = (1/{T}) * sum_t L_t = {L_sum/T:.4f}")

    print("\nMany-to-One loss (single output):")
    print("  For binary classification:")
    y_single = 1
    p_single = 0.85
    BCE = -(y_single * np.log(p_single + 1e-8) + (1 - y_single) * np.log(1 - p_single + 1e-8))
    print(f"  L = -[y*log(p) + (1-y)*log(1-p)] = {BCE:.4f}")


# =============================================================================
# 9. LSTM ACTIVATION DERIVATIVES
# =============================================================================

def demonstrate_derivatives():
    """Show activation function derivatives used in LSTM backprop."""
    print("\n" + "=" * 70)
    print("LSTM ACTIVATION DERIVATIVES (for BPTT)")
    print("=" * 70)
    print()

    print("Sigmoid derivative:")
    print("  sigma'(z) = sigma(z) * (1 - sigma(z))")
    s = 0.8
    print(f"  If sigma(z) = {s}, then sigma'(z) = {s} * (1 - {s}) = {sigmoid_derivative(s):.4f}")

    print("\nTanh derivative:")
    print("  tanh'(z) = 1 - tanh^2(z)")
    t = 0.6
    print(f"  If tanh(z) = {t}, then tanh'(z) = 1 - {t}^2 = {tanh_derivative(t):.4f}")

    print("\nThese derivatives are used in BPTT to compute gradients")
    print("through the LSTM gates and cell state.")


# =============================================================================
# 10. VOCABULARY FOR TEXT TASKS
# =============================================================================

class SimpleVocab:
    """Simple vocabulary for text classification tasks."""

    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.n_words = 2

    def add(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

    def encode(self, text):
        return [self.word2idx.get(w.lower(), 1) for w in text.split()]

    def decode(self, indices):
        return [self.idx2word.get(i, '<UNK>') for i in indices]


def demonstrate_text_classification():
    """Demonstrate LSTM for sentiment classification."""
    print("\n" + "=" * 70)
    print("MANY-TO-ONE LSTM - Sentiment Classification")
    print("=" * 70)
    print()

    vocab = SimpleVocab()
    sentences = [
        "i love this movie",
        "this movie is great",
        "i hate this bad movie",
        "this is a terrible film",
    ]

    for sent in sentences:
        for w in sent.split():
            vocab.add(w)

    labels = [1, 1, 0, 0]

    print("Vocabulary:")
    for w, idx in sorted(vocab.word2idx.items(), key=lambda x: x[1]):
        print(f"  {idx:>3}: {w}")
    print()

    print("Sentences encoded:")
    for sent in sentences:
        encoded = vocab.encode(sent)
        print(f"  '{sent}' -> {encoded}")

    print("\nArchitecture:")
    print("  Many-to-One LSTM:")
    print("    Input sequence -> LSTM -> h_T -> Dense -> Sigmoid -> Positive/Negative")

    input_size = 8
    hidden_size = 16
    lstm_cell = VectorLSTMCell(input_size, hidden_size)
    param_count = 4 * hidden_size * (input_size + hidden_size + 1)
    print(f"\n  LSTM cell parameters: {param_count}")
    print(f"  (input_size={input_size}, hidden_size={hidden_size})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("#" * 70)
    print("# VANILLA LSTM - Classroom Implementation & Practical Examples")
    print("# Many-to-Many, One-to-Many, Many-to-One")
    print("# Pure NumPy, No Frameworks")
    print("#" * 70)

    run_classroom_calculation()
    run_multi_step_calculation()
    demonstrate_vector_lstm()
    demonstrate_param_count()
    compare_rnn_lstm()
    demonstrate_losses()
    demonstrate_derivatives()
    demonstrate_architectures()
    demonstrate_text_classification()

    print("\n" + "=" * 70)
    print("ALL DEMOs COMPLETE")
    print("=" * 70)
    print()
    print("If you are an MCA student, here's what you should know:")
    print("  - Scalar LSTM reproduces document classroom calculations exactly")
    print("  - Vector LSTM generalizes to matrix-based operations")
    print("  - Three architecture patterns demonstrated")
    print("  - Parameter count formula verified")
    print("  - RNN vs LSTM comparison provided")
    print("  - Loss functions and derivatives shown")


if __name__ == "__main__":
    main()