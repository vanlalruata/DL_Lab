"""
Question 25: Implement a GRU cell and compare its gate structure to LSTM.
Demonstrates: Pure NumPy GRU Cell from scratch, gate activation inspection,
and direct structural comparison with standard LSTM gating mechanisms.
"""
import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class GRUCell:
    """Gated Recurrent Unit (GRU) cell implemented in pure NumPy.
    
    Mathematical Formulation:
        Reset gate:       r_t = sigma(W_r @ [x_t, h_{t-1}] + b_r)
        Update gate:      z_t = sigma(W_z @ [x_t, h_{t-1}] + b_z)
        Candidate state:  h~_t = tanh(W_h @ [x_t, r_t * h_{t-1}] + b_h)
        Hidden state:     h_t = (1 - z_t) * h_{t-1} + z_t * h~_t
    """
    def __init__(self, in_features, hidden_size):
        self.in_features = in_features
        self.hidden_size = hidden_size
        scale = np.sqrt(1.0 / hidden_size)

        # Concatenated weights for [x_t, h_{t-1}]
        self.Wr = np.random.randn(hidden_size, in_features + hidden_size) * scale
        self.br = np.zeros(hidden_size)

        self.Wz = np.random.randn(hidden_size, in_features + hidden_size) * scale
        self.bz = np.zeros(hidden_size)

        # Candidate state weights (reset applied to h_{t-1})
        self.Wh = np.random.randn(hidden_size, in_features + hidden_size) * scale
        self.bh = np.zeros(hidden_size)

    def step(self, x, h_prev):
        """Processes one timestep, returning hidden state and gate values."""
        xh = np.concatenate([x, h_prev])
        
        # 1. Reset gate: decides how much past information to forget
        r = sigmoid(self.Wr @ xh + self.br)
        
        # 2. Update gate: balances new candidate vs previous state
        z = sigmoid(self.Wz @ xh + self.bz)
        
        # 3. Candidate hidden state using reset context
        x_reset_h = np.concatenate([x, r * h_prev])
        h_cand = np.tanh(self.Wh @ x_reset_h + self.bh)
        
        # 4. Final hidden state (convex linear interpolation)
        h = (1.0 - z) * h_prev + z * h_cand
        
        return h, {"r": r, "z": z, "h_cand": h_cand}


def compare_gate_structures():
    print("=" * 70)
    print("COMPARISON: GRU GATE STRUCTURE VS LSTM GATE STRUCTURE")
    print("=" * 70)
    print("""
    +-------------------------+-----------------------------------------+
    | Feature                 | GRU (Cho et al., 2014)                  | LSTM (Hochreiter, 1997)                 |
    +-------------------------+-----------------------------------------+
    | Number of Gates         | 2 (Reset r, Update z)                   | 3 (Forget f, Input i, Output o)         |
    | Separate Cell State     | No (merges c_t into h_t)                | Yes (distinct c_t and h_t)              |
    | Gate Coupling           | Coupled update: (1 - z) and z           | Uncoupled: f_t and i_t independent      |
    | Output Exposure         | Entire h_t is exposed to downstream     | o_t filters how much c_t reaches h_t    |
    | Parameters per Cell     | 3 * (d_in * d_h + d_h^2 + d_h)          | 4 * (d_in * d_h + d_h^2 + d_h)          |
    | Parameter Efficiency    | 25% fewer weights                       | Baseline standard                       |
    +-------------------------+-----------------------------------------+
    """)


if __name__ == "__main__":
    np.random.seed(42)
    in_dim, h_dim = 2, 4
    cell = GRUCell(in_features=in_dim, hidden_size=h_dim)
    
    h = np.zeros(h_dim)
    sample_inputs = [np.array([0.5, -0.2]), np.array([0.1, 0.8]), np.array([-0.4, 0.3])]
    
    print(f"Running GRU Cell forward pass (input_dim={in_dim}, hidden_dim={h_dim}):\n")
    for t, x in enumerate(sample_inputs, start=1):
        h, gates = cell.step(x, h)
        print(f"Timestep {t}: Input x = {x}")
        print(f"  Reset gate  r_t = {np.round(gates['r'], 3)}")
        print(f"  Update gate z_t = {np.round(gates['z'], 3)}")
        print(f"  Hidden state h  = {np.round(h, 3)}\n")
        
    compare_gate_structures()

