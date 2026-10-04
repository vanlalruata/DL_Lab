"""
Part B — Exercise 59: GRU Sequence Modeling & Comparative Lab Exercise (PyTorch)
Demonstrates:
  1. Custom PyTorch GRU Cell & Layer implementation from scratch
  2. Native PyTorch nn.GRU usage for multi-step sequence forecasting
  3. Comprehensive benchmark: Vanilla RNN vs LSTM vs GRU
  4. Accuracy, loss, parameter footprint, and inference speed analysis
  5. Visual comparison curves saved to part_b/figures/
  6. Structured student lab exercises with guided experiments
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 1. CUSTOM GRU IMPLEMENTATION IN PYTORCH
# =============================================================================

class CustomGRUCellPyTorch(nn.Module):
    """Explicit PyTorch GRU cell exposing reset and update gate states."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset gate parameters
        self.w_xr = nn.Linear(input_size, hidden_size, bias=True)
        self.w_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Update gate parameters
        self.w_xz = nn.Linear(input_size, hidden_size, bias=True)
        self.w_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Candidate hidden state parameters
        self.w_xh = nn.Linear(input_size, hidden_size, bias=True)
        self.w_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t, h_prev):
        """Single timestep GRU update."""
        r_t = torch.sigmoid(self.w_xr(x_t) + self.w_hr(h_prev))
        z_t = torch.sigmoid(self.w_xz(x_t) + self.w_hz(h_prev))
        h_tilde = torch.tanh(self.w_xh(x_t) + self.w_hh(r_t * h_prev))
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
        return h_t, (r_t, z_t)


class CustomGRUNet(nn.Module):
    """Sequence model built with custom GRU cells."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cell = CustomGRUCellPyTorch(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        for t in range(seq_len):
            h, _ = self.cell(x[:, t, :], h)
            
        out = self.fc(h)
        return out


# =============================================================================
# 2. COMPARATIVE BENCHMARK MODELS (Native PyTorch)
# =============================================================================

class SequencePredictor(nn.Module):
    """Unified wrapper for nn.RNN, nn.LSTM, and nn.GRU."""
    
    def __init__(self, model_type, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.model_type = model_type.upper()
        
        if self.model_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.model_type == "LSTM":
            out, (h_n, c_n) = self.rnn(x)
            last_hidden = h_n[-1]
        else:
            out, h_n = self.rnn(x)
            last_hidden = h_n[-1]
            
        return self.fc(last_hidden)


# =============================================================================
# 3. SYNTHETIC SEQUENCE DATASET GENERATION
# =============================================================================

def generate_multi_harmonic_data(n_points=1200, seq_len=24, test_split=0.2):
    """Generates composite multi-harmonic signal with noise and forms sliding sequences."""
    np.random.seed(42)
    t = np.linspace(0, 40 * np.pi, n_points)
    
    # Composite signal: fundamental + harmonics + minor trend + noise
    signal = (
        np.sin(t) +
        0.5 * np.sin(2.5 * t) +
        0.3 * np.cos(5.0 * t) +
        0.05 * np.sin(0.2 * t) +
        np.random.normal(0, 0.08, size=n_points)
    )
    
    X, y = [], []
    for i in range(len(signal) - seq_len):
        X.append(signal[i:i + seq_len])
        y.append(signal[i + seq_len])
        
    X = np.array(X, dtype=np.float32)[:, :, np.newaxis]
    y = np.array(y, dtype=np.float32)[:, np.newaxis]
    
    split_idx = int(len(X) * (1 - test_split))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    return X_train, y_train, X_test, y_test, signal


# =============================================================================
# 4. TRAINING & EVALUATION PIPELINE
# =============================================================================

def train_and_evaluate(model, train_loader, X_test_t, y_test_t, epochs=15, lr=0.005):
    """Trains model and logs performance metrics."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            
        epoch_loss = total_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        
    elapsed_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_mse = criterion(test_preds, y_test_t).item()
        test_mae = torch.mean(torch.abs(test_preds - y_test_t)).item()
        
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "loss_history": loss_history,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "train_time": elapsed_time,
        "param_count": param_count,
        "predictions": test_preds.cpu().numpy()
    }


# =============================================================================
# 5. LAB EXERCISE & VISUALIZATION
# =============================================================================

def run_gru_exercise():
    """Runs the complete comparative exercise and produces visual artifacts."""
    print("=" * 80)
    print("Part B — Exercise 59: GRU Sequence Modeling & Comparative Benchmark")
    print("=" * 80)
    
    # 1. Prepare data
    seq_len = 24
    X_train, y_train, X_test, y_test, raw_signal = generate_multi_harmonic_data(
        n_points=1200, seq_len=seq_len, test_split=0.2
    )
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)
    
    print(f"Dataset generated:")
    print(f"  Training samples:   {X_train.shape[0]} (sequence length: {seq_len})")
    print(f"  Testing samples:    {X_test.shape[0]}")
    print(f"  Input dimension:    1 (univariate time-series forecasting)\n")
    
    # 2. Benchmark Models
    hidden_dim = 32
    models = {
        "Custom GRU": CustomGRUNet(input_size=1, hidden_size=hidden_dim, output_size=1),
        "PyTorch GRU": SequencePredictor("GRU", input_size=1, hidden_size=hidden_dim, output_size=1),
        "PyTorch LSTM": SequencePredictor("LSTM", input_size=1, hidden_size=hidden_dim, output_size=1),
        "PyTorch RNN": SequencePredictor("RNN", input_size=1, hidden_size=hidden_dim, output_size=1)
    }
    
    results = {}
    print(f"{'Model':<16} | {'Params':<8} | {'Train Time':<10} | {'Test MSE':<10} | {'Test MAE':<10}")
    print("-" * 65)
    
    for name, model in models.items():
        res = train_and_evaluate(model, train_loader, X_test_t, y_test_t, epochs=20, lr=0.005)
        results[name] = res
        print(f"{name:<16} | {res['param_count']:<8} | {res['train_time']:<8.2f}s | {res['test_mse']:<10.5f} | {res['test_mae']:<10.5f}")
        
    print("-" * 65)
    
    # 3. Create Plots
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Loss curves
    for name in results:
        ax1.plot(results[name]["loss_history"], label=name, lw=2)
    ax1.set_title("Training Loss Convergence (MSE vs Epoch)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Mean Squared Error", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")
    
    # Test Predictions overlay (first 100 points)
    n_plot = 100
    ax2.plot(y_test[:n_plot], label="Ground Truth", color="black", lw=2.5, linestyle="--")
    ax2.plot(results["PyTorch GRU"]["predictions"][:n_plot], label="GRU Prediction", color="royalblue", lw=1.8)
    ax2.plot(results["PyTorch LSTM"]["predictions"][:n_plot], label="LSTM Prediction", color="forestgreen", lw=1.8)
    ax2.plot(results["PyTorch RNN"]["predictions"][:n_plot], label="Vanilla RNN Prediction", color="crimson", lw=1.5, alpha=0.8)
    
    ax2.set_title(f"Test Set Prediction Comparison (First {n_plot} Timesteps)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Timestep", fontsize=11)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "gru_benchmark_comparison.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    
    print(f"\n[Artifact Saved] Comparison chart saved to: {fig_path}")
    
    # 4. Lab Practical Exercise Questions
    print("\n" + "=" * 80)
    print("PRACTICAL LAB EXERCISES FOR STUDENTS:")
    print("=" * 80)
    print("Exercise 1: Gate Inspection")
    print("  Modify `CustomGRUCellPyTorch` to record the mean activation of the reset gate r_t")
    print("  and update gate z_t over training. Which gate becomes more active as the model converges?")
    print("\nExercise 2: Sensitivity to Sequence Length")
    print("  Run this benchmark with seq_len in {12, 24, 48, 96}. Observe at what sequence length")
    print("  the Vanilla RNN begins to severely underperform relative to GRU and LSTM.")
    print("\nExercise 3: Efficiency Comparison")
    print(f"  Confirm mathematically that PyTorch GRU has ~{results['PyTorch GRU']['param_count']} parameters")
    print(f"  while LSTM has ~{results['PyTorch LSTM']['param_count']} parameters, saving exactly 25% weights.")
    print("=" * 80)


if __name__ == "__main__":
    run_gru_exercise()
