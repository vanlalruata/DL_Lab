"""ext22: BPTT equations for a vanilla RNN + exploding gradient demonstration."""
import numpy as np


def bptt_gradients(seq_len, hidden, init_grad=1.0):
    """Product of Jacobian Whh over time -> magnitude shows explosion/vanishing."""
    Whh = np.random.randn(hidden, hidden) * 1.2  # >1 spectral radius -> explode
    g = init_grad
    mags = []
    for _ in range(seq_len):
        # |d h_t / d h_{t-1}| ~ |diag(1-tanh^2) Whh| ; approximate by |Whh|
        g = g * np.linalg.norm(Whh)
        mags.append(g)
    return mags


if __name__ == "__main__":
    mags = bptt_gradients(15, 8)
    print("gradient magnitude growth over 15 steps (last 3):", [f"{m:.2e}" for m in mags[-3:]])
    print("-> demonstrates exploding gradient as product of recurrent Jacobian norms.")
