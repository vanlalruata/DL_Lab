"""ext36: The minimax GAN objective and Nash equilibrium (theoretical + toy check)."""
import torch


def minimax_value(d_real, d_fake):
    """Value function V = E[log D(x)] + E[log(1 - D(G(z)))]. Equilibrium at D=0.5."""
    return (torch.log(d_real).mean() + torch.log(1 - d_fake).mean()).item()


if __name__ == "__main__":
    # at equilibrium discriminator outputs 0.5 everywhere
    d_real = torch.full((4, 1), 0.5)
    d_fake = torch.full((4, 1), 0.5)
    print("V at D=0.5 (equilibrium):", minimax_value(d_real, d_fake))
    print("Generator wants to push D(G(z)) -> 1; Discriminator wants to separate.")
