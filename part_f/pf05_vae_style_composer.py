"""part_f / pf05 - Music representation & composition with a VAE.

Trains a Variational Autoencoder over the MIDI feature vectors to learn a latent
space of musical style. Visualizes the latent space (PCA) colored by genre, then
composes novel melodies by interpolating between genre centroids in latent space
and decoding to feature statistics (style transfer between genres). Saves the
composed MIDI and a latent-space plot.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import music_utils as mu

FIG = os.path.join(os.path.dirname(__file__), "figures")
GEN_DIR = os.path.join(os.path.dirname(__file__), "generated")
GEN = mu.GENRES
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VAE(nn.Module):
    def __init__(self, in_f=8, h=16, z=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_f, h), nn.ReLU())
        self.mu = nn.Linear(h, z); self.logvar = nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, in_f))

    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.logvar(h)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return self.dec(z), mu, lv


def main():
    data = mu.download_or_load()
    feats = np.array([mu.extract_features(s) for s in data["seq"]], dtype=np.float32)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(feats)
    X = torch.tensor(scaler.transform(feats), dtype=torch.float32, device=DEVICE)
    y = np.array(data["genre"])
    model = VAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(120):
        opt.zero_grad()
        recon, z_mu, z_lv = model(X)
        loss = ((recon - X) ** 2).mean() + (-0.5 * (1 + z_lv - z_mu ** 2 - z_lv.exp())).mean()
        loss.backward(); opt.step(); losses.append(loss.item())
    print("VAE final loss:", round(losses[-1], 4))
    plt.figure(); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("ELBO loss")
    plt.title("pf05 VAE training"); plt.savefig(f"{FIG}/pf05_loss.png"); plt.close()

    # latent means per genre
    model.eval()
    with torch.no_grad():
        h = model.enc(X); z_mu = model.mu(h).cpu().numpy()
    from sklearn.decomposition import PCA
    z2 = PCA(2).fit_transform(z_mu)
    plt.figure()
    for i, g in enumerate(GEN):
        plt.scatter(z2[y == g, 0], z2[y == g, 1], label=g)
    plt.legend(); plt.title("pf05 latent space (PCA) by genre")
    plt.tight_layout(); plt.savefig(f"{FIG}/pf05_latent.png"); plt.close()

    # compose by interpolating between two genre centroids
    centroids = {g: z_mu[y == g].mean(0) for g in GEN}
    a, b = "classical", "rock"
    for alpha in [0.0, 0.5, 1.0]:
        z = (1 - alpha) * centroids[a] + alpha * centroids[b]
        with torch.no_grad():
            feat_scaled = model.dec(torch.tensor(z, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        feat = scaler.inverse_transform(feat_scaled.reshape(1, -1))[0]
        melody = mu.compose_from_features(feat, seed=int(alpha * 100))
        tag = f"{a}_to_{b}_{int(alpha*100)}"
        ok = mu.save_midi(melody, os.path.join(GEN_DIR, f"vae_{tag}.mid"))
        print(f"composed {tag}: pitch_mu={feat[0]:.1f} dur_mu={feat[3]:.2f} (midi={ok})")
    print("Figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
