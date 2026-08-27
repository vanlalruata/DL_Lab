"""ext37: WGAN with weight clipping and a training loop (critic)."""
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def wgan_step(critic, real, fake, opt, clip=0.01):
    opt.zero_grad()
    loss = -(critic(real).mean() - critic(fake).mean())  # Wasserstein distance
    loss.backward()
    opt.step()
    for p in critic.parameters():
        p.data.clamp_(-clip, clip)
    return loss.item()


if __name__ == "__main__":
    C = Critic()
    opt = torch.optim.RMSprop(C.parameters(), lr=1e-3)
    real = torch.randn(64, 2)
    fake = torch.randn(64, 2)
    print("W critic loss:", wgan_step(C, real, fake, opt))
