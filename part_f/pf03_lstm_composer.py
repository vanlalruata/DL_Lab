"""part_f / pf03 - Music composition with an LSTM (character/event-level).

Trains an LSTM language model over tokenized melodies and generates novel melodies
by greedy (or temperature) sampling. Optionally conditions on a genre by seeding
with a genre-specific style. Saves the composed melody as MIDI (or .npz fallback)
and plots the training loss.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import music_utils as mu

FIG = os.path.join(os.path.dirname(__file__), "figures")
GEN_DIR = os.path.join(os.path.dirname(__file__), "generated")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTM_LM(nn.Module):
    def __init__(self, vocab, emb=32, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.head = nn.Linear(hid, vocab)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.head(h)


def main():
    data = mu.download_or_load()
    seqs = data["seq"]
    vocab, ivocab = mu.build_vocab(seqs)
    toks = [mu.tokenize_seq(s, vocab) for s in seqs]
    maxlen = max(len(t) for t in toks)
    pad = vocab["<pad>"]
    X = torch.tensor([t + [pad] * (maxlen - len(t)) for t in toks], dtype=torch.long, device=DEVICE)
    y = torch.tensor([t[1:] + [pad] * (maxlen - len(t)) for t in toks], dtype=torch.long, device=DEVICE)

    model = LSTM_LM(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=pad)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        out = model(X[:, :-1])
        loss = crit(out.reshape(-1, len(vocab)), y.reshape(-1))
        loss.backward(); opt.step(); losses.append(loss.item())
    print("LSTM final loss:", round(losses[-1], 3))
    plt.figure(); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("CE")
    plt.title("pf03 LSTM composer training loss"); plt.savefig(f"{FIG}/pf03_loss.png"); plt.close()

    # generate with temperature sampling (require a minimum length before stopping)
    def sample(logits, temp=0.8):
        return torch.multinomial(torch.softmax(logits / temp, -1), 1).item()

    model.eval()
    with torch.no_grad():
        ids = [vocab["<sos>"]]
        for _ in range(maxlen - 1):
            x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            nxt = sample(model(x)[0, -1])
            if nxt == vocab["<eos>"] and len(ids) > 10:
                break
            if nxt == vocab["<pad>"]:
                continue
            ids.append(nxt)
    melody = mu.detokenize(ids, ivocab)
    print(f"Composed melody ({len(melody)} notes): {melody[:12]} ...")
    path = os.path.join(GEN_DIR, "lstm_composition.mid")
    ok = mu.save_midi(melody, path)
    print(f"Saved composition to {path} (midi={ok})")


if __name__ == "__main__":
    main()
