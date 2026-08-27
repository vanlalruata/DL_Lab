"""part_f / pf04 - Music composition with a Transformer (decoder-only).

Trains a causal Transformer LM over tokenized melodies (genre-conditioned by
prepending a genre token) and generates melodies. Compares training dynamics with
the LSTM composer (pf03) and saves a composed MIDI.
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PosEnc(nn.Module):
    def __init__(self, d, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * -(np.log(10000) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class TransformerLM(nn.Module):
    def __init__(self, vocab, d=64, h=4):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab, d); self.pe = PosEnc(d)
        layer = nn.TransformerDecoderLayer(d, h, dim_feedforward=128, batch_first=False)
        self.tf = nn.TransformerDecoder(layer, num_layers=2)
        self.head = nn.Linear(d, vocab)

    def forward(self, x, mask):
        e = self.pe(self.emb(x) * self.d ** 0.5)
        # decoder-only: cross-attend to a zero memory (no future leakage)
        memory = torch.zeros_like(e)
        out = self.tf(e, memory, tgt_mask=mask)
        return self.head(out)


def main():
    data = mu.download_or_load()
    seqs = data["seq"]
    vocab, ivocab = mu.build_vocab(seqs)
    toks = [mu.tokenize_seq(s, vocab) for s in seqs]
    maxlen = max(len(t) for t in toks) + 1
    pad = vocab["<pad>"]
    X = torch.tensor([t + [pad] * (maxlen - len(t)) for t in toks], dtype=torch.long, device=DEVICE)
    y = torch.tensor([t[1:] + [pad] * (maxlen - len(t)) for t in toks], dtype=torch.long, device=DEVICE)

    model = TransformerLM(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=pad)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        mask = torch.triu(torch.ones(maxlen - 1, maxlen - 1, device=DEVICE), 1).bool()
        out = model(X[:, :-1].transpose(0, 1), mask)
        loss = crit(out.reshape(-1, len(vocab)), y.reshape(-1))
        loss.backward(); opt.step(); losses.append(loss.item())
    print("Transformer final loss:", round(losses[-1], 3))
    plt.figure(); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("CE")
    plt.title("pf04 Transformer composer training loss"); plt.savefig(f"{FIG}/pf04_loss.png"); plt.close()

    model.eval()
    with torch.no_grad():
        ids = [vocab["<sos>"]]
        for _ in range(maxlen - 1):
            x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(1)
            mask = torch.triu(torch.ones(len(ids), len(ids), device=DEVICE), 1).bool()
            logits = model(x, mask)[-1]
            p = torch.softmax(logits / 0.8, -1)
            nxt = torch.multinomial(p, 1).item()
            if nxt == vocab["<eos>"] and len(ids) > 10:
                break
            if nxt == vocab["<pad>"]:
                continue
            ids.append(nxt)
    melody = mu.detokenize(ids, ivocab)
    print(f"Composed melody ({len(melody)} notes): {melody[:12]} ...")
    path = os.path.join(GEN_DIR, "transformer_composition.mid")
    ok = mu.save_midi(melody, path)
    print(f"Saved composition to {path} (midi={ok})")


if __name__ == "__main__":
    main()
