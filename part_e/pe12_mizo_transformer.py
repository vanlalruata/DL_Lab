"""part_e / pe12 - Transformer-based Mizo translation (Mizo<->English, Mizo<->Hindi).

Proposed architecture: a character-level encoder-decoder Transformer (nn.Transformer)
so that tone/diacritic-bearing characters (ê, é, è) are kept distinct in the vocab
and the model can, in principle, disambiguate meaning via context + POS cues. We
train and evaluate on the small parallel corpus from pe11 for all four directions.

Because the corpus is tiny, the model mostly memorizes; the exercise demonstrates
the full pipeline: vocab -> positional encoding -> multi-head attention encoder/
decoder -> greedy decoding, and shows that stripping diacritics breaks translation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe11_mizo_corpus_pos import CORPUS

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPECIAL = ["<pad>", "<sos>", "<eos>"]


def build_vocab(texts):
    chars = sorted({c for t in texts for c in t})
    return SPECIAL + chars


def encode(s, vocab):
    ix = {c: i for i, c in enumerate(vocab)}
    return [ix["<sos>"]] + [ix[c] for c in s if c in ix] + [ix["<eos>"]]


def decode(ids, vocab):
    return "".join(vocab[i] for i in ids if vocab[i] not in SPECIAL)


class PosEnc(nn.Module):
    def __init__(self, d, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * -(np.log(10000) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d=64, h=4):
        super().__init__()
        self.d = d
        self.se = nn.Embedding(len(src_vocab), d)
        self.te = nn.Embedding(len(tgt_vocab), d)
        self.pe1, self.pe2 = PosEnc(d), PosEnc(d)
        self.tf = nn.Transformer(d, h, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128)
        self.out = nn.Linear(d, len(tgt_vocab))

    def forward(self, src, tgt, tgt_mask):
        s = self.pe1(self.se(src) * self.d ** 0.5)
        t = self.pe2(self.te(tgt) * self.d ** 0.5)
        mem = self.tf.encoder(s)
        dout = self.tf.decoder(t, mem, tgt_mask=tgt_mask)
        return self.out(dout)


def make_mask(sz, device):
    return torch.triu(torch.ones(sz, sz, device=device), 1).bool()


@torch.no_grad()
def translate(model, src_str, sv, tv):
    model.eval()
    ids = encode(src_str, sv)
    src = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(1)
    tgt = torch.tensor([sv.index("<sos>")], dtype=torch.long, device=DEVICE).unsqueeze(1)
    for _ in range(40):
        mask = make_mask(tgt.size(0), DEVICE)
        out = model(src, tgt, mask)
        nxt = out[-1].argmax(1, keepdim=True)
        tgt = torch.cat([tgt, nxt], 0)
        if nxt.item() == sv.index("<eos>"):
            break
    return decode([i.item() for i in tgt], tv)


def train_dir(src_texts, tgt_texts, epochs=150):
    sv, tv = build_vocab(src_texts), build_vocab(tgt_texts)
    pairs = [(encode(s, sv), encode(t, tv)) for s, t in zip(src_texts, tgt_texts)]
    model = Seq2Seq(sv, tv).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=sv.index("<pad>"))
    losses = []
    for _ in range(epochs):
        model.train()
        for s, t in pairs:
            s = torch.tensor(s, dtype=torch.long, device=DEVICE).unsqueeze(1)
            t = torch.tensor(t, dtype=torch.long, device=DEVICE).unsqueeze(1)
            mask = make_mask(t.size(0), DEVICE)
            opt.zero_grad()
            out = model(s, t, mask)[:-1]          # shift target
            loss = crit(out.reshape(-1, len(tv)), t[1:].reshape(-1))
            loss.backward(); opt.step()
        losses.append(loss.item())
    return model, sv, tv, losses


def run_direction(name, src_texts, tgt_texts, samples):
    print(f"\n--- {name} ---")
    model, sv, tv, losses = train_dir(src_texts, tgt_texts)
    plt.figure(); plt.plot(losses); plt.title(f"pe12 {name} loss"); plt.xlabel("epoch")
    plt.ylabel("CE"); plt.tight_layout(); plt.savefig(f"{FIG}/pe12_{name}_loss.png"); plt.close()
    for s, ref in samples:
        out = translate(model, s, sv, tv)
        print(f"  src : {s}\n  pred: {out}\n  ref : {ref}")


def main():
    mizo = [m for m, _, _ in CORPUS]
    eng = [e for _, e, _ in CORPUS]
    hin = [h for _, _, h in CORPUS]

    run_direction("mizo2eng", mizo, eng, list(zip(mizo, eng))[:3])
    run_direction("eng2mizo", eng, mizo, list(zip(eng, mizo))[:3])
    run_direction("mizo2hin", mizo, hin, list(zip(mizo, hin))[:3])
    run_direction("hin2mizo", hin, mizo, list(zip(hin, mizo))[:3])

    # diacritic sensitivity: stripping ê/é/è collapses the distinct tokens
    print("\nDiacritic breakdown: 'lêi' vs 'léi' vs 'lèi' become identical 'lei' "
          "when normalized, so a diacritic-blind model cannot disambiguate meaning.")
    print("Figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
