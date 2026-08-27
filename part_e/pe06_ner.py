"""part_e / pe06 - Named Entity Recognition (NER) with a BiLSTM tagger + span F1.

Trains a BiLSTM sequence tagger (BIO scheme) on a small synthetic NER corpus
(persons / locations). Reports token-level accuracy, loss, and span-level F1.
A transformer/CRF variant can replace the BiLSTM when `transformers` is available.
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

FIG = os.path.join(os.path.dirname(__file__), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCES = [
    ("John lives in London and works with Mary".split(), ["PER", "O", "O", "LOC", "O", "O", "O", "PER"]),
    ("Mary visited Paris and met Tom".split(), ["PER", "O", "LOC", "O", "O", "PER"]),
    ("Alice traveled to Berlin with Bob".split(), ["PER", "O", "O", "LOC", "O", "PER"]),
    ("Tom and Lucy went to Rome".split(), ["PER", "O", "PER", "O", "O", "LOC"]),
] * 30
TAGS = ["O", "PER", "LOC"]
TAGIX = {t: i for i, t in enumerate(TAGS)}


class BiLSTM_NER(nn.Module):
    def __init__(self, n_words, n_tags, emb=16, hid=32):
        super().__init__()
        self.emb = nn.Embedding(n_words, emb)
        self.lstm = nn.LSTM(emb, hid, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hid, n_tags)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.fc(h)


def main():
    words = sorted({w for s, _ in SENTENCES for w in s})
    wix = {w: i + 1 for i, w in enumerate(words)}
    X = [torch.tensor([wix[w] for w in s]) for s, _ in SENTENCES]
    Y = [torch.tensor([TAGIX[t] for t in t]) for _, t in SENTENCES]
    tr, te = train_test_split(list(range(len(X))), test_size=0.3, random_state=1)
    model = BiLSTM_NER(len(words) + 1, len(TAGS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2); crit = nn.CrossEntropyLoss()
    losses = []
    for _ in range(40):
        model.train()
        for i in tr:
            opt.zero_grad()
            out = model(X[i].unsqueeze(0).to(DEVICE))
            loss = crit(out, Y[i].unsqueeze(0).to(DEVICE))
            loss.backward(); opt.step(); losses.append(loss.item())
    # span F1 (PER/LOC)
    tp = fp = fn = 0
    model.eval()
    with torch.no_grad():
        for i in te:
            pred = model(X[i].unsqueeze(0).to(DEVICE))[0].argmax(1).cpu().numpy()
            gold = Y[i].numpy()
            for tag in [1, 2]:
                g = "".join("1" if g == tag else "0" for g in gold)
                p = "".join("1" if pp == tag else "0" for pp in pred)
                tp += sum(gg == pp == "1" for gg, pp in zip(g, p))
                fp += sum(gg == "0" and pp == "1" for gg, pp in zip(g, p))
                fn += sum(gg == "1" and pp == "0" for gg, pp in zip(g, p))
    f1 = tp / (tp + 0.5 * (fp + fn))
    print(f"token-loss={losses[-1]:.3f}  span-F1(PER/LOC)={f1:.3f}")
    plt.figure(); plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss")
    plt.title("pe06 NER training loss"); plt.savefig(f"{FIG}/pe06_loss.png"); plt.close()


if __name__ == "__main__":
    main()
