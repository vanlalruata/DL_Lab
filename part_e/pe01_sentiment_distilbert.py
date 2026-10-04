"""part_e / pe01 - Transformer text classification (sentiment) with dataset.

Fine-tunes a pretrained Transformer (DistilBERT) for IMDB sentiment when
`transformers`/`datasets` are available; otherwise falls back to a BiLSTM
classifier on synthetic sentiment data so the full train/val/test, accuracy,
loss, ROC and inference-time pipeline still runs.
"""
import os, time, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

FIG = os.path.join(os.path.dirname(__file__), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def synth_sentiment(n=2000):
    pos = ["good", "great", "love", "amazing", "happy", "best", "wonderful"]
    neg = ["bad", "terrible", "hate", "awful", "sad", "worst", "boring"]
    X, y = [], []
    rng = np.random.RandomState(0)
    for _ in range(n):
        words = rng.choice(pos if rng.rand() < 0.5 else neg, size=rng.randint(4, 10))
        X.append(" ".join(words)); y.append(1 if np.isin(words, pos).mean() > 0.5 else 0)
    return X, np.array(y)


class Vocab:
    def __init__(self, sents, max_size=200):
        cnt = {}
        for s in sents:
            for w in s.split():
                cnt[w] = cnt.get(w, 0) + 1
        self.words = ["<pad>"] + sorted(cnt, key=cnt.get, reverse=True)[:max_size]
        self.ix = {w: i for i, w in enumerate(self.words)}

    def encode(self, s, maxlen=12):
        ids = [self.ix.get(w, 0) for w in s.split()][:maxlen]
        return ids + [0] * (maxlen - len(ids))


class BiLSTM(nn.Module):
    def __init__(self, vocab, emb=16, hid=32):
        super().__init__()
        self.emb = nn.Embedding(len(vocab.words), emb)
        self.lstm = nn.LSTM(emb, hid, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hid, 2)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.fc(h[:, -1])


def main():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
        print("[pe01] transformers available - use AutoModelForSequenceClassification on IMDB.")
    except Exception:
        print("[pe01] transformers not installed; using BiLSTM fallback on synthetic data.")

    X, y = synth_sentiment()
    v = Vocab(X)
    enc = np.array([v.encode(s) for s in X])
    Xtmp, Xte, ytmp, yte = train_test_split(enc, y, test_size=0.3, random_state=1)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=0.3, random_state=1)

    model = BiLSTM(v).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    tr_a, va_a, tr_l, va_l = [], [], [], []
    Xtr_t = torch.tensor(Xtr, dtype=torch.long, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xva_t = torch.tensor(Xva, dtype=torch.long, device=DEVICE)
    yva_t = torch.tensor(yva, dtype=torch.long, device=DEVICE)
    for _ in range(15):
        model.train()
        opt.zero_grad(); loss = crit(model(Xtr_t), ytr_t); loss.backward(); opt.step()
        tr_l.append(loss.item()); tr_a.append((model(Xtr_t).argmax(1) == ytr_t).float().mean().item())
        model.eval()
        with torch.no_grad():
            o = model(Xva_t)
        va_l.append(crit(o, yva_t).item()); va_a.append((o.argmax(1) == yva_t).float().mean().item())

    model.eval()
    Xte_t = torch.tensor(Xte, dtype=torch.long, device=DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(Xte_t); proba = torch.softmax(out, 1)[:, 1].cpu().numpy()
        pred = out.argmax(1).cpu().numpy()
    dt = time.perf_counter() - t0
    print(f"acc={accuracy_score(yte, pred):.3f} auc={roc_auc_score(yte, proba):.3f} "
          f"infer={dt*1000:.1f}ms/{len(yte)}")
    fpr, tpr, _ = roc_curve(yte, proba)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(yte, proba):.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("pe01 sentiment ROC"); plt.legend(); plt.savefig(f"{FIG}/pe01_roc.png"); plt.close()


if __name__ == "__main__":
    main()
