"""part_d / security_utils.py

Shared utilities for the security dataset pipelines (pd11-pd13):
  * download + extract KDD'99, NSL-KDD, INSDN (with synthetic fallback if offline)
  * preprocessing (categorical encoding, scaling, binary label)
  * model builders: CNN1D, LSTM, GRU, Hybrid CNN+LSTM, GAN (augmentation)
  * training / evaluation with accuracy, loss, ROC, confusion matrix, timing
  * interpretability: ANOVA (f_classif), SHAP and LIME (optional, guarded)

Everything degrades gracefully: if a dataset cannot be downloaded, a synthetic
dataset with the same schema is generated so the full pipeline still runs.
SHAP/LIME are attempted only if installed; otherwise a message is printed.
"""
import os
import sys
import time
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve)
from sklearn.feature_selection import f_classif

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- dataset source URLs (public mirrors; fall back to synthetic on failure) ----
URLS = {
    "kdd99": "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99/kddcup.data_10_percent.gz",
    "nslkdd_train": "https://raw.githubusercontent.com/defcom17/NSL-KDD/master/KDDTrain+.txt",
    "nslkdd_test": "https://raw.githubusercontent.com/defcom17/NSL-KDD/master/KDDTest+.txt",
    "insdn": "https://raw.githubusercontent.com/rsrieb/INSDN/main/INSDN_Flows.csv",
}
KDD_COLS = ["duration", "protocol_type", "service", "flag"] + [f"f{i}" for i in range(4, 41)]
CAT_IDX = [1, 2, 3]  # protocol_type, service, flag


# ----------------------------------------------------------------------------- download
def _download(url, dest, timeout=20):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url} ...")
    urllib.request.urlretrieve(url, dest)  # raises on failure


def _maybe_download(name, dest):
    if os.path.exists(dest):
        return True
    try:
        _download(URLS[name], dest)
        return True
    except Exception as e:
        print(f"  [warn] download failed ({e}); using synthetic data.")
        return False


# ----------------------------------------------------------------------------- preprocessing
def _preprocess_kdd(path, is_nsl=False, n_synth=None):
    """Read KDD/NSL-KDD raw lines; encode categoricals; return X, y, feature_names."""
    with open(path) as f:
        lines = [ln.strip().split(",") for ln in f if ln.strip()]
    if n_synth:
        lines = lines[:n_synth]
    raw = np.array(lines, dtype=object)
    y_raw = raw[:, -1]
    y = (y_raw != "normal").astype(int)
    X = raw[:, :-1].copy()
    feat_names = list(KDD_COLS)
    # encode categorical columns
    for ci in CAT_IDX:
        le = LabelEncoder()
        col = X[:, ci].astype(str)
        col[col == "?"] = "unknown"
        X[:, ci] = le.fit_transform(col)
    X = X.astype(float)
    return X, y, feat_names


def _preprocess_insdn(path):
    """Generic CSV loader: detect a label column, keep numeric features."""
    import pandas as pd
    df = pd.read_csv(path)
    label_col = next((c for c in df.columns if c.lower() in ("label", "attack", "class")),
                     df.columns[-1])
    y = (df[label_col].astype(str).str.lower() != "normal").astype(int).values
    Xdf = df.drop(columns=[label_col])
    Xdf = Xdf.select_dtypes(include=[np.number]).fillna(0)
    return Xdf.values.astype(float), y, list(Xdf.columns)


def _synthetic(n=8000, F=41, seed=0):
    rng = np.random.RandomState(seed)
    nn = n // 2
    normal = rng.normal(0, 1, (nn, F))
    attack = rng.normal(0.6, 1, (n - nn, F))
    attack[:, 1:4] += rng.normal(1.5, 0.5, (n - nn, 3))
    X = np.vstack([normal, attack]); y = np.hstack([np.zeros(nn), np.ones(n - nn)])
    names = [f"f{i}" for i in range(F)]
    return X, y, names


def get_dataset(name):
    """Return (X, y, feature_names) using real data if available, else synthetic."""
    data_dir = os.path.join(HERE, "data", name)
    if name == "kdd99":
        dest = os.path.join(data_dir, "kddcup.data_10_percent.gz")
        if _maybe_download("kdd99", dest):
            import gzip, shutil
            txt = dest[:-3]
            if not os.path.exists(txt):
                with gzip.open(dest, "rb") as f_in, open(txt, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return _preprocess_kdd(txt, n_synth=8000)
    elif name == "nslkdd":
        tr = os.path.join(data_dir, "KDDTrain+.txt")
        if _maybe_download("nslkdd_train", tr):
            return _preprocess_kdd(tr, is_nsl=True, n_synth=8000)
    elif name == "insdn":
        dest = os.path.join(data_dir, "INSDN_Flows.csv")
        if _maybe_download("insdn", dest):
            return _preprocess_insdn(dest)
    print(f"[{name}] synthetic fallback active.")
    return _synthetic()


# ----------------------------------------------------------------------------- models
class CNN1D(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x.unsqueeze(1))  # (N,F)->(N,1,F)


class RNN_IDS(nn.Module):
    def __init__(self, F, cell="LSTM"):
        super().__init__()
        self.cell = cell
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn(1, 32, batch_first=True)
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (N,F)->(N,F,1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class Hybrid(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
                                 nn.MaxPool1d(2))
        self.rnn = nn.LSTM(16, 32, batch_first=True)
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        h = self.conv(x.unsqueeze(1))          # (N,16,L)
        h = h.transpose(1, 2)                  # (N,L,16)
        out, _ = self.rnn(h)
        return self.head(out[:, -1])


class Generator(nn.Module):
    def __init__(self, z=16, F=41):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z, 32), nn.ReLU(), nn.Linear(32, F), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, F=41):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(F, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


def make_model(kind, F):
    if kind == "cnn":
        return CNN1D(F)
    if kind in ("lstm", "gru"):
        return RNN_IDS(F, kind)
    if kind == "hybrid":
        return Hybrid(F)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- training
def train_model(model, Xtr, ytr, Xva, yva, epochs=10, batch=256, lr=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva_t = torch.tensor(yva, dtype=torch.long, device=DEVICE)
    n = len(Xtr_t)
    tr_a, va_a, tr_l, va_l = [], [], [], []
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train(); idx = torch.randperm(n)
        ca = cl = 0
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            opt.zero_grad(); loss = crit(model(Xtr_t[b]), ytr_t[b]); loss.backward(); opt.step()
            cl += loss.item() * len(b); ca += (model(Xtr_t[b]).argmax(1) == ytr_t[b]).sum().item()
        tr_l.append(cl / n); tr_a.append(ca / n)
        model.eval(); vc = vl = 0
        with torch.no_grad():
            out = model(Xva_t); vl += crit(out, yva_t).item() * len(yva)
            vc += (out.argmax(1) == yva_t).sum().item()
        va_l.append(vl / len(yva)); va_a.append(vc / len(yva))
    train_time = time.perf_counter() - t0
    return dict(model=model, tr_a=tr_a, va_a=va_a, tr_l=tr_l, va_l=va_l, train_time=train_time)


def evaluate(model, Xte, yte):
    model.eval()
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(Xte_t); proba = torch.softmax(out, 1)[:, 1].cpu().numpy()
        pred = out.argmax(1).cpu().numpy()
    infer_ms = (time.perf_counter() - t0) * 1000
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan")
    cm = confusion_matrix(yte, pred)
    return dict(acc=acc, auc=auc, proba=proba, pred=pred, cm=cm,
                infer_ms=infer_ms, n_params=sum(p.numel() for p in model.parameters()))


# ----------------------------------------------------------------------------- plots
def plot_results(name, kind, res, yte, tag):
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1); plt.plot(res["tr_a"], label="train"); plt.plot(res["va_a"], label="val")
    plt.ylabel("acc"); plt.legend(); plt.title(f"{tag}/{kind} accuracy")
    plt.subplot(1, 2, 2); plt.plot(res["tr_l"], label="train"); plt.plot(res["va_l"], label="val")
    plt.ylabel("loss"); plt.legend(); plt.title(f"{tag}/{kind} loss")
    plt.tight_layout(); plt.savefig(f"{FIG}/{tag}_{kind}_curves.png"); plt.close()

    auc = res.get("auc", float("nan"))
    if not np.isnan(auc):
        fpr, tpr, _ = roc_curve(yte, res["proba"])
        plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{tag}/{kind} ROC"); plt.legend()
        plt.savefig(f"{FIG}/{tag}_{kind}_roc.png"); plt.close()

    plt.figure(); plt.imshow(res["cm"], cmap="Blues")
    plt.title(f"{tag}/{kind} confusion matrix"); plt.xlabel("pred"); plt.ylabel("true")
    plt.colorbar(); plt.savefig(f"{FIG}/{tag}_{kind}_cm.png"); plt.close()


# ----------------------------------------------------------------------------- GAN augmentation
def gan_augment(X, y, F, n_synth=800, epochs=120):
    """Train a GAN on minority (attack) samples; return augmented (X_aug, y_aug)."""
    Xa = torch.tensor(X[y == 1], dtype=torch.float32, device=DEVICE)
    G, D = Generator(F=F).to(DEVICE), Discriminator(F=F).to(DEVICE)
    oG = torch.optim.Adam(G.parameters(), lr=1e-3)
    oD = torch.optim.Adam(D.parameters(), lr=1e-3)
    for _ in range(epochs):
        z = torch.randn(len(Xa), 16, device=DEVICE)
        fake = G(z).detach()
        oD.zero_grad()
        d_loss = nn.functional.binary_cross_entropy_with_logits(D(Xa), torch.ones(len(Xa), 1, device=DEVICE)) \
            + nn.functional.binary_cross_entropy_with_logits(D(fake), torch.zeros(len(Xa), 1, device=DEVICE))
        d_loss.backward(); oD.step()
        z = torch.randn(len(Xa), 16, device=DEVICE)
        oG.zero_grad()
        g_loss = nn.functional.binary_cross_entropy_with_logits(D(G(z)), torch.ones(len(Xa), 1, device=DEVICE))
        g_loss.backward(); oG.step()
    with torch.no_grad():
        synth = G(torch.randn(n_synth, 16, device=DEVICE)).cpu().numpy()
    return np.vstack([X, synth]), np.hstack([y, np.ones(n_synth)])


# ----------------------------------------------------------------------------- interpretability
def anova_top(X, y, feat_names, k=15):
    F, p = f_classif(X, y)
    order = np.argsort(F)[::-1][:k]
    plt.figure(); plt.barh([feat_names[i] for i in order][::-1], F[order][::-1])
    plt.title("ANOVA F-score (top features)"); plt.tight_layout()
    plt.savefig(f"{FIG}/interpret_anova.png"); plt.close()
    return order


def shap_summary(X, y, feat_names):
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        n = min(2000, len(X))
        rf = RandomForestClassifier(n_estimators=50, n_jobs=-1).fit(X[:n], y[:n])
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X[:200])
        plt.figure(); shap.summary_plot(sv, X[:200], feature_names=feat_names, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(f"{FIG}/interpret_shap.png"); plt.close()
        print("  SHAP summary saved.")
    except Exception as e:
        print(f"  SHAP skipped ({e}). Install with: pip install shap")


def lime_explain(X, y, feat_names):
    try:
        from lime.lime_tabular import LimeTabularExplainer
        from sklearn.ensemble import RandomForestClassifier
        n = min(2000, len(X))
        rf = RandomForestClassifier(n_estimators=50, n_jobs=-1).fit(X[:n], y[:n])
        exp = LimeTabularExplainer(X[:n], feature_names=feat_names,
                                   class_names=["normal", "attack"], discretize_continuous=True)
        fig = exp.explain_instance(X[n], rf.predict_proba, num_features=10).as_pyplot_figure()
        plt.tight_layout(); plt.savefig(f"{FIG}/interpret_lime.png"); plt.close()
        print("  LIME explanation saved.")
    except Exception as e:
        print(f"  LIME skipped ({e}). Install with: pip install lime")


# ----------------------------------------------------------------------------- orchestration
def run_pipeline(name, kinds=("cnn", "lstm", "gru", "hybrid"), epochs=10, subsample=8000):
    print(f"\n========== DATASET: {name} ==========")
    X, y, feat_names = get_dataset(name)
    if len(X) > subsample:
        X, y = X[:subsample], y[:subsample]
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=1)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    F = X.shape[1]
    print(f"samples={len(X)} features={F}  train/val/test={len(Xtr)}/{len(Xva)}/{len(Xte)}")

    summary = []
    for kind in kinds:
        print(f"  -- training {kind} --")
        model = make_model(kind, F)
        res = train_model(model, Xtr, ytr, Xva, yva, epochs=epochs)
        ev = evaluate(res["model"], Xte, yte)
        full = {**res, **ev}
        plot_results(name, kind, full, yte, name)
        summary.append((kind, ev["acc"], ev["auc"], ev["infer_ms"], ev["n_params"], res["train_time"]))
        print(f"     acc={ev['acc']:.4f} auc={ev['auc']:.4f} "
              f"infer={ev['infer_ms']:.2f}ms params={ev['n_params']:,} "
              f"train={res['train_time']:.2f}s")

    # GAN augmentation effect
    print("  -- GAN augmentation --")
    Xa, ya = gan_augment(Xtr, ytr, F)
    sc2 = StandardScaler().fit(Xa)
    Xa, Xva2, Xte2 = sc2.transform(Xa), sc2.transform(Xva), sc2.transform(Xte)
    m = make_model("cnn", F); r = train_model(m, Xa, ya, Xva2, yva, epochs=epochs)
    e2 = evaluate(r["model"], Xte2, yte)
    plot_results(name, "cnn_gan", {**r, **e2}, yte, name)
    summary.append(("cnn+gan", e2["acc"], e2["auc"], e2["infer_ms"], e2["n_params"], r["train_time"]))
    print(f"     GAN-aug acc={e2['acc']:.4f} auc={e2['auc']:.4f}")

    # interpretability on raw standardized features
    print("  -- interpretability --")
    anova_top(Xtr, ytr, feat_names)
    shap_summary(Xtr, ytr, feat_names)
    lime_explain(Xtr, ytr, feat_names)

    # summary table
    plt.figure(figsize=(9, 4))
    labels = [s[0] for s in summary]
    plt.subplot(1, 3, 1); plt.bar(labels, [s[1] for s in summary]); plt.title("accuracy")
    plt.subplot(1, 3, 2); plt.bar(labels, [s[2] for s in summary]); plt.title("ROC-AUC")
    plt.subplot(1, 3, 3); plt.bar(labels, [s[3] for s in summary]); plt.title("infer ms")
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}_summary.png"); plt.close()
    print(f"[{name}] done. Figures in part_d/figures/")
    return summary


if __name__ == "__main__":
    run_pipeline("kdd99")
    run_pipeline("nslkdd")
    run_pipeline("insdn")
