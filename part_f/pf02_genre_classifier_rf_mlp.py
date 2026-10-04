"""part_f / pf02 - Genre classification from MIDI features.

Trains a Random Forest and an MLP on the extracted MIDI features to classify genre.
Reports accuracy, confusion matrix, per-class ROC (one-vs-rest), and inference time.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)
import music_utils as mu

FIG = os.path.join(os.path.dirname(__file__), "figures")
GEN = mu.GENRES


def main():
    data = mu.download_or_load()
    X = np.array([mu.extract_features(s) for s in data["seq"]])
    y = np.array(data["genre"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    models = {"RF": RandomForestClassifier(n_estimators=120, random_state=0),
              "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=0)}
    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        proba = m.predict_proba(Xte)
        acc = accuracy_score(yte, pred)
        t0 = time.perf_counter(); _ = m.predict(Xte); dt = time.perf_counter() - t0
        # macro ROC-AUC (OvR) -- align proba columns with model.classes_
        from sklearn.preprocessing import label_binarize
        classes = list(m.classes_)
        Y = label_binarize(yte, classes=GEN)
        proba = m.predict_proba(Xte)[:, [classes.index(g) for g in GEN]]
        auc = roc_auc_score(Y, proba, average="macro", multi_class="ovr")
        results[name] = (acc, auc, dt * 1000)
        print(f"[{name}] acc={acc:.3f} macro-AUC={auc:.3f} infer={dt*1000:.2f}ms")
        print(confusion_matrix(yte, pred, labels=GEN))
        print(classification_report(yte, pred, labels=GEN, zero_division=0))

    # ROC curves (one-vs-rest)
    plt.figure()
    for name, m in models.items():
        classes = list(m.classes_)
        proba = m.predict_proba(Xte)[:, [classes.index(g) for g in GEN]]
        for i, g in enumerate(GEN):
            fpr, tpr, _ = roc_curve((yte == g).astype(int), proba[:, i])
            plt.plot(fpr, tpr, label=f"{name}-{g}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("pf02 genre ROC (one-vs-rest)"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/pf02_roc.png"); plt.close()

    names = list(results)
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1); plt.bar(names, [r[0] for r in results.values()]); plt.title("accuracy")
    plt.subplot(1, 2, 2); plt.bar(names, [r[1] for r in results.values()]); plt.title("macro-AUC")
    plt.tight_layout(); plt.savefig(f"{FIG}/pf02_summary.png"); plt.close()
    print("Figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
