"""part_d / pd02 - SDN DDoS detection using flow statistics.

Synthesises OpenFlow-style flow features (packet count, byte count, flow
duration, inter-arrival jitter, distinct ports) for benign vs DDoS floods.
Compares Logistic Regression vs MLP and Random Forest; reports ROC-AUC,
training loss, confusion matrix, and inference latency (controller use-case).
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve)

FIG = os.path.join(os.path.dirname(__file__), "figures")


def gen_sdn(n=4000, seed=0):
    rng = np.random.RandomState(seed)
    n_n = n // 2
    ben = np.column_stack([
        rng.gamma(2, 50, n_n),      # packet count
        rng.gamma(2, 800, n_n),     # byte count
        rng.gamma(3, 20, n_n),      # flow duration
        rng.normal(5, 1.5, n_n),    # inter-arrival jitter (ms)
        rng.randint(1, 20, n_n),    # distinct dst ports
    ])
    mal = np.column_stack([
        rng.gamma(2, 400, n - n_n), # floods: huge packet counts
        rng.gamma(2, 6000, n - n_n),
        rng.gamma(1, 2, n - n_n),   # very short duration
        rng.normal(0.5, 0.3, n - n_n),
        rng.randint(1, 3, n - n_n),
    ])
    X = np.vstack([ben, mal]); y = np.hstack([np.zeros(n_n), np.ones(n - n_n)])
    return X, y


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te); proba = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, yp); auc = roc_auc_score(y_te, proba)
    print(f"[{name}] acc={acc:.4f} AUC={auc:.4f}")
    print(confusion_matrix(y_te, yp))
    t0 = time.perf_counter(); _ = model.predict(X_te); dt = time.perf_counter() - t0
    if name == "LR":
        n_params = model.coef_.size + model.intercept_.size
    elif name == "MLP":
        n_params = sum(p.size for p in model.coefs_) + sum(p.size for p in model.intercepts_)
    else:
        n_params = sum(t.tree_.node_count for t in model.estimators_)
    print(f"  infer {dt*1000:.3f} ms/{len(X_te)} flows; params/nodes={n_params}")
    return name, auc, dt * 1000


def main():
    X, y = gen_sdn()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    sc = StandardScaler().fit(X_tr); X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    models = {"LR": LogisticRegression(max_iter=1000),
              "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=0),
              "RF": RandomForestClassifier(n_estimators=100, random_state=0)}
    res = [evaluate(n, m, X_tr, y_tr, X_te, y_te) for n, m in models.items()]

    plt.figure()
    for n, _, _ in res:
        m = models[n]
        proba = m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        plt.plot(fpr, tpr, label=f"{n} AUC={roc_auc_score(y_te, proba):.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("pd02 SDN DDoS ROC comparison"); plt.legend(); plt.savefig(f"{FIG}/pd02_roc.png"); plt.close()

    names = [r[0] for r in res]
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1); plt.bar(names, [r[1] for r in res]); plt.title("ROC-AUC")
    plt.subplot(1, 2, 2); plt.bar(names, [r[2] for r in res]); plt.title("infer ms")
    plt.tight_layout(); plt.savefig(f"{FIG}/pd02_complexity.png"); plt.close()
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
