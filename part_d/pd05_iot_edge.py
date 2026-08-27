"""part_d / pd05 - Lightweight IoT/edge intrusion model comparison.

For resource-constrained edge/IoT gateways, compares a tiny Decision Tree, a
small MLP, and a deeper MLP on synthetic IoT traffic. Reports accuracy, ROC-AUC,
serialized model size (bytes, proxy for flash/RAM footprint), and inference
latency — the key trade-offs for edge deployment.
"""
import os, time, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

FIG = os.path.join(os.path.dirname(__file__), "figures")


def gen_iot(n=5000, seed=0):
    rng = np.random.RandomState(seed)
    n_n = n // 2
    ben = rng.normal(0, 1, (n_n, 6))
    mal = np.column_stack([rng.normal(2.5, 1, n - n_n), rng.normal(-1.5, 1, n - n_n),
                           rng.normal(3, 1, n - n_n), rng.poisson(4, n - n_n),
                           rng.poisson(6, n - n_n), rng.normal(0.5, 0.5, n - n_n)]).astype(float)
    X = np.vstack([ben, mal]); y = np.hstack([np.zeros(n_n), np.ones(n - n_n)])
    return X, y


def size_bytes(model):
    return len(pickle.dumps(model))


def main():
    X, y = gen_iot()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    sc = StandardScaler().fit(X_tr); X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=0),
        "SmallMLP": MLPClassifier(hidden_layer_sizes=(8,), max_iter=150, random_state=0),
        "DeepMLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=0),
    }
    rows = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        acc = (m.predict(X_te) == y_te).mean()
        auc = roc_auc_score(y_te, proba)
        t0 = time.perf_counter(); _ = m.predict(X_te); dt = time.perf_counter() - t0
        sz = size_bytes(m)
        rows.append((name, acc, auc, sz, dt * 1000))
        print(f"{name}: acc={acc:.3f} AUC={auc:.3f} size={sz} bytes infer={dt*1000:.3f} ms")

    names = [r[0] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.subplot(2, 2, 1); plt.bar(names, [r[1] for r in rows]); plt.title("accuracy")
    plt.subplot(2, 2, 2); plt.bar(names, [r[2] for r in rows]); plt.title("ROC-AUC")
    plt.subplot(2, 2, 3); plt.bar(names, [r[3] for r in rows]); plt.title("size (bytes)")
    plt.subplot(2, 2, 4); plt.bar(names, [r[4] for r in rows]); plt.title("infer ms")
    plt.tight_layout(); plt.savefig(f"{FIG}/pd05_edge_tradeoff.png"); plt.close()

    # combined ROC
    plt.figure()
    for name, m in models.items():
        proba = m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_te, proba):.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("pd05 IoT edge model ROC"); plt.legend(); plt.savefig(f"{FIG}/pd05_roc.png"); plt.close()
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
