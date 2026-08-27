"""part_d / pd01 - Wireless network intrusion detection (802.11 features).

Synthetic Wi-Fi frame features (RSSI, SNR, packet rate, auth failures, replay
count) labelled normal vs attack (spoofing/DoS). Train MLP, plot accuracy/loss,
ROC, and inference latency for an edge AP use-case.
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.neural_network import MLPClassifier

FIG = os.path.join(os.path.dirname(__file__), "figures")


def gen_wireless(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    # normal: healthy RSSI/SNR, low auth failures, bursty-but-bounded rate
    n_n = n // 2
    normal = np.column_stack([
        rng.normal(-45, 6, n_n),        # RSSI (dBm)
        rng.normal(28, 4, n_n),         # SNR
        rng.gamma(2, 30, n_n),          # packet rate
        rng.poisson(0.2, n_n),          # auth failures
        rng.poisson(0.1, n_n),          # replay count
    ])
    # attack: weaker RSSI spoof, high failures/replay, flooding rate
    attack = np.column_stack([
        rng.normal(-70, 10, n - n_n),
        rng.normal(12, 5, n - n_n),
        rng.gamma(2, 120, n - n_n),
        rng.poisson(6, n - n_n),
        rng.poisson(9, n - n_n),
    ])
    X = np.vstack([normal, attack])
    y = np.hstack([np.zeros(n_n), np.ones(n - n_n)])
    return X, y


def main():
    X, y = gen_wireless()
    print("=== EDA ===  samples:", X.shape[0], "features:", X.shape[1],
          "positives(attack):", int(y.sum()))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    sc = StandardScaler().fit(X_tr); X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=0)
    model.fit(X_tr, y_tr)

    if hasattr(model, "loss_curve_"):
        plt.figure(); plt.plot(model.loss_curve_); plt.xlabel("epoch"); plt.ylabel("loss")
        plt.title("pd01 training loss"); plt.savefig(f"{FIG}/pd01_loss.png"); plt.close()

    yp = model.predict(X_te); proba = model.predict_proba(X_te)[:, 1]
    print("Test acc:", round(accuracy_score(y_te, yp), 4))
    print(confusion_matrix(y_te, yp))
    print(classification_report(y_te, yp, target_names=["normal", "attack"]))

    auc = roc_auc_score(y_te, proba)
    fpr, tpr, _ = roc_curve(y_te, proba)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("pd01 Wireless IDS ROC"); plt.legend()
    plt.savefig(f"{FIG}/pd01_roc.png"); plt.close()

    t0 = time.perf_counter(); _ = model.predict(X_te)
    dt = time.perf_counter() - t0
    n_params = sum(p.size for p in model.coefs_) + sum(p.size for p in model.intercepts_)
    print(f"Inference: {dt*1000:.3f} ms / {len(X_te)} flows "
          f"({dt/len(X_te)*1e6:.1f} us/flow); params={n_params}")
    print("Saved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
