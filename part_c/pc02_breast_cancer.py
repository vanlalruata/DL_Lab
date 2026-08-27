"""part_c / pc02 - Breast Cancer Wisconsin exercise (binary classification).

Compare LogisticRegression vs a small MLPClassifier. For each: train/val/test,
training loss curve, ROC-AUC, confusion matrix, inference latency, model size,
and a parameter-count / time-complexity comparison.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)
import os

FIG = os.path.join(os.path.dirname(__file__), "figures")


def evaluate(name, model, X_tr, y_tr, X_te, y_te, plot_loss=False):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    print(f"\n[{name}] test acc={acc:.4f}  ROC-AUC={auc:.4f}")
    print(confusion_matrix(y_te, y_pred))

    if plot_loss and hasattr(model, "loss_curve_"):
        plt.figure()
        plt.plot(model.loss_curve_, label=name)
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"{name} training loss")
        plt.savefig(os.path.join(FIG, f"pc02_loss_{name}.png")); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    # inference time
    t0 = time.perf_counter()
    _ = model.predict(X_te)
    dt = time.perf_counter() - t0
    # parameter count
    if name.startswith("LR"):
        n_params = model.coef_.size + model.intercept_.size
    else:
        n_params = sum(p.size for p in model.coefs_) + sum(p.size for p in model.intercepts_)
    print(f"  inference: {dt*1000:.3f} ms / {len(X_te)} samples "
          f"({dt/len(X_te)*1e6:.1f} us/sample); params={n_params}")
    return {"name": name, "acc": acc, "auc": auc, "params": n_params, "time_ms": dt * 1000}


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    lr = LogisticRegression(max_iter=1000)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=0)

    results = [evaluate("LR", lr, X_tr, y_tr, X_te, y_te),
               evaluate("MLP", mlp, X_tr, y_tr, X_te, y_te, plot_loss=True)]

    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Breast Cancer ROC comparison")
    plt.plot([0, 1], [0, 1], "k--"); plt.legend()
    plt.savefig(os.path.join(FIG, "pc02_roc.png")); plt.close()

    # complexity comparison bar chart
    names = [r["name"] for r in results]
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1); plt.bar(names, [r["params"] for r in results]); plt.title("params")
    plt.subplot(1, 2, 2); plt.bar(names, [r["time_ms"] for r in results]); plt.title("infer ms")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "pc02_complexity.png")); plt.close()
    print("\nSaved figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
