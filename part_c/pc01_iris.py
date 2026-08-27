"""part_c / pc01 - Iris dataset exercise (classical ML pipeline).

Steps: EDA, train/validate/test split, StandardScaler, LogisticRegression,
learning curve (accuracy vs training size), confusion matrix, per-class ROC
(one-vs-rest), and inference-time / complexity analysis.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)
import os

FIG = os.path.join(os.path.dirname(__file__), "figures")


def explore(X, y, feature_names, target_names):
    print("=== EDA ===")
    print("samples:", X.shape[0], "features:", X.shape[1])
    print("class counts:", np.bincount(y))
    plt.figure(figsize=(8, 5))
    plt.hist(X[:, 0], bins=20)
    plt.title("Feature 0 distribution (sepal length)")
    plt.savefig(os.path.join(FIG, "pc01_feature_hist.png"))
    plt.close()


def main():
    data = load_iris()
    X, y, fnames, tnames = data.data, data.target, data.feature_names, data.target_names
    explore(X, y, fnames, tnames)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    model = LogisticRegression(max_iter=200)
    model.fit(X_tr, y_tr)

    # validation split from training for the learning curve
    train_sz, tr_sc, va_sc = learning_curve(model, X_tr, y_tr, cv=5,
                                            train_sizes=np.linspace(0.1, 1.0, 8),
                                            scoring="accuracy")
    plt.errorbar(train_sz, tr_sc.mean(1), tr_sc.std(1), label="train")
    plt.errorbar(train_sz, va_sc.mean(1), va_sc.std(1), label="val")
    plt.xlabel("training samples"); plt.ylabel("accuracy"); plt.legend()
    plt.title("Iris learning curve (LogisticRegression)")
    plt.savefig(os.path.join(FIG, "pc01_learning_curve.png")); plt.close()

    # test evaluation
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)
    acc = accuracy_score(y_te, y_pred)
    print("Test accuracy:", round(acc, 4))
    print(confusion_matrix(y_te, y_pred))
    print(classification_report(y_te, y_pred, target_names=tnames))

    # ROC one-vs-rest
    auc = roc_auc_score(y_te, y_proba, multi_class="ovr")
    print("Macro ROC-AUC (OvR):", round(auc, 4))
    plt.figure(figsize=(6, 5))
    for i in range(len(tnames)):
        fpr, tpr, _ = roc_curve((y_te == i).astype(int), y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{tnames[i]} (AUC={roc_auc_score((y_te==i).astype(int), y_proba[:,i]):.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Iris ROC (one-vs-rest)")
    plt.legend(); plt.savefig(os.path.join(FIG, "pc01_roc.png")); plt.close()

    # inference time / complexity
    n_params = sum(c.size for c in [model.coef_, model.intercept_])
    t0 = time.perf_counter()
    _ = model.predict(X_te)
    dt = time.perf_counter() - t0
    print(f"Inference: {dt*1000:.3f} ms for {len(X_te)} samples "
          f"({dt/len(X_te)*1e6:.1f} us/sample)")
    print("Model parameters (coef+intercept):", n_params)
    print("Saved figures:", os.listdir(FIG))


if __name__ == "__main__":
    main()
