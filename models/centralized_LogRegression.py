import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    auc
)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)

##############################################################################
# 1. DATA LOADING
##############################################################################
def load_and_preprocess_data():
    DATA_PATH = r"PATH_TO_FILE\preprocessed_data_enriched.pkl"
    with open(DATA_PATH, "rb") as f:
        preprocessed = pickle.load(f)

    X_ts     = np.nan_to_num(preprocessed["X_ts"],   nan=0.0).astype(np.float32)
    X_static = np.nan_to_num(preprocessed["X_static"], nan=0.0).astype(np.float32)
    y        = preprocessed["y"].astype(int)  # ensure integer labels

    # —— flatten the time‑series then concatenate static features ——
    n_samples, time_steps, ts_dim = X_ts.shape
    X_ts_flat = X_ts.reshape(n_samples, time_steps * ts_dim)
    X_flat    = np.concatenate([X_ts_flat, X_static], axis=1)

    # scale features for LR
    scaler = StandardScaler().fit(X_flat)
    X_flat = scaler.transform(X_flat).astype(np.float32)
    return X_flat, y

##############################################################################
# 2. TRAIN / VALIDATE
##############################################################################
def train_and_evaluate():
    X, y = load_and_preprocess_data()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # class imbalance weighting (~5× positive)
    w = {0: 1.0, 1: 5.0}

    base_clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=200,
        class_weight=w
    )

    # Platt‑scaling (sigmoid) calibration on the fly
    clf = CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=5
    )

    clf.fit(X_tr, y_tr)

    # ── metrics ────────────────────────────────────────────────────────────
    prob_val    = clf.predict_proba(X_val)[:, 1]
    thresholds  = np.linspace(0, 1, 101)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (prob_val >= t).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds_best = (prob_val >= best_t).astype(int)
    print(f"Best threshold = {best_t:.3f}, F1 = {best_f1:.4f}")
    print("Confusion:\n", confusion_matrix(y_val, preds_best))
    print("Report:\n", classification_report(y_val, preds_best, digits=3, zero_division=0))

    fpr, tpr, _ = roc_curve(y_val, prob_val)
    roc_auc = auc(fpr, tpr)
    print(f"Validation AUC = {roc_auc:.4f}")

    # ROC plot
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "--", lw=2)
    plt.xlabel("False‑Positive Rate")
    plt.ylabel("True‑Positive Rate")
    plt.title("ROC – Centralised Logistic Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve_logreg.png")
    plt.close()
    print("ROC curve saved as 'roc_curve_logreg.png'")

    # save calibrated model
    save_dir = r"PATH_TO_FILE\centralized_logreg"
    os.makedirs(save_dir, exist_ok=True)
    import joblib
    joblib.dump(clf, os.path.join(save_dir, "logreg_calibrated.pkl"))
    print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    train_and_evaluate()
