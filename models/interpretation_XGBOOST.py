import os
import pickle
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import shap

# ------------------------------
# 1. Load & Preprocess Data
# ------------------------------
def load_and_preprocess_data():
    DATA_PATH = r"C:\Graduation Project\dataset\preprocessed_data_enriched.pkl"
    with open(DATA_PATH, "rb") as f:
        preprocessed = pickle.load(f)
    X_ts = preprocessed["X_ts"]      # shape: (N, T, ts_dim)
    X_static = preprocessed["X_static"]  # shape: (N, static_dim)
    y = preprocessed["y"]

    # Handle NaNs and ensure correct data types
    X_ts = np.nan_to_num(X_ts, nan=0.0).astype(np.float32)
    X_static = np.nan_to_num(X_static, nan=0.0).astype(np.float32)
    y = y.astype(np.float32)
    
    N, T, ts_dim = X_ts.shape
    static_dim = X_static.shape[1]

    # Flatten time-series part and then append static features
    X_ts_flat = X_ts.reshape((N, T * ts_dim))
    X_combined = np.concatenate([X_ts_flat, X_static], axis=1)  # shape: (N, T*ts_dim + static_dim)
    return X_combined, y

def main():
    # Load data
    X, y = load_and_preprocess_data()
    print("X shape:", X.shape, "y shape:", y.shape)
    
    # Train/Validation Split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train samples:", len(y_train), "Validation samples:", len(y_val))
    
    # ------------------------------
    # 2. Train XGBoost Model
    # ------------------------------
    # Specify eval_metric in the constructor and remove unsupported kwargs.
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",  # or "gpu_hist" if using GPU
        objective="binary:logistic",
        eval_metric="auc"
    )
    
    # Remove early_stopping_rounds and evals_result because they're not supported.
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )
    
    # ------------------------------
    # 3. Plot AUC Over Boosting Rounds (if available)
    # ------------------------------
    try:
        evals_result = xgb_model.evals_result()
        if "validation_1" in evals_result and "auc" in evals_result["validation_1"]:
            val_auc_log = evals_result["validation_1"]["auc"]
            plt.figure()
            plt.plot(val_auc_log, label="Validation AUC")
            plt.xlabel("Boosting Round")
            plt.ylabel("AUC")
            plt.title("AUC over Boosting Rounds (XGBoost)")
            plt.legend()
            plt.savefig("auc_over_time_xgb.png")
            plt.close()
            print("AUC over time plot saved as 'auc_over_time_xgb.png'")
        else:
            print("AUC tracking data not found in evals_result.")
    except Exception as e:
        print("Error retrieving evals_result:", e)
    
    # ------------------------------
    # 4. Threshold Search & ROC Curve
    # ------------------------------
    val_preds_proba = xgb_model.predict_proba(X_val)[:, 1]
    platt = LogisticRegression()
    platt.fit(val_preds_proba.reshape(-1, 1), y_val)
    calibrated_probs = platt.predict_proba(val_preds_proba.reshape(-1, 1))[:, 1]
    
    thresholds = np.linspace(0.0, 1.0, 101)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (calibrated_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f"Best threshold: {best_t:.3f}, F1 = {best_f1:.4f}")
    preds_best = (calibrated_probs >= best_t).astype(int)
    print("Confusion Matrix:\n", confusion_matrix(y_val, preds_best))
    print("Classification Report:\n", classification_report(y_val, preds_best, digits=3, zero_division=0))
    
    fpr, tpr, _ = roc_curve(y_val, calibrated_probs)
    final_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {final_auc:.2f})")
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - XGBoost")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_xgb.png")
    plt.close()
    print("ROC curve saved as 'roc_curve_xgb.png'")
    
    # ------------------------------
    # 5. SHAP Explanation
    # ------------------------------
    bg_size = 100
    X_bg = X_val[:bg_size]
    X_explain = X_val[bg_size:bg_size+3]
    explainer = shap.TreeExplainer(xgb_model, data=X_bg)
    shap_values = explainer.shap_values(X_explain)
    print("Shape of SHAP values:", shap_values.shape)
    shap.summary_plot(shap_values, X_explain, show=False)
    plt.title("SHAP Summary Plot - XGBoost")
    plt.savefig("shap_summary_plot_xgb.png")
    plt.close()
    print("SHAP summary plot saved as 'shap_summary_plot_xgb.png'")
    
    # ------------------------------
    # 6. Save the XGBoost Model
    # ------------------------------
    MODEL_SAVE_PATH_XGB = os.path.join(r"C:\Graduation Project\saved models\interpretation_XGBOOST", "interpretation_XGBOOST.pkl")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH_XGB), exist_ok=True)
    joblib.dump(xgb_model, MODEL_SAVE_PATH_XGB)
    print(f"XGBoost model saved as '{MODEL_SAVE_PATH_XGB}'")
    
if __name__ == "__main__":
    main()
