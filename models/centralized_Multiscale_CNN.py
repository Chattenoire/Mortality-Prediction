import os
import pickle
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import shap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)

##############################################################################
# 1. CONFIGURATION & CUSTOM METRIC
##############################################################################
class SigmoidAUC(tf.keras.metrics.Metric):
    """Custom AUC metric that applies a sigmoid before computing AUC."""
    def __init__(self, name="auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC()
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_prob = tf.sigmoid(y_pred)
        self.auc.update_state(y_true, y_prob, sample_weight)
    def result(self):
        return self.auc.result()
    def reset_state(self):
        self.auc.reset_state()

##############################################################################
# 2. DATA LOADING & PREPROCESSING
##############################################################################
def load_and_preprocess_data():
    DATA_PATH = r"C:\Graduation Project\dataset\preprocessed_data_enriched.pkl"
    with open(DATA_PATH, "rb") as f:
        preprocessed = pickle.load(f)
    X_ts, X_static, y = preprocessed["X_ts"], preprocessed["X_static"], preprocessed["y"]

    X_ts = np.nan_to_num(X_ts, nan=0.0).astype(np.float32)
    X_static = np.nan_to_num(X_static, nan=0.0).astype(np.float32)
    y = y.astype(np.float32)

    # Combine time-series and static features:
    time_steps = X_ts.shape[1]
    ts_dim = X_ts.shape[2]
    static_dim = X_static.shape[1]
    # Repeat static features along the time axis
    X_static_repeated = np.repeat(X_static[:, np.newaxis, :], time_steps, axis=1)
    # Concatenate along the last axis
    X_combined = np.concatenate([X_ts, X_static_repeated], axis=-1)  # shape: (n_samples, time_steps, ts_dim+static_dim)
    feature_dim = ts_dim + static_dim

    # Build feature names for SHAP visualization:
    ts_feature_names = ['HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2', 'Temperature',
                          'WBC', 'Hemoglobin', 'Platelets', 'Sodium', 'Potassium', 'Chloride', 'BUN',
                          'Creatinine', 'Glucose', 'Arterial_pH', 'Arterial_Lactate']
    static_feature_names = [
        'age', 'gender', 'Myocardial_Infarction', 'Congestive_Heart_Failure', 'Peripheral_Vascular_Disease',
        'Cerebrovascular_Disease', 'Dementia', 'Chronic_Pulmonary_Disease', 'Rheumatologic_Disease',
        'Peptic_Ulcer_Disease', 'Mild_Liver_Disease', 'Diabetes', 'Diabetes_with_Complications',
        'Hemiplegia', 'Moderate_to_Severe_Renal_Disease', 'Any_Malignancy',
        'Moderate_to_Severe_Liver_Disease', 'Metastatic_Solid_Tumor', 'AIDS'
    ]
    ts_feature_names_full = [f"{var}_t{t+1}" for t in range(time_steps) for var in ts_feature_names]
    input_feature_names = ts_feature_names_full + static_feature_names
    expected_features = (ts_dim * time_steps) + len(static_feature_names)
    assert len(input_feature_names) == expected_features, f"Expected {expected_features} features, got {len(input_feature_names)}"
    
    return X_combined, feature_dim, y, time_steps, input_feature_names

##############################################################################
# 3. BUILD MULTI-SCALE CNN MODEL WITH HYPERPARAMETER TUNING
##############################################################################
def build_multiscale_cnn_model(hp, input_shape):
    # Hyperparameters for each branch
    conv_filters_1 = hp.Int("conv_filters_1", min_value=16, max_value=64, step=16)
    conv_filters_2 = hp.Int("conv_filters_2", min_value=16, max_value=64, step=16)
    conv_filters_3 = hp.Int("conv_filters_3", min_value=16, max_value=64, step=16)
    kernel_size1 = hp.Choice("kernel_size1", values=[3, 5])
    kernel_size2 = hp.Choice("kernel_size2", values=[3, 5])
    kernel_size3 = hp.Choice("kernel_size3", values=[3, 5])
    # Additional branch after concatenation
    conv_filters_final = hp.Int("conv_filters_final", min_value=32, max_value=128, step=32)
    
    dense_units = hp.Int("dense_units", min_value=64, max_value=256, step=64)
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")
    
    inputs = tf.keras.Input(shape=input_shape, name="combined_input")
    
    # Multi-scale branches:
    branch1 = tf.keras.layers.Conv1D(filters=conv_filters_1, kernel_size=kernel_size1, activation="relu", padding="same")(inputs)
    branch1 = tf.keras.layers.MaxPooling1D(pool_size=2)(branch1)
    
    branch2 = tf.keras.layers.Conv1D(filters=conv_filters_2, kernel_size=kernel_size2, activation="relu", padding="same")(inputs)
    branch2 = tf.keras.layers.MaxPooling1D(pool_size=2)(branch2)
    
    branch3 = tf.keras.layers.Conv1D(filters=conv_filters_3, kernel_size=kernel_size3, activation="relu", padding="same")(inputs)
    branch3 = tf.keras.layers.MaxPooling1D(pool_size=2)(branch3)
    
    # Concatenate branches
    concatenated = tf.keras.layers.Concatenate()([branch1, branch2, branch3])
    
    # Additional convolution on concatenated features
    x = tf.keras.layers.Conv1D(filters=conv_filters_final, kernel_size=3, activation="relu", padding="same")(concatenated)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation=None, name="logits")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[SigmoidAUC(name="auc")]
    )
    return model

##############################################################################
# 4. HYPERPARAMETER TUNING WITH KERAS TUNER
##############################################################################
def tune_model(train_ds, val_ds, input_shape):
    tuner = kt.Hyperband(
        lambda hp: build_multiscale_cnn_model(hp, input_shape),
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=40,
        factor=3,
        directory="kt_dir",
        project_name="centralized_multiscale_cnn_tuning"
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=3,
        restore_best_weights=True,
        mode='max'
    )
    class_weights = {0: 0.58, 1: 3.65}
    tuner.search(
        train_ds,
        epochs=40,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters found:")
    for param, val in best_hps.values.items():
        print(f"  {param}: {val}")
    return best_hps

##############################################################################
# 5. MAIN FUNCTION: TRAINING, EVALUATION, & SHAP
##############################################################################
def main():
    # A) Load and preprocess data
    X_combined, feature_dim, y, time_steps, input_feature_names = load_and_preprocess_data()
    
    # Stratified Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(2000).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
    
    # B) Define input shape: (time_steps, feature_dim)
    input_shape = (X_train.shape[1], feature_dim)
    
    # C) Hyperparameter tuning
    best_hps = tune_model(train_ds, val_ds, input_shape)
    
    # D) Train the best model using tuned hyperparameters
    best_model = build_multiscale_cnn_model(best_hps, input_shape)
    history = best_model.fit(
        train_ds,
        epochs=40,
        validation_data=val_ds,
        class_weight={0: 0.58, 1: 3.65},
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True)],
        verbose=1
    )
    val_loss, val_auc = best_model.evaluate(val_ds, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}")
    
    # Plot training history (AUC over epochs)
    plt.figure()
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title("Centralized Multi-scale CNN Training History")
    plt.legend()
    plt.savefig("training_history_multiscale_cnn.png")
    plt.close()
    print("Training history plot saved as 'training_history_multiscale_cnn.png'")
    
    # E) Threshold Search & ROC Curve
    all_logits, all_labels = [], []
    for X_b, y_b in val_ds:
        logits = best_model(X_b, training=False).numpy().flatten()
        all_logits.append(logits)
        all_labels.append(y_b.numpy().flatten())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    platt = LogisticRegression()
    platt.fit(all_logits.reshape(-1, 1), all_labels)
    calibrated_probs = platt.predict_proba(all_logits.reshape(-1, 1))[:, 1]
    thresholds = np.linspace(0.0, 1.0, 101)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (calibrated_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f"Best threshold: {best_t:.3f}, F1 = {best_f1:.4f}")
    preds_best = (calibrated_probs >= best_t).astype(int)
    print("Confusion Matrix:\n", confusion_matrix(all_labels, preds_best))
    print("Classification Report:\n", classification_report(all_labels, preds_best, digits=3, zero_division=0))
    
    fpr, tpr, _ = roc_curve(all_labels, calibrated_probs)
    final_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {final_auc:.2f})")
    plt.plot([0,1],[0,1], color="navy", lw=2, linestyle="--")
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Centralized Multi-scale CNN")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_multiscale_cnn.png")
    plt.close()
    print("ROC curve saved as 'roc_curve_multiscale_cnn.png'")
    
    # F) SHAP Explanation
    background = X_val[:100]  # Raw 3D data: (n_samples, time_steps, feature_dim)
    X_explain = X_val[50:53]   # Raw 3D data: (samples, time_steps, feature_dim)
    explainer = shap.GradientExplainer(best_model, background)
    shap_values = explainer.shap_values(X_explain)
    print("Shape of SHAP values:", np.array(shap_values).shape)  # Expected: (samples, time_steps, feature_dim, 1)
    # Average over the time axis (axis=1) and squeeze the last dimension
    shap_values_avg = np.mean(shap_values, axis=1).squeeze(-1)  # (samples, feature_dim)
    X_explain_avg = np.mean(X_explain, axis=1)                   # (samples, feature_dim)
    shap.summary_plot(shap_values_avg, X_explain_avg, feature_names=input_feature_names, show=False)
    plt.title("SHAP Summary Plot - Multi-scale CNN")
    plt.savefig("shap_summary_plot_multiscale_cnn.png")
    plt.close()
    print("SHAP summary plot saved as 'shap_summary_plot_multiscale_cnn.png'")
    
    # G) Save the final model
    model_save_path = r"C:\Graduation Project\saved models\centralized_multiscale_cnn"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_model.save(model_save_path, save_format="tf")
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()
