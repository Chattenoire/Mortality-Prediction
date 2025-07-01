import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import shap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)

##############################################################################
# 1. GLOBAL CONFIGURATION & BEST HYPERPARAMETERS
##############################################################################
CONFIG = {
    "num_clients": 5,
    "federated_rounds": 30,
    "local_epochs": 2
}

# Best hyperparameters from your centralized multi-scale CNN run
BEST_HPS = {
    "conv_filters_1": 48,
    "conv_filters_2": 16,
    "conv_filters_3": 64,
    "kernel_size1": 3,
    "kernel_size2": 5,
    "kernel_size3": 3,
    "conv_filters_final": 128,
    "dense_units": 64,
    "dropout_rate": 0.2,
    "learning_rate": 5.785373543760055e-05
}

# Class weights for imbalanced data
CLASS_WEIGHTS = {0: 0.58, 1: 3.65}

##############################################################################
# 2. CUSTOM METRIC: SIGMOIDAUC
##############################################################################
class SigmoidAUC(tf.keras.metrics.Metric):
    """Custom AUC metric that applies sigmoid before computing AUC."""
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
# 3. DATA LOADING & PREPROCESSING
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
    X_static_repeated = np.repeat(X_static[:, np.newaxis, :], time_steps, axis=1)
    X_combined = np.concatenate([X_ts, X_static_repeated], axis=-1)
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
# 4. MODEL DEFINITION: MULTI-SCALE CNN
##############################################################################
def create_multiscale_cnn_model(input_shape, hps):
    inputs = tf.keras.Input(shape=input_shape, name="combined_input")
    
    # Branch 1: Conv1D with kernel_size1
    branch1 = tf.keras.layers.Conv1D(filters=hps["conv_filters_1"],
                                     kernel_size=hps["kernel_size1"],
                                     activation="relu",
                                     padding="same")(inputs)
    branch1 = tf.keras.layers.MaxPooling1D(pool_size=2)(branch1)
    
    # Branch 2: Conv1D with kernel_size2
    branch2 = tf.keras.layers.Conv1D(filters=hps["conv_filters_2"],
                                     kernel_size=hps["kernel_size2"],
                                     activation="relu",
                                     padding="same")(inputs)
    branch2 = tf.keras.layers.MaxPooling1D(pool_size=2)(branch2)
    
    # Branch 3: Conv1D with kernel_size3
    branch3 = tf.keras.layers.Conv1D(filters=hps["conv_filters_3"],
                                     kernel_size=hps["kernel_size3"],
                                     activation="relu",
                                     padding="same")(inputs)
    branch3 = tf.keras.layers.MaxPooling1D(pool_size=2)(branch3)
    
    # Concatenate multi-scale branches
    concatenated = tf.keras.layers.Concatenate()([branch1, branch2, branch3])
    
    # Additional convolution on concatenated features
    x = tf.keras.layers.Conv1D(filters=hps["conv_filters_final"],
                               kernel_size=3,
                               activation="relu",
                               padding="same")(concatenated)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hps["dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(hps["dropout_rate"])(x)
    outputs = tf.keras.layers.Dense(1, activation=None, name="logits")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hps["learning_rate"]),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[SigmoidAUC(name="auc")]
    )
    return model

##############################################################################
# 5. FEDERATED AGGREGATION (FedAvg)
##############################################################################
def aggregate_models(global_model, client_models, client_sizes):
    global_weights = global_model.get_weights()
    total_size = sum(client_sizes)
    for i in range(len(global_weights)):
        global_weights[i] = np.zeros_like(global_weights[i])
    for client_model, size in zip(client_models, client_sizes):
        weight_factor = size / total_size
        for gw, cw in zip(global_weights, client_model.get_weights()):
            gw += cw * weight_factor
    global_model.set_weights(global_weights)
    return global_model

##############################################################################
# 6. MAIN FUNCTION: FEDERATED TRAINING, EVALUATION, ROC-AUC, & SHAP
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
    
    # B) Define input shape for the CNN model
    input_shape = (X_train.shape[1], feature_dim)  # (time_steps, ts_dim+static_dim)
    
    # C) Initialize global model using BEST_HPS
    global_model = create_multiscale_cnn_model(input_shape, BEST_HPS)
    dummy_input = tf.zeros((1, X_train.shape[1], feature_dim), dtype=tf.float32)
    global_model(dummy_input)  # Build model
    
    # D) Federated Training Loop
    num_clients = CONFIG["num_clients"]
    train_size = X_train.shape[0]
    client_size = train_size // num_clients
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = (i+1)*client_size if i < num_clients - 1 else train_size
        X_client = X_train[start_idx:end_idx]
        y_client = y_train[start_idx:end_idx]
        ds = tf.data.Dataset.from_tensor_slices((X_client, y_client)).shuffle(1000).batch(32)
        client_datasets.append(ds)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
    
    auc_over_time = []
    for round_num in range(CONFIG["federated_rounds"]):
        print(f"\nRound {round_num+1}/{CONFIG['federated_rounds']}")
        client_models = []
        client_sizes = []
        for ds in client_datasets:
            client_model = create_multiscale_cnn_model(input_shape, BEST_HPS)
            client_model(dummy_input)
            client_model.set_weights(global_model.get_weights())
            client_model.fit(
                ds,
                epochs=CONFIG["local_epochs"],
                class_weight=CLASS_WEIGHTS,
                verbose=0
            )
            client_models.append(client_model)
            client_sizes.append(sum(1 for _ in ds))
        global_model = aggregate_models(global_model, client_models, client_sizes)
        val_loss, val_auc = global_model.evaluate(val_dataset, verbose=0)
        auc_over_time.append(val_auc)
        print(f"Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
    
    # Plot AUC over federated rounds
    plt.figure()
    plt.plot(range(1, len(auc_over_time)+1), auc_over_time, marker='o')
    plt.xlabel("Federated Round")
    plt.ylabel("Validation AUC")
    plt.title("AUC over Federated Rounds - FedAvg Multi-scale CNN")
    plt.savefig("auc_over_time_fedavg_multiscale_cnn.png")
    plt.close()
    print("AUC plot saved as 'auc_over_time_fedavg_multiscale_cnn.png'")
    
    # E) Threshold Search & ROC Curve
    all_logits, all_labels = [], []
    for X_b, y_b in val_dataset:
        logits = global_model(X_b, training=False).numpy().flatten()
        all_logits.append(logits)
        all_labels.append(y_b.numpy().flatten())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    platt = LogisticRegression()
    platt.fit(all_logits.reshape(-1,1), all_labels)
    calibrated_probs = platt.predict_proba(all_logits.reshape(-1,1))[:, 1]
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
    plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - FedAvg Multi-scale CNN")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_fedavg_multiscale_cnn.png")
    plt.close()
    print("ROC curve saved as 'roc_curve_fedavg_multiscale_cnn.png'")
    
    # F) SHAP EXPLANATION
    background = X_val[:100]  # Use raw 3D validation data as background
    X_explain = X_val[50:53]   # Select a few raw 3D samples for explanation
    explainer = shap.GradientExplainer(global_model, background)
    shap_values = explainer.shap_values(X_explain)
    print("Shape of SHAP values:", np.array(shap_values).shape)  # Expected: (samples, time_steps, feature_dim, 1)
    # Average over the time axis (axis=1) and squeeze the last dimension
    shap_values_avg = np.mean(shap_values, axis=1).squeeze(-1)  # (samples, feature_dim)
    X_explain_avg = np.mean(X_explain, axis=1)                   # (samples, feature_dim)
    shap.summary_plot(shap_values_avg, X_explain_avg, feature_names=input_feature_names, show=False)
    plt.title("SHAP Summary Plot - FedAvg Multi-scale CNN")
    plt.savefig("shap_summary_plot_fedavg_multiscale_cnn.png")
    plt.close()
    print("SHAP summary plot saved as 'shap_summary_plot_fedavg_multiscale_cnn.png'")
    
    # G) Save the final global model
    model_save_path = r"C:\Graduation Project\saved models\federated_fedavg_multiscale_cnn"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    global_model.save(model_save_path, save_format="tf")
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()
