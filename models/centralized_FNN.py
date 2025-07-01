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
CONFIG = {
    "t_norm_method": "softmin",
    "lambda_val": 10.0,
    "log_fusion_stats": False
}

class SigmoidAUC(tf.keras.metrics.Metric):
    """Custom AUC metric that applies sigmoid activation before computing AUC."""
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
# 2. UTILITY FUNCTIONS & FUZZY RULE LAYER
##############################################################################
def soft_min_t_norm(membership, lambda_val=10.0):
    weights = tf.nn.softmax(-lambda_val * membership, axis=1)
    return tf.reduce_sum(membership * weights, axis=1, keepdims=True)

class FuzzyRuleLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=1, t_norm_method="softmin", lambda_val=10.0, name="FuzzyRuleLayer"):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.t_norm_method = t_norm_method
        self.lambda_val = lambda_val
        self.m = tf.Variable(tf.zeros([input_dim]), trainable=True, dtype=tf.float32, name="m")
        self.sigma = tf.Variable(tf.ones([input_dim]) * 5.0, trainable=True, dtype=tf.float32, name="sigma")
        self.theta = self.add_weight(shape=(input_dim + 1, output_dim), initializer="random_normal", trainable=True, name="theta")
        # In centralized training, all fuzzy rules remain active (static).
        self.s = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="activation_flag")
    
    def call(self, x):
        membership = tf.maximum(0.0, 1.0 - tf.abs(x - self.m) / tf.maximum(tf.abs(self.sigma), 1e-6))
        firing = soft_min_t_norm(membership, lambda_val=self.lambda_val) * self.s
        batch_size = tf.shape(x)[0]
        x_aug = tf.concat([tf.ones((batch_size, 1), dtype=x.dtype), x], axis=1)
        consequent = tf.matmul(x_aug, self.theta)
        return firing, consequent

##############################################################################
# 3. DATA LOADING & PREPROCESSING
##############################################################################
def load_and_preprocess_data():
    DATA_PATH = r"PATH_TO_FILE\preprocessed_data_enriched.pkl"
    with open(DATA_PATH, "rb") as f:
        preprocessed = pickle.load(f)
    X_ts, X_static, y = preprocessed["X_ts"], preprocessed["X_static"], preprocessed["y"]

    X_ts = np.nan_to_num(X_ts, nan=0.0).astype(np.float32)
    X_static = np.nan_to_num(X_static, nan=0.0).astype(np.float32)
    y = y.astype(np.float32)

    time_steps = X_ts.shape[1]
    ts_dim = X_ts.shape[2]
    static_dim = X_static.shape[1]
    # Repeat static features across time steps and concatenate along the last axis
    X_static_repeated = np.repeat(X_static[:, np.newaxis, :], time_steps, axis=1)
    X_combined = np.concatenate([X_ts, X_static_repeated], axis=-1)
    feature_dim = ts_dim + static_dim

    # Build feature names for SHAP visualization
    ts_feature_names = [
        ......
    ]
    static_feature_names = [
        ......
    ]
    ts_feature_names_full = [f"{var}_t{t+1}" for t in range(time_steps) for var in ts_feature_names]
    input_feature_names = ts_feature_names_full + static_feature_names
    expected_features = (ts_dim * time_steps) + len(static_feature_names)
    assert len(input_feature_names) == expected_features, f"Expected {expected_features} features, got {len(input_feature_names)}"
    
    return X_combined, feature_dim, y, time_steps, input_feature_names

##############################################################################
# 4. FEDFNN MODEL DEFINITION (CENTRALIZED)
##############################################################################
class FedFNNModel(tf.keras.Model):
    def __init__(self, ts_input_dim, static_input_dim, gru_units=128, num_rules=10, 
                 t_norm_method="softmin", lambda_val=10.0, dropout_rate=0.0, name="FedFNNModel"):
        super().__init__(name=name)
        # Process time-series with GRU layers
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=True, name="gru_layer1")
        self.gru2 = tf.keras.layers.GRU(gru_units, return_sequences=True, name="gru_layer2")
        self.attention = tf.keras.layers.Attention()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # Concatenate GRU output with static features
        self.concat_dim = gru_units + static_input_dim
        self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.num_rules = num_rules
        self.fuzzy_rules = [FuzzyRuleLayer(self.concat_dim, t_norm_method=t_norm_method, lambda_val=lambda_val, name=f"FuzzyRule_{i}")
                             for i in range(num_rules)]
        self.refine_dense = tf.keras.layers.Dense(1, activation=None, name="refine_dense")
    
    def call(self, inputs, return_firing=False):
        ts, static = inputs  # ts: [batch, time, ts_input_dim], static: [batch, static_input_dim]
        gru_out = self.gru(ts)
        gru_out = self.gru2(gru_out)
        attention_out = self.attention([gru_out, gru_out])
        attention_pooled = tf.reduce_mean(attention_out, axis=1)
        attention_pooled = self.dropout(attention_pooled)
        fusion = tf.concat([attention_pooled, static], axis=1)
        fusion_norm = self.norm_layer(fusion)
        
        firing_list = []
        consequent_list = []
        for rule in self.fuzzy_rules:
            f_val, g_val = rule(fusion_norm)
            firing_list.append(f_val)
            consequent_list.append(g_val)
        
        firing_all = tf.concat(firing_list, axis=1)
        firing_sum = tf.reduce_sum(firing_all, axis=1, keepdims=True) + 1e-9
        normalized_firing = firing_all / firing_sum
        g_all = tf.concat(consequent_list, axis=1)
        fuzzy_out = tf.reduce_sum(normalized_firing * g_all, axis=1, keepdims=True)
        refined_output = self.refine_dense(fuzzy_out)
        
        if return_firing:
            return refined_output, normalized_firing
        return refined_output

##############################################################################
# 5. HYPERPARAMETER TUNING FUNCTION (Keras Tuner)
##############################################################################
def build_model(hp, ts_input_dim, static_input_dim):
    t_norm_method = hp.Choice("t_norm_method", ["softmin"])
    lambda_val = hp.Float("lambda_val", min_value=5.0, max_value=20.0, step=2.5)
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")
    num_rules = hp.Int("num_rules", min_value=5, max_value=30, step=5)
    gru_units = hp.Int("gru_units", min_value=32, max_value=256, step=32)
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    
    model = FedFNNModel(
        ts_input_dim=ts_input_dim,
        static_input_dim=static_input_dim,
        gru_units=gru_units,
        num_rules=num_rules,
        t_norm_method=t_norm_method,
        lambda_val=lambda_val,
        dropout_rate=dropout_rate
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[SigmoidAUC(name="auc")]
    )
    return model

##############################################################################
# 6. MAIN FUNCTION: CENTRALIZED TRAINING, EVALUATION, & SHAP EXPLANATION
##############################################################################
def main():
    # A) Load and preprocess data
    X_combined, feature_dim, y, time_steps, input_feature_names = load_and_preprocess_data()
    # X_combined shape: (n_samples, time_steps, ts_dim+static_dim)
    
    # Train/Validation Split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Recover original dimensions for time-series and static parts
    with open(r"PATH_TO_FILE\preprocessed_data_enriched.pkl", "rb") as f:
        preprocessed_full = pickle.load(f)
    ts_dim = preprocessed_full["X_ts"].shape[2]
    static_dim = preprocessed_full["X_static"].shape[1]
    
    # Extract time-series and static features from X_combined:
    X_train_ts = X_train[:, :, :ts_dim]       # first ts_dim channels are time-series
    X_val_ts   = X_val[:, :, :ts_dim]
    # For static features, average over time (they are repeated):
    X_static_train = np.mean(X_train[:, :, ts_dim:], axis=1)
    X_static_val   = np.mean(X_val[:, :, ts_dim:], axis=1)
    
    train_ds = tf.data.Dataset.from_tensor_slices(((X_train_ts, X_static_train), y_train)).shuffle(2000).batch(32)
    val_ds   = tf.data.Dataset.from_tensor_slices(((X_val_ts, X_static_val), y_val)).batch(32)
    
    # B) Hyperparameter tuning with Keras Tuner
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, ts_input_dim=ts_dim, static_input_dim=static_dim),
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=40,
        factor=3,
        directory="kt_dir",
        project_name="centralized_fnn_tuning"
    )
    class_weights = {0: 0.58, 1: 3.65}
    tuner.search(
        train_ds,
        epochs=40,
        validation_data=val_ds,
        class_weight=class_weights,
        verbose=1
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters found:")
    print(best_hps.values)
    
    # C) Train the best model (fixed 40 epochs)
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        train_ds,
        epochs=40,
        validation_data=val_ds,
        class_weight=class_weights,
        verbose=1
    )
    val_loss, val_auc = best_model.evaluate(val_ds, verbose=0)
    print(f"Final Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}")
    
    # Plot training history (AUC over epochs)
    plt.figure()
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title("Centralized FNN Training History")
    plt.legend()
    plt.savefig("training_history_fnn.png")
    plt.close()
    print("Training history plot saved as 'training_history_fnn.png'")
    
    # D) Threshold Search & Final Evaluation
    all_logits, all_labels = [], []
    for (x_ts_b, x_static_b), y_b in val_ds:
        logits = best_model((x_ts_b, x_static_b), training=False).numpy().flatten()
        all_logits.append(logits)
        all_labels.append(y_b.numpy().flatten())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    
    platt = LogisticRegression()
    platt.fit(all_logits.reshape(-1, 1), all_labels)
    calibrated_probs = platt.predict_proba(all_logits.reshape(-1, 1))[:, 1]
    
    thresholds = np.arange(0.15, 0.26, 0.005)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (calibrated_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f"Best threshold for F1: {best_t:.3f} (F1 = {best_f1:.4f})")
    preds_best = (calibrated_probs >= best_t).astype(int)
    print("Confusion Matrix:\n", confusion_matrix(all_labels, preds_best))
    print("Classification Report:\n", classification_report(all_labels, preds_best, digits=3, zero_division=0))
    
    fpr, tpr, _ = roc_curve(all_labels, calibrated_probs)
    final_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC (area = {final_auc:.2f})")
    plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Centralized FNN")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_fnn.png")
    plt.close()
    print("ROC curve saved as 'roc_curve_fnn.png'")
    
    # E) SHAP EXPLANATION
    # For SHAP, we need a Functional API wrapper because our model is subclassed.
    # Create two Input layers corresponding to the two inputs: time-series and static.
    ts_input = tf.keras.Input(shape=(X_train.shape[1], ts_dim), name="ts_input")
    static_input = tf.keras.Input(shape=(static_dim,), name="static_input")
    outputs = best_model((ts_input, static_input), training=False)
    functional_model = tf.keras.Model(inputs=[ts_input, static_input], outputs=outputs)
    
    # Use the first 100 validation samples as background.
    background_ts = X_val[:100, :, :ts_dim]
    background_static = np.mean(X_val[:100, :, -static_dim:], axis=1)
    background = [background_ts, background_static]
    
    X_explain_ts = X_val[50:53, :, :ts_dim]
    X_explain_static = np.mean(X_val[50:53, :, -static_dim:], axis=1)
    X_explain_data = [X_explain_ts, X_explain_static]
    
    explainer = shap.GradientExplainer(functional_model, background)
    shap_values = explainer.shap_values(X_explain_data)
    print("Shape of SHAP values (list for each input):", [sv.shape for sv in shap_values])
    # For visualization, we combine the SHAP values by averaging over time for the time-series part.
    shap_values_ts = np.mean(shap_values[0], axis=1)  # shape: (samples, ts_dim)
    # static SHAP values are as is, shape: (samples, static_dim)
    combined_shap = np.concatenate([shap_values_ts, shap_values[1]], axis=1)  # shape: (samples, ts_dim+static_dim)
    
    # For inputs, average time-series over time
    X_explain_ts_avg = np.mean(X_explain_ts, axis=1)  # shape: (samples, ts_dim)
    combined_input = np.concatenate([X_explain_ts_avg, X_explain_static], axis=1)  # shape: (samples, ts_dim+static_dim)
    
    ts_feature_names = [
        ......
    ]
    static_feature_names = [
        ......
    ]
    aggregated_ts_names = [f"avg_{var}" for var in ts_feature_names]
    aggregated_feature_names = np.array(aggregated_ts_names + static_feature_names)
    shap.summary_plot(combined_shap, combined_input, feature_names=aggregated_feature_names, show=False)
    plt.title("SHAP Summary Plot - Centralized FNN")
    plt.savefig("shap_summary_plot_fnn.png")
    plt.close()
    print("SHAP summary plot saved as 'shap_summary_plot_fnn.png'")
    
    # F) Save the final model
    model_save_path = r"PATH_TO_FILE\centralized_fnn"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_model.save(model_save_path, save_format="tf")
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()
