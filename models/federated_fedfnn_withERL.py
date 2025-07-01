import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression, LassoLarsIC
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import shap
import matplotlib.pyplot as plt

from time_series_augmentation import augment_minority_randomwarp

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ------------------------------
# Global Configuration
# ------------------------------
CONFIG = {
    "num_clients": 5,
    "federated_rounds": 30,
    "local_epochs": 2,
}

# Best hyperparameters
BEST_HPS = {
    "t_norm_method": "softmin",
    "lambda_val": 12.5,
    "learning_rate": 0.001757862999508398,
    "num_rules": 15,
    "gru_units": 224,
    "dropout_rate": 0.2
}

# Feature Mapping
static_feature_names = [
    ......
]

ts_feature_names = [
    ......
]

static_feature_count = len(static_feature_names)

# ------------------------------
# Custom Metric: SigmoidAUC
# ------------------------------
class SigmoidAUC(tf.keras.metrics.Metric):
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

# ------------------------------
# T-Norm Function
# ------------------------------
def soft_min_t_norm(membership, lambda_val=10.0):
    weights = tf.nn.softmax(-lambda_val * membership, axis=1)
    return tf.reduce_sum(membership * weights, axis=1, keepdims=True)

# ------------------------------
# FuzzyRuleLayer
# ------------------------------
class FuzzyRuleLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=1, t_norm_method="softmin", lambda_val=10.0, name="FuzzyRuleLayer"):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.t_norm_method = t_norm_method
        self.lambda_val = lambda_val
        self.m = tf.Variable(initial_value=tf.zeros([input_dim]), trainable=True, dtype=tf.float32, name="m")
        self.sigma = tf.Variable(initial_value=tf.ones([input_dim]) * 5.0, trainable=True, dtype=tf.float32, name="sigma")
        self.theta = self.add_weight(shape=(input_dim + 1, output_dim), initializer="random_normal", trainable=True, name="theta")
        self.s = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="activation_flag")
    
    def call(self, x):
        membership = tf.maximum(0.0, 1.0 - tf.abs(x - self.m) / tf.maximum(tf.abs(self.sigma), 1e-6))
        firing = soft_min_t_norm(membership, lambda_val=self.lambda_val) * self.s
        batch_size = tf.shape(x)[0]
        x_aug = tf.concat([tf.ones((batch_size, 1), dtype=x.dtype), x], axis=1)
        consequent = tf.matmul(x_aug, self.theta)
        return firing, consequent

# ------------------------------
# FedFNNModel
# ------------------------------
class FedFNNModel(tf.keras.Model):
    def __init__(self, ts_input_dim, static_input_dim, gru_units=128, num_rules=10, 
                 t_norm_method="softmin", lambda_val=10.0, dropout_rate=0.0, name="FedFNNModel"):
        super().__init__(name=name)
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=True, name="gru_layer1")
        self.gru2 = tf.keras.layers.GRU(gru_units, return_sequences=True, name="gru_layer2")
        self.attention = tf.keras.layers.Attention()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.concat_dim = gru_units + static_input_dim
        self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.num_rules = num_rules
        self.fuzzy_rules = [
            FuzzyRuleLayer(self.concat_dim, t_norm_method=t_norm_method, lambda_val=lambda_val, name=f"FuzzyRule_{i}")
            for i in range(num_rules)
        ]
        self.refine_dense = tf.keras.layers.Dense(1, activation=None, name="refine_dense")
    
    def call(self, inputs, return_firing=False):
        ts, static = inputs
        gru_out = self.gru(ts)
        gru_out = self.gru2(gru_out)
        attention_out = self.attention([gru_out, gru_out])
        attention_pooled = tf.reduce_mean(attention_out, axis=1)
        attention_pooled = self.dropout(attention_pooled)
        fusion = tf.concat([attention_pooled, static], axis=1)
        fusion_norm = self.norm_layer(fusion)
        
        firing_list, consequent_list = [], []
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

# ------------------------------
# Create Model with Best Hyperparameters
# ------------------------------
def create_fuzzy_model(ts_input_dim, static_input_dim, hps):
    model = FedFNNModel(
        ts_input_dim=ts_input_dim,
        static_input_dim=static_input_dim,
        gru_units=hps["gru_units"],
        num_rules=hps["num_rules"],
        t_norm_method=hps["t_norm_method"],
        lambda_val=hps["lambda_val"],
        dropout_rate=hps["dropout_rate"]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hps["learning_rate"]),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[SigmoidAUC(name="auc")]
    )
    return model

# ------------------------------
# Evolutionary Rule Learning
# ------------------------------
def evolve_rules(model, client_datasets, contrib_threshold=0.05, round_num=None, total_rounds=30):
    firing_strengths_all = []
    for client_data in client_datasets:
        for (x_ts_b, x_stat_b), _ in client_data:
            _, firing = model((x_ts_b, x_stat_b), return_firing=True)
            firing_strengths_all.append(tf.reduce_mean(firing, axis=0).numpy())
    
    avg_firing = np.mean(firing_strengths_all, axis=0)
    if round_num is not None:
        print(f"Round {round_num + 1} - Average firing strengths: {avg_firing}")
    
    min_firing = np.min(avg_firing)
    max_firing = np.max(avg_firing)
    normalized_firing = (avg_firing - min_firing) / (max_firing - min_firing + 1e-9)
    if round_num is not None:
        print(f"Round {round_num + 1} - Normalized firing strengths: {normalized_firing}")
    
    dynamic_threshold = max(0.05, 0.2 - (round_num / total_rounds) * 0.15)
    active_rules = normalized_firing >= dynamic_threshold
    min_rules = 5
    if np.sum(active_rules) < min_rules:
        inactive_indices = np.where(~active_rules)[0]
        reactivate_idx = np.random.choice(inactive_indices, min_rules - np.sum(active_rules), replace=False)
        active_rules[reactivate_idx] = True
    
    for i, rule in enumerate(model.fuzzy_rules):
        rule.s.assign(1.0 if active_rules[i] else 0.0)
    return active_rules

# ------------------------------
# Federated Aggregation
# ------------------------------
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

# ------------------------------
# Interpret Fuzzy Rules
# ------------------------------
def interpret_fuzzy_rules(model, rule_feature_names, top_features=None):
    active_rules = []
    for i, rule in enumerate(model.fuzzy_rules):
        if rule.s.numpy() == 1.0:
            m = rule.m.numpy()
            sigma = rule.sigma.numpy()
            theta = rule.theta.numpy()
            rule_str = f"Rule {i}: IF "
            if top_features is not None:
                indices = top_features
            else:
                indices = range(len(rule_feature_names))
            for j in indices:
                if j < len(rule_feature_names):
                    m_j = m[j]
                    sigma_j = sigma[j]
                    rule_str += f"{rule_feature_names[j]} is around {m_j:.2f} ± {sigma_j:.2f} AND "
            rule_str = rule_str[:-5]  # Remove trailing " AND "
            rule_str += f" THEN mortality risk += {float(theta[0]):.2f} (bias: {float(theta[1]):.2f})"
            active_rules.append(rule_str)
    return active_rules

# ------------------------------
# Main Function
# ------------------------------
def main():
    # Load enriched data
    data_path = r"PATH_TO_FILE\preprocessed_data_enriched.pkl"
    with open(data_path, "rb") as f:
        preprocessed = pickle.load(f)
    
    X_ts, X_static, y = preprocessed["X_ts"], preprocessed["X_static"], preprocessed["y"]
    X_ts = np.nan_to_num(X_ts, nan=0.0).astype(np.float32)
    X_static = np.nan_to_num(X_static, nan=0.0).astype(np.float32)
    y = y.astype(np.float32)
    
    # Stratified split
    X_ts_train, X_ts_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
        X_ts, X_static, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
    
    # Feature Mapping for SHAP
    time_steps = X_ts.shape[1]
    ts_variables = X_ts.shape[2]
    ts_feature_names_full = [f"{var}_t{t+1}" for t in range(time_steps) for var in ts_feature_names]
    input_feature_names = ts_feature_names_full + static_feature_names
    
    # Verify feature alignment
    expected_features = (ts_variables * time_steps) + len(static_feature_names)
    assert len(input_feature_names) == expected_features, \
        f"Feature names length ({len(input_feature_names)}) does not match expected ({expected_features})"
    
    # Simulate client data
    num_clients = CONFIG["num_clients"]
    client_size = len(y_train) // num_clients
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = (i + 1) * client_size if i < num_clients - 1 else len(y_train)
        X_ts_client = X_ts_train[start_idx:end_idx]
        X_static_client = X_static_train[start_idx:end_idx]
        y_client = y_train[start_idx:end_idx]
        
        X_ts_aug, X_static_aug, y_aug = augment_minority_randomwarp(
            X_ts_client, X_static_client, y_client, max_warp_factor=0.1, augment_ratio=1.0
        )
        dataset = tf.data.Dataset.from_tensor_slices(((X_ts_aug, X_static_aug), y_aug)).shuffle(500).batch(32)
        client_datasets.append(dataset)
    
    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices(((X_ts_val, X_static_val), y_val)).batch(32)
    
    # Initialize global model
    global_model = create_fuzzy_model(
        ts_input_dim=X_ts.shape[2],
        static_input_dim=X_static.shape[1],
        hps=BEST_HPS
    )
    dummy_ts = tf.zeros((1, X_ts.shape[1], X_ts.shape[2]), dtype=tf.float32)
    dummy_static = tf.zeros((1, X_static.shape[1]), dtype=tf.float32)
    global_model([dummy_ts, dummy_static])  # Build model
    class_weights = {0: 0.58, 1: 3.65}
    
    # AUC Tracking
    auc_over_time = []
    
    # Federated training
    for round_num in range(CONFIG["federated_rounds"]):
        print(f"\nRound {round_num + 1}/{CONFIG['federated_rounds']}")
        client_models = []
        client_sizes = []
        for client_data in client_datasets:
            client_model = create_fuzzy_model(
                ts_input_dim=X_ts.shape[2],
                static_input_dim=X_static.shape[1],
                hps=BEST_HPS
            )
            client_model([dummy_ts, dummy_static])
            client_model.set_weights(global_model.get_weights())
            client_model.fit(
                client_data,
                epochs=CONFIG["local_epochs"],
                class_weight=class_weights,
                verbose=0
            )
            client_models.append(client_model)
            client_sizes.append(sum(1 for _ in client_data))
        
        global_model = aggregate_models(global_model, client_models, client_sizes)
        active_rules = evolve_rules(
            global_model, client_datasets, round_num=round_num, total_rounds=CONFIG["federated_rounds"]
        )
        print(f"Active rules: {np.sum(active_rules)} / {len(active_rules)}")
        
        val_loss, val_auc = global_model.evaluate(val_dataset, verbose=0)
        auc_over_time.append(val_auc)
        print(f"Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
    
    # Plot AUC over time
    plt.figure()
    plt.plot(range(1, len(auc_over_time) + 1), auc_over_time, marker='o', color='blue')
    plt.xlabel('Federated Round')
    plt.ylabel('Validation AUC')
    plt.title('AUC Convergence over Federated Rounds')
    plt.savefig("auc_over_time.png")
    plt.close()
    print("AUC over time plot saved as 'auc_over_time.png'")
    
    # Save the trained model
    model_save_path = r"PATH_TO_FILE\federated_fedfnn_withERL"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    global_model.save(model_save_path, save_format="tf")
    print(f"Model saved to: {model_save_path}")
    
    # Threshold search and evaluation
    def threshold_search(model, dataset):
        all_logits, all_labels = [], []
        for (x_ts_b, x_stat_b), y_b in dataset:
            logits = model((x_ts_b, x_stat_b), training=False)
            all_logits.append(logits.numpy().flatten())
            all_labels.append(y_b.numpy().flatten())
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        
        platt = LogisticRegression()
        platt.fit(all_logits.reshape(-1, 1), all_labels)
        probs = platt.predict_proba(all_logits.reshape(-1, 1))[:, 1]
        
        thresholds = np.linspace(0.0, 1.0, 101)
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            y_pred = (probs >= t).astype(int)
            f1 = f1_score(all_labels, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        y_pred_best = (probs >= best_t).astype(int)
        print(f"Best threshold: {best_t:.3f}, F1: {best_f1:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(all_labels, y_pred_best))
        print("Classification Report:\n", classification_report(all_labels, y_pred_best, digits=3))
        return platt, probs, all_labels
    
    platt, all_probs, all_labels = threshold_search(global_model, val_dataset)
    
    # Plot ROC-AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()
    print("ROC curve saved as 'roc_curve.png'")
    
    # Select background data: 100 samples from validation set
    background_ts = X_ts_val[:100]
    background_static = X_static_val[:100]

    # Define input layers for the Functional API wrapper
    ts_input = tf.keras.Input(shape=(X_ts.shape[1], X_ts.shape[2]), name='ts_input')
    static_input = tf.keras.Input(shape=(X_static.shape[1],), name='static_input')

    # Pass inputs through the subclassed model
    outputs = global_model([ts_input, static_input], training=False)  # Explicitly set training=False

    # Create a Functional API model
    functional_model = tf.keras.Model(inputs=[ts_input, static_input], outputs=outputs)

    # Create GradientExplainer with the wrapped model
    explainer = shap.GradientExplainer(functional_model, [background_ts, background_static])

    # Select samples to explain: 3 samples from validation set
    X_explain_ts = X_ts_val[50:53]
    X_explain_static = X_static_val[50:53]

    # Compute SHAP values
    shap_values = explainer.shap_values([X_explain_ts, X_explain_static])

    # Debug: Print shapes to verify
    print("Shape of shap_values[0] (time-series):", shap_values[0].shape)
    print("Shape of shap_values[1] (static):", shap_values[1].shape)

    # Calculate global SHAP importance for static features
    global_shap_static = np.abs(shap_values[1]).mean(axis=0)
    global_shap_static = global_shap_static.squeeze()

    # For time-series features, average over time steps, then compute importance
    shap_values_ts_mean = np.mean(shap_values[0], axis=1)  # Shape: (n_samples, ts_variables)
    global_shap_ts = np.abs(shap_values_ts_mean).mean(axis=0)
    global_shap_ts = global_shap_ts.squeeze()

    # Print top features by SHAP importance
    print("\nTop Time-Series Variables by SHAP Importance:")
    for idx in np.argsort(global_shap_ts)[::-1][:5]:
        print(f"{ts_feature_names[idx]}: {global_shap_ts[idx]:.4f}")

    print("\nTop Static Features by SHAP Importance:")
    for idx in np.argsort(global_shap_static)[::-1][:5]:
        print(f"{static_feature_names[idx]}: {global_shap_static[idx]:.4f}")

    # Generate SHAP summary plot for static features
    static_shap_2d = shap_values[1].squeeze(axis=-1) #shape: (3,19)
    shap.summary_plot(static_shap_2d, X_explain_static, feature_names=static_feature_names, show=False)
    plt.title("SHAP Feature Importance for Static Features")
    plt.savefig("shap_summary_plot_static.png")
    plt.close()
    print("SHAP summary plot for static features saved as 'shap_summary_plot_static.png'")
    
    # Rule simplification with top static features (retained from original)
    feature_importance = np.concatenate([global_shap_ts, global_shap_static])
    top_indices = np.argsort(feature_importance)[::-1][:5]  # Top 5 features overall
    print("Top 5 features:", top_indices)
    print("\nInterpreting rules with top 5 features:")
    rules = interpret_fuzzy_rules(
        global_model,
        rule_feature_names=[f"GRU_feature_{i}" for i in range(BEST_HPS["gru_units"])] + static_feature_names,
        top_features=top_indices
    )
    for rule in rules:
        print(rule)

if __name__ == "__main__":
    main()
