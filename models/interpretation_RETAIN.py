import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import shap

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class RetainWrapper(nn.Module):
    def __init__(self, model):
        super(RetainWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Call the RETAIN model; it returns (out, alpha, beta)
        # Return only the prediction output ('out')
        return self.model(x)[0]


##############################################################################
# 1. Data Loading & Preprocessing
##############################################################################
def load_and_preprocess_data():
    DATA_PATH = r"C:\Graduation Project\dataset\preprocessed_data_enriched.pkl"
    with open(DATA_PATH, "rb") as f:
        preprocessed = pickle.load(f)
    X_ts = preprocessed["X_ts"]      # shape: (n_samples, T, ts_dim)
    X_static = preprocessed["X_static"]  # shape: (n_samples, static_dim)
    y = preprocessed["y"]

    # Replace NaNs and convert to float32
    X_ts = np.nan_to_num(X_ts, nan=0.0).astype(np.float32)
    X_static = np.nan_to_num(X_static, nan=0.0).astype(np.float32)
    y = y.astype(np.float32)

    # Repeat static features along the time axis and concatenate
    time_steps = X_ts.shape[1]
    ts_dim = X_ts.shape[2]
    static_dim = X_static.shape[1]
    X_static_repeated = np.repeat(X_static[:, np.newaxis, :], time_steps, axis=1)
    X_combined = np.concatenate([X_ts, X_static_repeated], axis=-1)  # shape: (n_samples, T, ts_dim+static_dim)
    feature_dim = ts_dim + static_dim

    # Build feature names for SHAP visualization
    ts_feature_names = [
        ......
    ]
    static_feature_names = [
        ......
    ]
    # Note: In the RETAIN model, we use the combined input (time-series + static)
    ts_feature_names_full = [f"{var}_t{t+1}" for t in range(time_steps) for var in ts_feature_names]
    input_feature_names = ts_feature_names_full + static_feature_names
    expected_features = (ts_dim * time_steps) + len(static_feature_names)
    assert len(input_feature_names) == expected_features, f"Expected {expected_features} features, got {len(input_feature_names)}"
    
    return X_combined, feature_dim, y, time_steps, input_feature_names

##############################################################################
# 2. RETAIN MODEL DEFINITION
##############################################################################
class RETAIN(nn.Module):
    def __init__(self, input_dim, embed_dim, alpha_hidden_size, beta_hidden_size):
        """
        RETAIN model:
          - input_dim: Dimension of input features per time step (i.e., feature_dim)
          - embed_dim: Dimension for the embedding layer
          - alpha_hidden_size: Hidden size for the GRU computing alpha weights
          - beta_hidden_size: Hidden size for the GRU computing beta weights
        """
        super(RETAIN, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # GRU for alpha weights (time-level attention); bidirectional
        self.alpha_rnn = nn.GRU(embed_dim, alpha_hidden_size, batch_first=True, bidirectional=True)
        self.alpha_dense = nn.Linear(alpha_hidden_size * 2, 1)  # scalar per time step
        
        # GRU for beta weights (feature-level attention); bidirectional
        self.beta_rnn = nn.GRU(embed_dim, beta_hidden_size, batch_first=True, bidirectional=True)
        self.beta_dense = nn.Linear(beta_hidden_size * 2, embed_dim)  # vector per time step
        
        # Final prediction layer
        self.fc = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        # x: (batch, T, input_dim)
        embed = self.embedding(x)  # (batch, T, embed_dim)
        rev_embed = torch.flip(embed, dims=[1])  # Reverse time sequence
        alpha_rnn_out, _ = self.alpha_rnn(rev_embed)  # (batch, T, 2*alpha_hidden_size)
        alpha_scores = self.alpha_dense(alpha_rnn_out)  # (batch, T, 1)
        alpha_scores = torch.flip(alpha_scores, dims=[1])
        alpha = torch.softmax(alpha_scores, dim=1)  # (batch, T, 1)
        
        beta_rnn_out, _ = self.beta_rnn(rev_embed)  # (batch, T, 2*beta_hidden_size)
        beta = self.beta_dense(beta_rnn_out)  # (batch, T, embed_dim)
        beta = torch.flip(beta, dims=[1])
        beta = torch.tanh(beta)
        
        # Compute context vector as weighted sum over time
        c = torch.sum(alpha * beta * embed, dim=1)  # (batch, embed_dim)
        out = self.fc(c)  # (batch, 1)
        return out, alpha, beta

##############################################################################
# 3. TRAINING FUNCTION FOR RETAIN (with AUC history plotting)
##############################################################################
def train_retain(model, train_loader, val_loader, num_epochs=40, learning_rate=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_auc_history = []
    val_auc_history = []
    best_val_auc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        all_train_preds = []
        all_train_labels = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_train_preds.extend(preds.flatten())
            all_train_labels.extend(batch_y.cpu().numpy().flatten())
        try:
            train_fpr, train_tpr, _ = roc_curve(all_train_labels, all_train_preds)
            train_auc = auc(train_fpr, train_tpr)
        except Exception:
            train_auc = 0.0
        train_auc_history.append(train_auc)
        
        model.eval()
        epoch_val_losses = []
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits, _, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                epoch_val_losses.append(loss.item())
                preds = torch.sigmoid(logits).cpu().numpy()
                all_val_preds.extend(preds.flatten())
                all_val_labels.extend(batch_y.cpu().numpy().flatten())
        try:
            val_fpr, val_tpr, _ = roc_curve(all_val_labels, all_val_preds)
            val_auc = auc(val_fpr, val_tpr)
        except Exception:
            val_auc = 0.0
        val_auc_history.append(val_auc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {np.mean(epoch_train_losses):.4f}, Train AUC: {train_auc:.4f}, Val Loss: {np.mean(epoch_val_losses):.4f}, Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
    
    model.load_state_dict(best_model_state)
    return model, train_auc_history, val_auc_history

##############################################################################
# 4. THRESHOLD SEARCH & EVALUATION FUNCTION FOR RETAIN
##############################################################################
def evaluate_model(model, data_loader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits, _, _ = model(batch_x)
            all_logits.append(logits.cpu().numpy().flatten())
            all_labels.append(batch_y.cpu().numpy().flatten())
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
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - RETAIN")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_retain.png")
    plt.close()
    print("ROC curve saved as 'roc_curve_retain.png'")

##############################################################################
# 5. MAIN FUNCTION: RETAIN TRAINING, AUC PLOTTING, EVALUATION, & SHAP
##############################################################################
def main():
    # A) Load and preprocess data
    X_combined, feature_dim, y, time_steps, input_feature_names = load_and_preprocess_data()
    print("Loaded data. Combined shape:", X_combined.shape)
    
    # Train/Validation Split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val)
    y_val_tensor = torch.tensor(y_val).unsqueeze(1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # RETAIN processes each time step with input_dim = feature_dim (ts_dim+static_dim)
    input_dim = feature_dim
    embed_dim = 128
    alpha_hidden_size = 64
    beta_hidden_size = 64
    
    model = RETAIN(input_dim, embed_dim, alpha_hidden_size, beta_hidden_size).to(device)
    wrapped_model = RetainWrapper(model)

    # Train RETAIN model while recording AUC over epochs
    model, train_auc_history, val_auc_history = train_retain(model, train_loader, val_loader, num_epochs=40, learning_rate=1e-3)
    
    # Plot AUC history over epochs (AUC over time plot)
    plt.figure()
    plt.plot(range(1, 41), train_auc_history, label='Train AUC')
    plt.plot(range(1, 41), val_auc_history, label='Val AUC')
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("RETAIN Training History (AUC)")
    plt.legend()
    plt.savefig("training_history_retain.png")
    plt.close()
    print("Training history plot saved as 'training_history_retain.png'")
    
    # Evaluate RETAIN model on validation data
    evaluate_model(model, val_loader)

    # Temporarily disable cuDNN (workaround for RNN backward in training mode)
    was_cudnn_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    # Set the wrapped model to training mode for gradient computation
    wrapped_model.train()
    
    # SHAP Explanation using GradientExplainer
    background = X_val[:100]  # shape: (n_samples, time_steps, feature_dim)
    X_explain = X_val[50:53]  # shape: (samples, time_steps, feature_dim)
    background_tensor = torch.tensor(background).to(device)
    X_explain_tensor = torch.tensor(X_explain).to(device)
    explainer = shap.DeepExplainer(wrapped_model, background_tensor)
    shap_values = explainer.shap_values(X_explain_tensor,check_additivity=False)
    print("Shape of SHAP values:", np.array(shap_values).shape)  # Expected: (samples, time_steps, feature_dim, 1)
    
    # Re-enable cuDNN after SHAP computation
    torch.backends.cudnn.enabled = was_cudnn_enabled

    # Aggregate over the time axis and squeeze to get (samples, feature_dim)
    shap_values_avg = np.mean(shap_values, axis=1).squeeze(-1)
    X_explain_avg = np.mean(X_explain, axis=1)
    
    # Build aggregated feature names: time-series features are aggregated (avg_{var}) then static features.
    ts_feature_names = [
        ......
    ]
    static_feature_names = [
        ......
    ]
    aggregated_ts_names = [f"avg_{var}" for var in ts_feature_names]
    aggregated_feature_names = np.array(aggregated_ts_names + static_feature_names)
    
    shap.summary_plot(shap_values_avg, X_explain_avg, feature_names=aggregated_feature_names, show=False)
    plt.title("SHAP Summary Plot - RETAIN")
    plt.savefig("shap_summary_plot_retain.png")
    plt.close()
    print("SHAP summary plot saved as 'shap_summary_plot_retain.png'")
    
    # Save the RETAIN model state
    MODEL_SAVE_DIR = r"PATH_TO_FILE\interpretation_retain"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "interpretation_retain.pkl")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"RETAIN model saved as '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    main()
