import os
import pandas as pd
import numpy as np
import pickle
import re

# -------------------------
# Configuration: All CSV files are in one folder.
# -------------------------
DATASET_DIR = r"C:\Graduation Project\dataset"
COHORT_FILE = os.path.join(DATASET_DIR, "cohort.csv")
STATIC_FILE = os.path.join(DATASET_DIR, "static_features_with_comorbidities.csv")
TS_FILE = os.path.join(DATASET_DIR, "time_series_features.csv")
LABELS_FILE = os.path.join(DATASET_DIR, "labels.csv")
OUTPUT_PICKLE = os.path.join(DATASET_DIR, "preprocessed_data.pkl")

# -------------------------
# Utility Function for Loading CSVs
# -------------------------
def load_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

# -------------------------
# Data Loading
# -------------------------
print("Loading cohort data...")
cohort_df = load_csv(COHORT_FILE)
print(f"Cohort data loaded: {cohort_df.shape[0]} records.")

print("Loading static features with comorbidities...")
static_df = load_csv(STATIC_FILE)
print(f"Static features loaded: {static_df.shape[0]} records.")

print("Loading time-series features...")
ts_df = load_csv(TS_FILE)
print(f"Time-series features loaded: {ts_df.shape[0]} records.")

print("Loading labels...")
labels_df = load_csv(LABELS_FILE)
print(f"Labels loaded: {labels_df.shape[0]} records.")

# -------------------------
# Ensure Merge Keys are Consistent
# -------------------------
# Convert possible uppercase keys to lowercase for merging.
if "hadm_id" not in static_df.columns and "HADM_ID" in static_df.columns:
    static_df.rename(columns={"HADM_ID": "hadm_id"}, inplace=True)
if "hadm_id" not in labels_df.columns and "HADM_ID" in labels_df.columns:
    labels_df.rename(columns={"HADM_ID": "hadm_id"}, inplace=True)
# Optionally, do the same for cohort_df if needed:
if "hadm_id" not in cohort_df.columns and "HADM_ID" in cohort_df.columns:
    cohort_df.rename(columns={"HADM_ID": "hadm_id"}, inplace=True)

# -------------------------
# Merging Data
# -------------------------
print("Merging data...")

# Step 1: Merge time-series features with cohort using "stay_id"
if "stay_id" not in ts_df.columns:
    raise KeyError("The time_series_features.csv file must contain a 'stay_id' column.")

# We assume cohort_df has both 'hadm_id' and 'stay_id'
merged_ts = pd.merge(cohort_df[['hadm_id', 'stay_id']], ts_df, on="stay_id", how="left")
print("After merging time-series with cohort, shape:", merged_ts.shape)

# Step 2: Merge with static features (by hadm_id)
merged_static = pd.merge(static_df, merged_ts, on="hadm_id", how="left")
print("After merging with static features, shape:", merged_static.shape)

# Step 3: Merge with labels (by hadm_id)
merged_df = pd.merge(merged_static, labels_df, on="hadm_id", how="left")
print("After merging with labels, final shape:", merged_df.shape)

print("Merged data sample:")
print(merged_df.head())

# -------------------------
# (Optional) Prepare Model Input Arrays
# -------------------------
# Identify time-series columns: assume they do not include "stay_id" or "hadm_id"
import re

# Select columns that match the pattern: feature name followed by an underscore and a number
ts_feature_columns = [col for col in merged_df.columns if re.match(r'^(?!mask_).+_\d+$', col)]

# Group columns by feature prefix
features = {}
for col in ts_feature_columns:
    m = re.match(r'(.+)_([0-9]+)$', col)
    if m:
        feature_name = m.group(1)
        hour = int(m.group(2))
        features.setdefault(feature_name, {})[hour] = col

# Sort feature names (you can define your preferred order)
features_ordered = sorted(features.keys())
TIME_WINDOW = 48  # expected hours

# For each feature, collect columns for hours 0 to TIME_WINDOW-1 in order
all_columns_ordered = []
for feat in features_ordered:
    cols = [features[feat][h] for h in range(TIME_WINDOW)]
    all_columns_ordered.extend(cols)

N = merged_df.shape[0]
n_ts_features = len(features_ordered)
X_ts = merged_df[all_columns_ordered].values.astype(np.float32).reshape(N, TIME_WINDOW, n_ts_features)

# For static features, we use the static columns from static_df.
static_feature_names = [col for col in static_df.columns if col != "hadm_id"]
X_static = merged_df[static_feature_names].values.astype(np.float32) if static_feature_names else None

# Extract labels (assumed column name is "MORTALITY")
y = merged_df["MORTALITY"].values.astype(np.float32) if "MORTALITY" in merged_df.columns else None

# -------------------------
# Pack Data and Save as Pickle
# -------------------------
preprocessed_data = {
    "merged_df": merged_df,   # The full merged DataFrame
    "X_ts": X_ts,             # Time-series array (if available)
    "X_static": X_static,     # Static features array (demographics + comorbidities)
    "y": y                    # Labels array
}

with open(OUTPUT_PICKLE, "wb") as f:
    pickle.dump(preprocessed_data, f)

print("Preprocessed data has been saved to", OUTPUT_PICKLE)
