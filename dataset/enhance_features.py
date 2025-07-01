import os
import pickle
import pandas as pd

# Define file paths (adjust these if necessary)
BASE_DIR = r"C:\Graduation Project\dataset"
# Preprocessed data from your initial pipeline
PREPROCESSED_FILE = os.path.join(BASE_DIR, "preprocessed_data.pkl")
# Additional features computed in previous steps:
SEVERITY_FILE = os.path.join(BASE_DIR, "severity_scores.csv")
MEDICATION_FILE = os.path.join(BASE_DIR, "medication_features.csv")
LAB_TRENDS_FILE = os.path.join(BASE_DIR, "lab_trends_extended.csv")
# Output merged file
OUTPUT_FILE = os.path.join(BASE_DIR, "preprocessed_data_enriched.pkl")

# Step 1: Load the original preprocessed data and the new feature files.
with open(PREPROCESSED_FILE, "rb") as f:
    preprocessed = pickle.load(f)

# Assuming your original preprocessed data is a dictionary with keys like "merged_df", "X_ts", "X_static", "y"
# and that "merged_df" (or "cohort") has identifiers such as 'hadm_id' (and maybe 'stay_id', 'subject_id')
original_df = preprocessed.get("merged_df")  # or preprocessed["cohort"] if that's what you used

# Load additional features
severity_df = pd.read_csv(SEVERITY_FILE)
medication_df = pd.read_csv(MEDICATION_FILE)
lab_trends_df = pd.read_csv(LAB_TRENDS_FILE)

# Step 2: Merge the additional features into the original dataset.
# Here, we use 'hadm_id' as the key, but you may adjust if needed.
# You can merge sequentially.

# Merge severity scores
merged_df = pd.merge(original_df, severity_df, on="hadm_id", how="left")
print("After merging severity scores:", merged_df.shape)

# Merge medication features (if your medication file also has 'hadm_id'; if only 'stay_id' is available, merge accordingly)
if "hadm_id" in medication_df.columns:
    merged_df = pd.merge(merged_df, medication_df, on="hadm_id", how="left")
else:
    # If medication is by 'stay_id', and original_df has 'stay_id' too:
    merged_df = pd.merge(merged_df, medication_df, on="stay_id", how="left")
print("After merging medication features:", merged_df.shape)

# Merge lab trends features (they are by hadm_id)
merged_df = pd.merge(merged_df, lab_trends_df, on="hadm_id", how="left")
print("After merging lab trends:", merged_df.shape)

# Step 3: Handle missing values for the new features.
# For example, if any new feature is missing (NaN), you can fill with 0 or use another imputation strategy.
new_feature_cols = ["SOFA_score", "cardio_score", "liver_score", "renal_score",
                    "ventilation_flag", "total_sedation_dose", "sedation_flag",
                    "bilirubin_slope", "creatinine_slope", "lactate_slope", "platelets_slope"]
for col in new_feature_cols:
    if col in merged_df.columns:
        merged_df[col].fillna(0, inplace=True)

# Optionally, create an enriched static feature matrix.
# For example, if your original static features were stored in preprocessed["X_static"],
# you might update them by adding the new columns.
# Here, we assume that your merged_df now contains all relevant features, so you can extract the static ones.
# For instance:
static_feature_cols = ["cardio_score", "liver_score", "renal_score",
                       "ventilation_flag", "total_sedation_dose", "sedation_flag",
                       "bilirubin_slope", "creatinine_slope", "lactate_slope", "platelets_slope"]
X_static_enriched = merged_df[static_feature_cols].to_numpy()

# You can now update your preprocessed data dictionary.
preprocessed["merged_df_enriched"] = merged_df
preprocessed["X_static_enriched"] = X_static_enriched

# Save the enriched preprocessed data
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(preprocessed, f)

print("Enriched preprocessed data saved to:", OUTPUT_FILE)
