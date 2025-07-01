import os
import pandas as pd
import numpy as np

# ------------------------------
# Define paths for MIMIC-IV data
# ------------------------------
BASE_DIR = r"C:\Graduation Project\dataset\mimic-iv"
ICU_DIR = os.path.join(BASE_DIR, "icu")
HOSP_DIR = os.path.join(BASE_DIR, "hosp")

# Dictionary files (you already have these full dictionaries)
d_items_file = os.path.join(ICU_DIR, "d_items.csv.gz")
d_labitems_file = os.path.join(HOSP_DIR, "d_labitems.csv.gz")

# Raw data files
icustays_file = os.path.join(ICU_DIR, "icustays.csv.gz")
chartevents_file = os.path.join(ICU_DIR, "chartevents.csv.gz")
labevents_file = os.path.join(HOSP_DIR, "labevents.csv.gz")

# Output files
merged_sofa_data_file = os.path.join(BASE_DIR, "merged_sofa_data.csv")
severity_scores_file = os.path.join(BASE_DIR, "severity_scores.csv")

# ------------------------------
# Step A: Define ITEMIDs directly based on your dictionary review
# ------------------------------
# For MAP, we use the “Arterial Blood Pressure mean” ITEMID
MAP_ITEMID = 220052

# For vasopressor usage, we use the “Vasopressin” ITEMID
VASOPRESSOR_ITEMID = 222315

# For lab variables, you may continue to use your original filtering
# (Adjust these if needed based on your dictionary)
# For bilirubin and creatinine, we can still filter by the lab dictionary using 'bilirubin' and 'creatinine'
# Here we load the dictionaries and perform filtering for labs.

print("Loading dictionary files...")
d_items = pd.read_csv(d_items_file)
d_labitems = pd.read_csv(d_labitems_file)

# For bilirubin
bilirubin_items = d_labitems[d_labitems['label'].str.contains("bilirubin", case=False, na=False)]
bilirubin_itemids = bilirubin_items['itemid'].unique().tolist()

# For creatinine
creatinine_items = d_labitems[d_labitems['label'].str.contains("creatinine", case=False, na=False)]
creatinine_itemids = creatinine_items['itemid'].unique().tolist()

# ------------------------------
# Step B: Load Raw Data Files
# ------------------------------
print("Loading ICU stays data...")
icustays = pd.read_csv(icustays_file)
print(f"ICU stays: {icustays.shape[0]} records")

print("Loading chartevents data...")
# When loading chartevents, we filter only for the ITEMIDs of interest to save memory.
# We'll load in chunks and only keep rows with MAP_ITEMID or VASOPRESSOR_ITEMID.
chunksize = 10 ** 6
chartevent_chunks = []
for chunk in pd.read_csv(chartevents_file, compression='gzip', chunksize=chunksize):
    filtered_chunk = chunk[chunk['itemid'].isin([MAP_ITEMID, VASOPRESSOR_ITEMID])]
    chartevent_chunks.append(filtered_chunk)
chartevents = pd.concat(chartevent_chunks, ignore_index=True)
print(f"Filtered chartevents: {chartevents.shape[0]} records")

print("Loading labevents data...")
labevents = pd.read_csv(labevents_file, compression='gzip')
print(f"Labevents: {labevents.shape[0]} records")

# ------------------------------
# Step C: Extract Variables from Chartevents (ICU)
# ------------------------------
# Extract MAP values: we use only ITEMID 220052
map_events = chartevents[chartevents["itemid"] == MAP_ITEMID].copy()
# Compute the mean MAP per ICU stay
map_by_stay = map_events.groupby("stay_id")["valuenum"].mean().reset_index()
map_by_stay.rename(columns={"valuenum": "MAP"}, inplace=True)

# Extract vasopressor usage: if any event for ITEMID 222315 exists for a stay, flag as 1.
vaso_events = chartevents[chartevents["itemid"] == VASOPRESSOR_ITEMID].copy()
vaso_by_stay = vaso_events.groupby("stay_id").size().reset_index(name="vasopressor")
vaso_by_stay["vasopressor"] = 1  # Flag usage

# Merge MAP and vasopressor info on stay_id
icu_chartevent_vars = pd.merge(map_by_stay, vaso_by_stay, on="stay_id", how="left")
icu_chartevent_vars["vasopressor"].fillna(0, inplace=True)

# ------------------------------
# Step D: Extract Variables from Labevents (Hospital)
# ------------------------------
# Extract bilirubin values
bilirubin_events = labevents[labevents["itemid"].isin(bilirubin_itemids)].copy()
bilirubin_by_hadm = bilirubin_events.groupby("hadm_id")["valuenum"].mean().reset_index()
bilirubin_by_hadm.rename(columns={"valuenum": "bilirubin"}, inplace=True)

# Extract creatinine values
creatinine_events = labevents[labevents["itemid"].isin(creatinine_itemids)].copy()
creatinine_by_hadm = creatinine_events.groupby("hadm_id")["valuenum"].mean().reset_index()
creatinine_by_hadm.rename(columns={"valuenum": "creatinine"}, inplace=True)

# Merge lab variables by hadm_id
labs_by_hadm = pd.merge(bilirubin_by_hadm, creatinine_by_hadm, on="hadm_id", how="outer")

# ------------------------------
# Step E: Merge ICU and Lab Data
# ------------------------------
# Merge icustays with chartevent variables on stay_id
icu_merged = pd.merge(icustays, icu_chartevent_vars, on="stay_id", how="left")
# Merge with lab variables on hadm_id
merged_sofa_data = pd.merge(icu_merged, labs_by_hadm, on="hadm_id", how="left")

# Keep only the necessary columns.
merged_sofa_data = merged_sofa_data[["subject_id", "hadm_id", "stay_id", "MAP", "vasopressor", "bilirubin", "creatinine"]]

# Instead of dropping all rows with missing values, you might consider imputing or allowing some missingness.
# For now, we will drop rows with missing MAP, bilirubin, or creatinine.
merged_sofa_data.dropna(subset=["MAP", "bilirubin", "creatinine"], inplace=True)

print(f"After merging and dropping missing values, we have {merged_sofa_data.shape[0]} records.")

# Save merged data (optional)
merged_sofa_data.to_csv(merged_sofa_data_file, index=False)
print("Merged SOFA data saved to:", merged_sofa_data_file)

# ------------------------------
# Step F: Compute Simplified SOFA Scores
# ------------------------------
def compute_sofa_subscores(df):
    # Cardiovascular subscore:
    # If MAP >= 70, score 0; if MAP < 70, score 1; if vasopressor==1, override to score 2.
    df['cardio_score'] = np.where(df['MAP'] >= 70, 0, 1)
    df['cardio_score'] = np.where(df['vasopressor'] == 1, 2, df['cardio_score'])
    
    # Liver subscore using bilirubin (mg/dL)
    df['liver_score'] = np.select(
        [df['bilirubin'] < 1.2,
         (df['bilirubin'] >= 1.2) & (df['bilirubin'] < 2.0),
         (df['bilirubin'] >= 2.0) & (df['bilirubin'] < 6.0),
         (df['bilirubin'] >= 6.0) & (df['bilirubin'] < 12.0)],
        [0, 1, 2, 3],
        default=4
    )
    
    # Renal subscore using creatinine (mg/dL)
    df['renal_score'] = np.select(
        [df['creatinine'] < 1.2,
         (df['creatinine'] >= 1.2) & (df['creatinine'] < 2.0),
         (df['creatinine'] >= 2.0) & (df['creatinine'] < 3.5),
         (df['creatinine'] >= 3.5) & (df['creatinine'] < 5.0)],
        [0, 1, 2, 3],
        default=4
    )
    
    # Total SOFA score is the sum of the subscores.
    df['SOFA_score'] = df['cardio_score'] + df['liver_score'] + df['renal_score']
    return df

sofa_data = compute_sofa_subscores(merged_sofa_data)

# ------------------------------
# Step G: Save Severity Scores
# ------------------------------
severity_scores = sofa_data[['subject_id', 'hadm_id', 'SOFA_score', 'cardio_score', 'liver_score', 'renal_score']]
severity_scores.to_csv(severity_scores_file, index=False)
print("Severity scores saved to:", severity_scores_file)
print("Example severity scores:")
print(severity_scores.head())
