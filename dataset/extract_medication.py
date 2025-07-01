import os
import pandas as pd
import numpy as np

# Define paths
BASE_DIR = r"C:\Graduation Project\dataset\mimic-iv"
ICU_DIR = os.path.join(BASE_DIR, "icu")
HOSP_DIR = os.path.join(BASE_DIR, "hosp")
RAW_CHARTEVENTS_FILE = os.path.join(ICU_DIR, "chartevents.csv.gz")
RAW_INPUTEVENTS_FILE = os.path.join(ICU_DIR, "inputevents.csv.gz")
MED_FEATURES_FILE = os.path.join(BASE_DIR, "medication_features.csv")

# Define ITEMIDs (replace these with validated ITEMIDs from your dictionary review)
# Ventilation-related ITEMIDs (e.g., for ventilator type, mode, invasive/non-invasive flags)
VENTILATION_ITEMIDS = [223848, 223849, 225792, 225794, 226260]
# Sedation-related ITEMIDs (e.g., for propofol, midazolam, fentanyl, dexmedetomidine)
SEDATION_ITEMIDS = [222168, 221668, 221744, 225150]

# ----- Process Ventilation Events from chartevents -----
print("Processing chartevents for ventilation events...")
ventilation_chunks = []
for chunk in pd.read_csv(RAW_CHARTEVENTS_FILE, compression='gzip', chunksize=10**6):
    vent_chunk = chunk[chunk["itemid"].isin(VENTILATION_ITEMIDS)]
    if not vent_chunk.empty:
        ventilation_chunks.append(vent_chunk)
if ventilation_chunks:
    ventilation_events = pd.concat(ventilation_chunks, ignore_index=True)
else:
    ventilation_events = pd.DataFrame()

print(f"Chartevents - Ventilation events found: {ventilation_events.shape[0]} records")

# For ventilation, we create a binary flag per ICU stay.
vent_flags = ventilation_events.groupby("stay_id").size().reset_index(name="ventilation_count")
vent_flags["ventilation_flag"] = 1  # set flag = 1 if any event exists

# ----- Process Sedation Events from both chartevents and inputevents -----
print("Processing chartevents for sedation events...")
sedation_char_chunks = []
for chunk in pd.read_csv(RAW_CHARTEVENTS_FILE, compression='gzip', chunksize=10**6):
    sed_chunk = chunk[chunk["itemid"].isin(SEDATION_ITEMIDS)]
    if not sed_chunk.empty:
        sedation_char_chunks.append(sed_chunk)
sedation_char_df = pd.concat(sedation_char_chunks, ignore_index=True) if sedation_char_chunks else pd.DataFrame()
print(f"Chartevents - Sedation events found: {sedation_char_df.shape[0]} records")

print("Processing inputevents for sedation events...")
sedation_inp_chunks = []
for chunk in pd.read_csv(RAW_INPUTEVENTS_FILE, compression='gzip', chunksize=10**6):
    sed_chunk = chunk[chunk["itemid"].isin(SEDATION_ITEMIDS)]
    if not sed_chunk.empty:
        sedation_inp_chunks.append(sed_chunk)
sedation_inp_df = pd.concat(sedation_inp_chunks, ignore_index=True) if sedation_inp_chunks else pd.DataFrame()
print(f"Inputevents - Sedation events found: {sedation_inp_df.shape[0]} records")

# Combine sedation events from both sources
sedation_events = pd.concat([sedation_char_df, sedation_inp_df], ignore_index=True)
print(f"Total Sedation events: {sedation_events.shape[0]} records")

# ----- Aggregate Sedation Data: Compute Total Dose if Available -----
# Check if there's a column that holds the dosage amount (commonly named 'amount')
if "amount" in sedation_events.columns:
    # Ensure the column is numeric
    sedation_events["amount"] = pd.to_numeric(sedation_events["amount"], errors="coerce")
    # Sum the total sedation dose per ICU stay
    sed_agg = sedation_events.groupby("stay_id")["amount"].sum().reset_index()
    sed_agg.rename(columns={"amount": "total_sedation_dose"}, inplace=True)
    print("Aggregated sedation dose computed.")
else:
    # Fall back on event count if no dosage is available
    sed_agg = sedation_events.groupby("stay_id").size().reset_index(name="sedation_count")
    sed_agg["total_sedation_dose"] = sed_agg["sedation_count"]
    print("No sedation dose column found; using event count as proxy.")

# Create a binary flag for sedation usage (flag=1 if any event exists)
sed_agg["sedation_flag"] = 1

# ----- Merge Medication Features -----
# Load icustays (which should include stay_id and hadm_id)
icustays_file = os.path.join(ICU_DIR, "icustays.csv.gz")
icustays = pd.read_csv(icustays_file)

# Start with a basic table from icustays and merge ventilation and sedation data on stay_id
med_features = icustays[["stay_id", "hadm_id"]].copy()
med_features = med_features.merge(vent_flags[["stay_id", "ventilation_flag"]], on="stay_id", how="left")
med_features = med_features.merge(sed_agg[["stay_id", "total_sedation_dose", "sedation_flag"]], on="stay_id", how="left")

# Replace missing values (NaNs) with 0
med_features["ventilation_flag"].fillna(0, inplace=True)
med_features["total_sedation_dose"].fillna(0, inplace=True)
med_features["sedation_flag"].fillna(0, inplace=True)

# Save the medication features to CSV
med_features.to_csv(MED_FEATURES_FILE, index=False)
print("Medication features saved to:", MED_FEATURES_FILE)
print(med_features.head())
