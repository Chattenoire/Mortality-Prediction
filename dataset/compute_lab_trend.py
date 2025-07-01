import os
import pandas as pd
import numpy as np
from scipy.stats import linregress

# ------------------------------
# Define Paths
# ------------------------------
BASE_DIR = r"C:\Graduation Project\dataset\mimic-iv"
HOSP_DIR = os.path.join(BASE_DIR, "hosp")
LABEVENTS_FILE = os.path.join(HOSP_DIR, "labevents.csv.gz")
OUTPUT_TREND_FILE = os.path.join(BASE_DIR, "lab_trends_extended.csv")

# ------------------------------
# Define ITEMIDs for lab variables
# ------------------------------
# Use the validated ITEMIDs based on your investigation.
BILIRUBIN_ITEMIDS = [50838, 50883, 50884, 50885, 51028, 51049, 51464, 51465, 51568, 51569, 51570, 51783, 51812, 51932, 51966, 53089]
CREATININE_ITEMIDS = [50841, 50912, 51021, 51032, 51052, 51067, 51070, 51073, 51080, 51081, 51082, 51099, 51106, 51787, 51937, 51963, 51977, 52000, 52024, 52546]
LACTATE_ITEMIDS = [50813, 52442, 53154]
PLATELET_ITEMIDS = [51240]

# ------------------------------
# Function to compute slope for a given group of measurements
# ------------------------------
def compute_slope(group, value_col, time_col):
    """
    Compute the slope of lab values over time using linear regression.
    If there is no variability in the time variable, returns 0.0.
    """
    # Ensure the group is sorted by the time column
    group = group.sort_values(time_col)
    # Convert time column to datetime and calculate hours since ICU admission
    hours = group['hours_since_intime']
    values = group[value_col].astype(float)
    
    # If there is only one unique time value, linear regression is not feasible.
    if hours.nunique() == 1:
        return 0.0
    try:
        slope, _, _, _, _ = linregress(hours, values)
        return slope
    except Exception as e:
        return 0.0

# ------------------------------
# Main processing: compute lab trends per hadm_id for multiple labs
# ------------------------------
print("Loading labevents data...")
labevents = pd.read_csv(LABEVENTS_FILE, compression='gzip')

# Before computing slopes, ensure that 'charttime' is converted to datetime.
labevents['charttime'] = pd.to_datetime(labevents['charttime'])
labevents['intime'] = labevents.groupby('hadm_id')['charttime'].transform('min')
# Compute hours since ICU admission
labevents['hours_since_intime'] = (labevents['charttime'] - labevents['intime']).dt.total_seconds() / 3600.0

# Function to compute slopes for a given lab type
def compute_lab_slope(lab_itemids, value_name):
    lab_events = labevents[labevents["itemid"].isin(lab_itemids)].copy()
    # Drop rows without necessary time or value information
    lab_events.dropna(subset=["charttime", "valuenum", "hours_since_intime"], inplace=True)
    # Group by hadm_id and compute slope for each group
    slopes = lab_events.groupby("hadm_id").apply(
        lambda g: compute_slope(g, "valuenum", "charttime")
    ).reset_index(name=f"{value_name}_slope")
    return slopes

print("Computing bilirubin slopes...")
bilirubin_slopes = compute_lab_slope(BILIRUBIN_ITEMIDS, "bilirubin")

print("Computing creatinine slopes...")
creatinine_slopes = compute_lab_slope(CREATININE_ITEMIDS, "creatinine")

print("Computing lactate slopes...")
lactate_slopes = compute_lab_slope(LACTATE_ITEMIDS, "lactate")

print("Computing platelet slopes...")
platelet_slopes = compute_lab_slope(PLATELET_ITEMIDS, "platelets")

# ------------------------------
# Merge all slopes together on hadm_id
# ------------------------------
print("Merging lab trends...")
lab_trends = pd.merge(bilirubin_slopes, creatinine_slopes, on="hadm_id", how="outer")
lab_trends = pd.merge(lab_trends, lactate_slopes, on="hadm_id", how="outer")
lab_trends = pd.merge(lab_trends, platelet_slopes, on="hadm_id", how="outer")

# Impute missing slopes with 0 (alternative imputation strategies can be used)
for col in ["bilirubin_slope", "creatinine_slope", "lactate_slope", "platelets_slope"]:
    lab_trends[col].fillna(0, inplace=True)

print(f"Computed lab trends for {lab_trends.shape[0]} hospital admissions.")
lab_trends.to_csv(OUTPUT_TREND_FILE, index=False)
print("Extended lab trend features saved to:", OUTPUT_TREND_FILE)
