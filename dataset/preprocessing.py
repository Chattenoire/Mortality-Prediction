import os
import pandas as pd
import numpy as np
from datetime import timedelta

def load_compressed_csv(filepath, usecols=None):
    """Load a gzipped CSV file with optional columns."""
    return pd.read_csv(filepath, compression='gzip', usecols=usecols)

def load_filtered_events(filepath, usecols, filter_dict, chunksize=1000000):
    """
    Reads a gzipped CSV file in chunks and filters rows based on filter_dict.
    
    filter_dict: dictionary where keys are column names and values are sets of acceptable values.
    """
    chunks = []
    for chunk in pd.read_csv(filepath, compression='gzip', usecols=usecols, chunksize=chunksize):
        mask = np.ones(len(chunk), dtype=bool)
        for col, valid_set in filter_dict.items():
            mask &= chunk[col].isin(valid_set)
        if mask.any():
            chunks.append(chunk[mask])
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.DataFrame(columns=usecols)
    return df

def extract_cohort(icu_df, adm_df, pat_df):
    """
    Merge ICU stays with admissions and patients to build the cohort.
    Filters for adult patients (using 'anchor_age' if available) and keeps only the first ICU stay 
    per patient with at least 48 hours in the ICU.
    Note: In MIMIC-IV, the ICU stay identifier is "stay_id".
    """
    df = icu_df.merge(adm_df, on=['hadm_id', 'subject_id'], how='inner')
    df = df.merge(pat_df, on='subject_id', how='inner')
    
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df['intime'] = pd.to_datetime(df['intime'])
    df['outtime'] = pd.to_datetime(df['outtime'])
    if 'dod' in df.columns:
        df['dod'] = pd.to_datetime(df['dod'], errors='coerce')
    
    df['MORTALITY'] = df.apply(lambda row: 1 if pd.notna(row.get('dod')) and row['admittime'] <= row['dod'] <= row['dischtime'] else 0, axis=1)
    
    if 'anchor_age' in df.columns:
        df = df[df['anchor_age'] >= 18]
    elif 'age' in df.columns:
        df = df[df['age'] >= 18]
    
    df = df.sort_values(by=['subject_id', 'intime']).drop_duplicates(subset=['subject_id'], keep='first')
    
    df['icu_length'] = (df['outtime'] - df['intime']).dt.total_seconds() / 3600.0
    df = df[df['icu_length'] >= 48]
    
    return df

def extract_time_series(cohort_df, chartevents_df, labevents_df, window_hours=48):
    """
    Extract time-series data from the first 48 hours of each ICU stay.
    
    - For chartevents (vitals), merge on "stay_id".
    - For labevents (labs), merge on "hadm_id" and then bring in "stay_id" from the cohort.
    
    Then, for each selected feature:
      1. Restrict observations to within 48 hours of ICU admission.
      2. Compute the integer hour (0 to 47) from ICU admission.
      3. Aggregate values by hour (averaging if multiple measurements exist).
      4. Pivot the data so that each ICU stay is a row with 48 columns per feature.
      5. Create a binary mask indicating if a value was recorded.
      6. Forward-fill missing values (row-wise) and, for labs, fill remaining leading NaNs with the median.
      7. Normalize each numeric feature column (z‑score normalization).
    """
    # Process chartevents (vitals)
    chartevents_df['charttime'] = pd.to_datetime(chartevents_df['charttime'])
    cohort_times_vitals = cohort_df[['stay_id', 'intime']].copy()
    cohort_times_vitals['intime'] = pd.to_datetime(cohort_times_vitals['intime'])
    chartevents_df = chartevents_df.merge(cohort_times_vitals, on='stay_id', how='inner')
    chartevents_df = chartevents_df[(chartevents_df['charttime'] >= chartevents_df['intime']) & 
                                    (chartevents_df['charttime'] <= chartevents_df['intime'] + pd.Timedelta(hours=window_hours))]
    chartevents_df['hour'] = ((chartevents_df['charttime'] - chartevents_df['intime']).dt.total_seconds() // 3600).astype(int)
    
    # Process labevents (labs)
    labevents_df['charttime'] = pd.to_datetime(labevents_df['charttime'])
    cohort_times_labs = cohort_df[['hadm_id', 'intime', 'stay_id']].copy()
    cohort_times_labs['intime'] = pd.to_datetime(cohort_times_labs['intime'])
    labevents_df = labevents_df.merge(cohort_times_labs, on='hadm_id', how='inner')
    labevents_df = labevents_df[(labevents_df['charttime'] >= labevents_df['intime']) & 
                                (labevents_df['charttime'] <= labevents_df['intime'] + pd.Timedelta(hours=window_hours))]
    labevents_df['hour'] = ((labevents_df['charttime'] - labevents_df['intime']).dt.total_seconds() // 3600).astype(int)
    
    # Define features of interest with corresponding MIMIC item IDs.
    vital_signs = {
        'HeartRate': [211, 220045],
        'SysBP': [51, 442, 220179],
        'DiasBP': [8368, 8441, 220180],
        'MeanBP': [52, 456, 220181],
        'RespRate': [618, 615, 220210],
        'SpO2': [646, 220277],
        'Temperature': [676, 223761],
    }
    lab_tests = {
        'WBC': [51300],
        'Hemoglobin': [51222],
        'Platelets': [51265],
        'Sodium': [50824],
        'Potassium': [50821],
        'Chloride': [50806],
        'BUN': [51006],
        'Creatinine': [50912],
        'Glucose': [50931],
        'Arterial_pH': [50820],
        'Arterial_Lactate': [50813],
    }
    
    # Dictionary to hold pivoted DataFrames for each feature.
    feature_dfs = {}
    
    # Process vital signs from chartevents.
    for feat, itemids in vital_signs.items():
        df_feat = chartevents_df[chartevents_df['itemid'].isin(itemids)]
        agg = df_feat.groupby(['stay_id', 'hour'])['value'].mean().reset_index()
        pivot_df = agg.pivot(index='stay_id', columns='hour', values='value')
        pivot_df = pivot_df.reindex(columns=range(window_hours), fill_value=np.nan)
        pivot_df = pivot_df.add_prefix(f'{feat}_')
        feature_dfs[feat] = pivot_df

    # Process lab tests from labevents.
    for feat, itemids in lab_tests.items():
        df_feat = labevents_df[labevents_df['itemid'].isin(itemids)]
        agg = df_feat.groupby(['stay_id', 'hour'])['value'].mean().reset_index()
        pivot_df = agg.pivot(index='stay_id', columns='hour', values='value')
        pivot_df = pivot_df.reindex(columns=range(window_hours), fill_value=np.nan)
        pivot_df = pivot_df.add_prefix(f'{feat}_')
        feature_dfs[feat] = pivot_df
    
    # Combine all individual feature DataFrames (outer join on stay_id).
    ts_data = None
    for df in feature_dfs.values():
        ts_data = df if ts_data is None else ts_data.join(df, how='outer')
    
    # Create binary masks: 1 if a value is recorded, 0 otherwise.
    mask_data = ts_data.notna().astype(int)
    mask_data = mask_data.add_prefix('mask_')
    ts_data = ts_data.join(mask_data)
    
    # Forward-fill missing values for all non-mask columns (row-wise).
    non_mask_cols = [col for col in ts_data.columns if not col.startswith('mask_')]
    ts_data[non_mask_cols] = ts_data[non_mask_cols].ffill(axis=1)
    
    # For lab test features, fill any remaining leading NaNs with the column median.
    for feat in lab_tests.keys():
        cols = [f'{feat}_{h}' for h in range(window_hours)]
        median_val = ts_data[cols].stack().median()
        ts_data[cols] = ts_data[cols].fillna(median_val)
    
    # Normalize (z-score) each numeric feature column (skip mask columns).
    for col in non_mask_cols:
        mean = ts_data[col].mean()
        std = ts_data[col].std()
        std = std if std != 0 else 1
        ts_data[col] = (ts_data[col] - mean) / std
            
    return ts_data

def main():
    # Define dataset directories.
    DATASET_DIR = r"C:\Graduation Project\dataset\mimic-iv"
    OUTPUT_DIR = r"C:\Graduation Project\dataset"
    ICU_DIR = os.path.join(DATASET_DIR, "icu")
    HOSP_DIR = os.path.join(DATASET_DIR, "hosp")
    
    # File paths for key CSV files.
    ICU_STAYS_FILE = os.path.join(ICU_DIR, "icustays.csv.gz")
    ADMISSIONS_FILE = os.path.join(HOSP_DIR, "admissions.csv.gz")
    PATIENTS_FILE = os.path.join(HOSP_DIR, "patients.csv.gz")
    
    # For chartevents (vitals), use columns: stay_id, charttime, itemid, value.
    CHARTEVENTS_FILE = os.path.join(ICU_DIR, "chartevents.csv.gz")
    chartevents_usecols = ['stay_id', 'charttime', 'itemid', 'value']
    
    # For labevents (labs), use columns: hadm_id, charttime, itemid, value.
    LABEVENTS_FILE = os.path.join(HOSP_DIR, "labevents.csv.gz")
    labevents_usecols = ['hadm_id', 'charttime', 'itemid', 'value']
    
    print("Loading ICU stays, admissions, and patients...")
    icu_df = load_compressed_csv(ICU_STAYS_FILE)
    adm_df = load_compressed_csv(ADMISSIONS_FILE)
    pat_df = load_compressed_csv(PATIENTS_FILE)
    
    print("Extracting cohort...")
    cohort_df = extract_cohort(icu_df, adm_df, pat_df)
    print(f"Cohort size: {cohort_df.shape[0]}")
    
    # For chartevents, filter by "stay_id" using the cohort.
    cohort_stay_ids = set(cohort_df['stay_id'])
    
    # For labevents, filter by "hadm_id". Use the cohort's hadm_id.
    cohort_hadm_ids = set(cohort_df['hadm_id'])
    
    # Define the union of item IDs for vital signs and labs.
    vital_itemids = set([item for sublist in {
        'HeartRate': [211, 220045],
        'SysBP': [51, 442, 220179],
        'DiasBP': [8368, 8441, 220180],
        'MeanBP': [52, 456, 220181],
        'RespRate': [618, 615, 220210],
        'SpO2': [646, 220277],
        'Temperature': [676, 223761],
    }.values() for item in sublist])
    
    lab_itemids = set([item for sublist in {
        'WBC': [51300],
        'Hemoglobin': [51222],
        'Platelets': [51265],
        'Sodium': [50824],
        'Potassium': [50821],
        'Chloride': [50806],
        'BUN': [51006],
        'Creatinine': [50912],
        'Glucose': [50931],
        'Arterial_pH': [50820],
        'Arterial_Lactate': [50813],
    }.values() for item in sublist])
    
    print("Loading chartevents data (filtered)...")
    chartevents_filters = {
        'stay_id': cohort_stay_ids,
        'itemid': vital_itemids
    }
    chartevents_df = load_filtered_events(CHARTEVENTS_FILE, usecols=chartevents_usecols, filter_dict=chartevents_filters)
    chartevents_df['value'] = pd.to_numeric(chartevents_df['value'], errors='coerce')
    
    print("Loading labevents data (filtered)...")
    labevents_filters = {
        'hadm_id': cohort_hadm_ids,
        'itemid': lab_itemids
    }
    labevents_df = load_filtered_events(LABEVENTS_FILE, usecols=labevents_usecols, filter_dict=labevents_filters)
    labevents_df['value'] = pd.to_numeric(labevents_df['value'], errors='coerce')
    
    print("Extracting time-series data (first 48 hours)...")
    ts_data = extract_time_series(cohort_df, chartevents_df, labevents_df, window_hours=48)
    print("Time-series data shape:", ts_data.shape)
    
    output_file = os.path.join(OUTPUT_DIR, "time_series_features.csv")
    ts_data.to_csv(output_file, index=True)
    print(f"Time-series features saved to: {output_file}")

if __name__ == "__main__":
    main()
