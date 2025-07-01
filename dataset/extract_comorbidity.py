import os
import pandas as pd
import numpy as np

# -------- Configuration --------
DATASET_DIR = r"C:\Graduation Project\dataset"
COHORT_FILE = os.path.join(DATASET_DIR, "cohort.csv")
STATIC_FEATURES_FILE = os.path.join(DATASET_DIR, "static_features.csv")
DIAGNOSES_ICD_FILE = os.path.join(DATASET_DIR, "mimic-iv", "hosp", "diagnoses_icd.csv.gz")
# You may include d_icd_diagnoses.csv if needed for more advanced mapping, but here we use a built-in mapping.
OUTPUT_FILE = os.path.join(DATASET_DIR, "static_features_with_comorbidities.csv")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# -------- Utility Functions --------
def load_compressed_csv(filepath):
    """Load a CSV file that might be compressed (gzip)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, compression='infer')

# -------- Charlson Comorbidity Extraction --------
def extract_charlson_comorbidities(diag_icd_path, cohort_df):
    """
    Extract Charlson Comorbidity flags from the MIMIC-IV diagnoses_icd file using an expanded ICD-9 mapping.
    
    Parameters:
      diag_icd_path: Path to diagnoses_icd.csv.gz (expects header: subject_id, hadm_id, seq_num, icd_code, icd_version)
      cohort_df: DataFrame containing at least 'hadm_id' from the full cohort.
      
    Returns:
      comorb_df: DataFrame with hadm_id and binary flags for each Charlson comorbidity category.
    """
    # Load diagnoses data
    diag_icd = load_compressed_csv(diag_icd_path)
    diag_icd['icd_code'] = diag_icd['icd_code'].astype(str)
    
    # Filter diagnoses to those admissions in the cohort
    cohort_hadm_ids = cohort_df['hadm_id'].unique()
    diag_icd = diag_icd[diag_icd['hadm_id'].isin(cohort_hadm_ids)]
    
    # Expanded Charlson mapping dictionary (based on Quan et al., 2005 and other literature)
    expanded_charlson_mapping = {
        'Myocardial_Infarction': ['410', '412'],
        'Congestive_Heart_Failure': ['428'],
        'Peripheral_Vascular_Disease': ['440', '443', '785.4', 'V43.4'],
        'Cerebrovascular_Disease': ['430','431','432','433','434','435','436','437','438'],
        'Dementia': ['290', '294.1', '331.2'],
        'Chronic_Pulmonary_Disease': ['490','491','492','493','494','495','496'],
        'Rheumatologic_Disease': ['710','711','714','725'],
        'Peptic_Ulcer_Disease': ['531','532','533','534'],
        'Mild_Liver_Disease': ['571.2','571.4','573.3'],
        'Diabetes': ['250'],
        'Diabetes_with_Complications': ['250.4','250.5','250.6','250.7'],
        'Hemiplegia': ['342','344.1'],
        'Moderate_to_Severe_Renal_Disease': ['585','586','V42.0'],
        'Any_Malignancy': [str(x) for x in range(140, 172)] + [str(x) for x in range(174, 209)],
        'Moderate_to_Severe_Liver_Disease': ['572.2','572.3','572.4','572.8'],
        'Metastatic_Solid_Tumor': [str(x) for x in range(196, 200)],
        'AIDS': ['042'] 
    }
    
    # Create a DataFrame to hold comorbidity flags for each hadm_id in the cohort
    charlson_categories = list(expanded_charlson_mapping.keys())
    comorb_df = pd.DataFrame(0, index=cohort_hadm_ids, columns=charlson_categories)
    
    # Group the diagnoses by hadm_id for efficiency
    grouped = diag_icd.groupby('hadm_id')
    for hadm_id, group in grouped:
        codes = group['icd_code'].unique()
        for category, prefixes in expanded_charlson_mapping.items():
            # Check if any code in this group starts with one of the prefixes.
            if any(any(code.startswith(prefix) for prefix in prefixes) for code in codes):
                comorb_df.at[hadm_id, category] = 1
    
    comorb_df = comorb_df.reset_index().rename(columns={'index': 'hadm_id'})
    return comorb_df

# -------- Merge Static Features --------
def merge_static_features(cohort_file, static_features_file, diag_icd_path, output_path):
    """
    Merge the preprocessed static features with Charlson comorbidity flags.
    
    Parameters:
      cohort_file: Path to the full cohort CSV.
      static_features_file: Path to the existing static_features.csv (demographics only).
      diag_icd_path: Path to diagnoses_icd.csv.gz.
      output_path: Path to save the merged static features CSV.
      
    Returns:
      merged_df: DataFrame with hadm_id, gender, anchor_age, and comorbidity flags.
    """
    # Load the full cohort and static features
    cohort_df = pd.read_csv(cohort_file)
    static_df = pd.read_csv(static_features_file)
    
    # Check that static_df has hadm_id, gender, anchor_age
    if not set(['hadm_id', 'gender', 'anchor_age']).issubset(static_df.columns):
        raise ValueError("static_features.csv must contain columns: hadm_id, gender, anchor_age")
    
    # Extract comorbidities using the cohort information
    comorb_df = extract_charlson_comorbidities(diag_icd_path, cohort_df)
    
    # Merge static features with comorbidities on hadm_id
    merged_df = pd.merge(static_df, comorb_df, on='hadm_id', how='left')
    
    # Fill missing comorbidity flags with 0
    for col in merged_df.columns:
        if col not in ['hadm_id', 'gender', 'anchor_age']:
            merged_df[col] = merged_df[col].fillna(0)
    
    # Optionally, normalize continuous static features (e.g., anchor_age)
    merged_df['anchor_age'] = (merged_df['anchor_age'] - merged_df['anchor_age'].mean()) / merged_df['anchor_age'].std()
    
    # Save merged static features to CSV
    merged_df.to_csv(output_path, index=False)
    print("Merged static features with comorbidities saved to:", output_path)
    return merged_df

# -------- Main Execution --------
if __name__ == "__main__":
    # Define file paths
    cohort_file = os.path.join(DATASET_DIR, "cohort.csv")
    static_features_file = os.path.join(DATASET_DIR, "static_features.csv")
    diag_icd_path = os.path.join(DATASET_DIR, "mimic-iv", "hosp", "diagnoses_icd.csv.gz")
    output_static_features = os.path.join(DATASET_DIR, "static_features_with_comorbidities.csv")
    
    # Run the merge process
    merged_static_features = merge_static_features(cohort_file, static_features_file, diag_icd_path, output_static_features)
    
    # Display a sample of the merged static features
    print("Sample of merged static features:")
    print(merged_static_features.head())
