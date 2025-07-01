# ---------------------------------------------------------------------
#  FedFNN-ERL – Feature enhancement module
#  Copyright (c) 2025  Po-Kang Tsai
#  Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------------

import pandas as pd
import numpy as np

# ---------- Example feature enhancement ------------------------------------------------

def enhance_with_trends(df: pd.DataFrame, lab_trends: pd.DataFrame) -> pd.DataFrame:
    """Join lab trends (slopes) to the original patient-level data."""
    df = df.merge(lab_trends, on="hadm_id", how="left")
    df.fillna(0, inplace=True)
    return df

def enhance_with_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically-relevant derived features, e.g., age, BMI, etc."""
    df["age_at_admission"] = (df["admittime"] - df["dob"]).dt.days // 365
    df["bmi"] = df["weight"] / (df["height"]**2)  # BMI formula
    df.fillna(0, inplace=True)
    return df
