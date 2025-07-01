# ---------------------------------------------------------------------
#  FedFNN-ERL – Comorbidity extraction utilities
#  Copyright (c) 2025  Po-Kang Tsai
#  Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------------

from __future__ import annotations
import pandas as pd

# ---------- Extract comorbidities helper ---------------------------------------

def extract_comorbidities(df: pd.DataFrame, icd_codes: list[str]) -> pd.DataFrame:
    """Extract comorbidity features based on ICD-9/ICD-10 codes."""
    comorbidities = df[df["icd_code"].isin(icd_codes)]
    return comorbidities


def process_comorbidities(comorb_df: pd.DataFrame) -> pd.DataFrame:
    """Create binary comorbidity flags (1 for presence, 0 for absence)."""
    comorb_df["comorbidity_flag"] = 1
    comorb_df = comorb_df.drop_duplicates(subset=["hadm_id", "icd_code"])
    return comorb_df.pivot_table(index="hadm_id", columns="icd_code", values="comorbidity_flag", fill_value=0)
