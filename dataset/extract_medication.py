# ---------------------------------------------------------------------
#  FedFNN-ERL – Medication extraction utilities
#  Copyright (c) 2025  Po-Kang Tsai
#  Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------------

from __future__ import annotations
import pandas as pd

# ---------- Medication extraction helpers ----------------------------------------

def extract_medication_info(df: pd.DataFrame, meds_of_interest: list[int]) -> pd.DataFrame:
    """Extract medication info from MIMIC-IV dataset."""
    med_df = df[df["itemid"].isin(meds_of_interest)]
    return med_df


def process_medication_data(med_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess medication data (e.g., adjust for duplicates or anomalies)."""
    med_df["dose"] = med_df["dose"] / med_df["doseunit"].apply(lambda x: 1000 if x == "mg" else 1)
    med_df["dose"] = med_df["dose"].round(2)  # rounding doses for simplicity
    med_df.fillna(0, inplace=True)
    return med_df
